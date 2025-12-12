package coop.rchain.casper.blocks.proposer

import cats.effect.{Concurrent, Sync}
import cats.syntax.all._
import cats.instances.list._
import com.google.protobuf.ByteString
import coop.rchain.blockstorage.BlockStore
import coop.rchain.blockstorage.deploy.DeployStorage
import coop.rchain.blockstorage.syntax._
import coop.rchain.casper.protocol.{Header, _}
import coop.rchain.casper.util.{ConstructDeploy, ProtoUtil}
import coop.rchain.casper.util.rholang.RuntimeManager.StateHash
import coop.rchain.casper.util.rholang._
import coop.rchain.casper.util.rholang.costacc.{CloseBlockDeploy, SlashDeploy}
import coop.rchain.casper.{CasperSnapshot, PrettyPrinter, ValidatorIdentity}
import coop.rchain.crypto.PrivateKey
import coop.rchain.crypto.signatures.Signed
import coop.rchain.metrics.{Metrics, Span}
import coop.rchain.models.BlockHash.BlockHash
import coop.rchain.rholang.interpreter.SystemProcesses.BlockData
import coop.rchain.shared.{Base16, Log, Stopwatch, Time}

object BlockCreator {
  private[this] val ProcessDeploysAndCreateBlockMetricsSource =
    Metrics.Source(Metrics.BaseSource, "create-block")

  /*
   * Overview of createBlock
   *
   *  1. Rank each of the block cs's latest messages (blocks) via the LMD GHOST estimator.
   *  2. Let each latest message have a score of 2^(-i) where i is the index of that latest message in the ranking.
   *     Take a subset S of the latest messages such that the sum of scores is the greatest and
   *     none of the blocks in S conflicts with each other. S will become the parents of the
   *     about-to-be-created block.
   *  3. Extract all valid deploys that aren't already in all ancestors of S (the parents).
   *  4. Create a new block that contains the deploys from the previous step.
   */
  def create[F[_]: Concurrent: Log: Time: BlockStore: DeployStorage: Metrics: RuntimeManager: Span](
      s: CasperSnapshot[F],
      validatorIdentity: ValidatorIdentity,
      dummyDeployOpt: Option[(PrivateKey, String)] = None,
      allowEmptyBlocks: Boolean = false
  )(implicit runtimeManager: RuntimeManager[F]): F[BlockCreatorResult] =
    Span[F].trace(ProcessDeploysAndCreateBlockMetricsSource) {
      val selfId         = ByteString.copyFrom(validatorIdentity.publicKey.bytes)
      val nextSeqNum     = s.maxSeqNums(selfId) + 1
      val nextBlockNum   = s.maxBlockNum + 1
      val parents        = s.parents
      val justifications = s.justifications

      def prepareUserDeploys(blockNumber: Long): F[Set[Signed[DeployData]]] =
        for {
          unfinalized         <- DeployStorage[F].readAll
          earliestBlockNumber = blockNumber - s.onChainState.shardConf.deployLifespan

          // Categorize deploys for logging
          futureDeploys  = unfinalized.filter(d => !notFutureDeploy(blockNumber, d.data))
          expiredDeploys = unfinalized.filter(d => !notExpiredDeploy(earliestBlockNumber, d.data))

          valid = unfinalized.filter(
            d =>
              notFutureDeploy(blockNumber, d.data) &&
                notExpiredDeploy(earliestBlockNumber, d.data)
          )
          // this is required to prevent resending the same deploy several times by validator
          validUnique    = valid -- s.deploysInScope
          alreadyInScope = valid.intersect(s.deploysInScope)

          // Log deploy selection details when there are any deploys in the pool
          _ <- if (unfinalized.nonEmpty || s.deploysInScope.nonEmpty)
                Log[F].info(
                  s"Deploy selection for block #$blockNumber: " +
                    s"pool=${unfinalized.size}, " +
                    s"future=${futureDeploys.size} (validAfterBlockNumber >= $blockNumber), " +
                    s"expired=${expiredDeploys.size} (validAfterBlockNumber <= $earliestBlockNumber), " +
                    s"valid=${valid.size}, " +
                    s"alreadyInScope=${alreadyInScope.size}, " +
                    s"selected=${validUnique.size}"
                )
              else ().pure[F]

          // Log details for filtered-out deploys (to help debug why deploys aren't included)
          _ <- futureDeploys.toList.traverse_(
                d =>
                  Log[F].warn(
                    s"Deploy ${Base16.encode(d.sig.toByteArray.take(8))}... FILTERED (future): " +
                      s"validAfterBlockNumber=${d.data.validAfterBlockNumber} >= currentBlock=$blockNumber"
                  )
              )
          _ <- expiredDeploys.toList.traverse_(
                d =>
                  Log[F].warn(
                    s"Deploy ${Base16.encode(d.sig.toByteArray.take(8))}... FILTERED (expired): " +
                      s"validAfterBlockNumber=${d.data.validAfterBlockNumber} <= earliestBlock=$earliestBlockNumber"
                  )
              )
          _ <- alreadyInScope.toList.traverse_(
                d =>
                  Log[F].warn(
                    s"Deploy ${Base16.encode(d.sig.toByteArray.take(8))}... FILTERED (already in scope): " +
                      s"deploy already exists in DAG within lifespan window"
                  )
              )
        } yield validUnique

      def prepareSlashingDeploys(seqNum: Int): F[Seq[SlashDeploy]] =
        for {
          ilm <- s.dag.invalidLatestMessages
          // if the node is already not active as per main parent, the node won't slash once more
          ilmFromBonded = ilm.toList.filter {
            case (validator, _) => s.onChainState.bondsMap.getOrElse(validator, 0L) > 0L
          }
          // TODO: Add `slashingDeploys` to DeployStorage
          slashingDeploys = ilmFromBonded
            .map(_._2)
            .map(
              invalidBlockHash =>
                SlashDeploy(
                  invalidBlockHash,
                  validatorIdentity.publicKey,
                  SystemDeployUtil.generateSlashDeployRandomSeed(selfId, seqNum)
                )
            )
          _ <- slashingDeploys.traverse_(
                sd =>
                  Log[F].info(
                    s"Issuing slashing deploy justified by block ${PrettyPrinter.buildString(sd.invalidBlockHash)}"
                  )
              )
        } yield slashingDeploys

      def prepareDummyDeploy(blockNumber: Long, shardId: String): Seq[Signed[DeployData]] =
        dummyDeployOpt match {
          case Some((privateKey, term)) =>
            Seq(
              ConstructDeploy.sourceDeployNow(
                source = term,
                sec = privateKey,
                vabn = blockNumber - 1,
                shardId = shardId
              )
            )
          case None => Seq.empty[Signed[DeployData]]
        }

      val createBlockProcess = for {
        _ <- Log[F].info(
              s"Creating block #${nextBlockNum} (seqNum ${nextSeqNum})"
            )
        shardId         = s.onChainState.shardConf.shardName
        userDeploys     <- prepareUserDeploys(nextBlockNum)
        dummyDeploys    = prepareDummyDeploy(nextBlockNum, shardId)
        slashingDeploys <- prepareSlashingDeploys(nextSeqNum)
        // make sure closeBlock is the last system Deploy
        systemDeploys = slashingDeploys :+ CloseBlockDeploy(
          SystemDeployUtil
            .generateCloseDeployRandomSeed(selfId, nextSeqNum)
        )
        deploys = userDeploys -- s.deploysInScope ++ dummyDeploys
        r <- if (allowEmptyBlocks || deploys.nonEmpty || slashingDeploys.nonEmpty)
              // When allowEmptyBlocks is true (heartbeat enabled), always create blocks.
              // They will always have system deploys (CloseBlockDeploy), making them valid.
              // Empty blocks are necessary for liveness during periods of no user activity.
              // When false (default), use original behavior: only create blocks with user deploys.
              for {
                now           <- Time[F].currentMillis
                invalidBlocks = s.invalidBlocks
                blockData     = BlockData(now, nextBlockNum, validatorIdentity.publicKey, nextSeqNum)
                checkpointData <- InterpreterUtil.computeDeploysCheckpoint(
                                   parents,
                                   deploys.toSeq,
                                   systemDeploys,
                                   s,
                                   runtimeManager,
                                   blockData,
                                   invalidBlocks
                                 )
                (
                  preStateHash,
                  postStateHash,
                  processedDeploys,
                  rejectedDeploys,
                  processedSystemDeploys
                )             = checkpointData
                newBonds      <- runtimeManager.computeBonds(postStateHash)
                _             <- Span[F].mark("before-packing-block")
                casperVersion = s.onChainState.shardConf.casperVersion
                // unsignedBlock got blockHash(hashed without signature)
                unsignedBlock = packageBlock(
                  blockData,
                  parents.map(_.blockHash),
                  justifications.toSeq,
                  preStateHash,
                  postStateHash,
                  processedDeploys,
                  rejectedDeploys,
                  processedSystemDeploys,
                  newBonds,
                  shardId,
                  casperVersion
                )
                _ <- Span[F].mark("block-created")
                // signedBlock add signature and replace hashed-without-signature
                // blockHash to hashed-with-signature blockHash
                signedBlock = validatorIdentity.signBlock(unsignedBlock)
                _           <- Span[F].mark("block-signed")
              } yield BlockCreatorResult.created(signedBlock)
            else
              BlockCreatorResult.noNewDeploys.pure[F]
      } yield r

      for {
        // Create block and measure duration
        r                      <- Stopwatch.duration(createBlockProcess)
        (blockStatus, elapsed) = r
        _ <- blockStatus match {
              case Created(block) =>
                val blockInfo   = PrettyPrinter.buildString(block, short = true)
                val deployCount = block.body.deploys.size
                Log[F].info(s"Block created: $blockInfo (${deployCount}d) [$elapsed]")
              case _ => ().pure[F]
            }
      } yield blockStatus
    }

  private def packageBlock(
      blockData: BlockData,
      parents: Seq[BlockHash],
      justifications: Seq[Justification],
      preStateHash: StateHash,
      postStateHash: StateHash,
      deploys: Seq[ProcessedDeploy],
      rejectedDeploys: Seq[ByteString],
      systemDeploys: Seq[ProcessedSystemDeploy],
      bondsMap: Seq[Bond],
      shardId: String,
      version: Long
  ): BlockMessage = {
    val state = RChainState(preStateHash, postStateHash, bondsMap.toList, blockData.blockNumber)
    val body =
      Body(
        state,
        deploys.toList,
        rejectedDeploys.map(r => RejectedDeploy(r)).toList,
        systemDeploys.toList
      )
    val header = Header(parents.toList, blockData.timeStamp, version)
    ProtoUtil.unsignedBlockProto(body, header, justifications, shardId, blockData.seqNum)
  }

  private def notExpiredDeploy(earliestBlockNumber: Long, d: DeployData): Boolean =
    d.validAfterBlockNumber > earliestBlockNumber

  private def notFutureDeploy(currentBlockNumber: Long, d: DeployData): Boolean =
    d.validAfterBlockNumber < currentBlockNumber
}
