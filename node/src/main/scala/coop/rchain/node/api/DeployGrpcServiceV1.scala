package coop.rchain.node.api

import cats.effect.Concurrent
import cats.effect.concurrent.Ref
import cats.syntax.all._
import cats.{Applicative, Foldable}
import com.google.protobuf.ByteString
import coop.rchain.blockstorage.BlockStore
import coop.rchain.casper.api._
import coop.rchain.casper.engine.EngineCell._
import coop.rchain.casper.protocol._
import java.nio.file.Path
import coop.rchain.casper.protocol.deploy.v1._
import coop.rchain.casper.{ProposeFunction, SafetyOracle}
import coop.rchain.node.configuration.FileUploadConf
import coop.rchain.catscontrib.TaskContrib._
import coop.rchain.comm.discovery.NodeDiscovery
import coop.rchain.comm.rp.Connect.{ConnectionsCell, RPConfAsk}
import coop.rchain.graphz._
import coop.rchain.metrics.{Metrics, Span}
import coop.rchain.models.StacksafeMessage
import coop.rchain.models.syntax._
import coop.rchain.monix.Monixable
import coop.rchain.node.web.{
  BlockInfoEnricher,
  CacheTransactionAPI,
  Transaction,
  TransactionResponse
}
import coop.rchain.shared.Log
import coop.rchain.shared.ThrowableOps._
import coop.rchain.shared.syntax._
import monix.eval.Task
import monix.execution.Scheduler
import monix.reactive.Observable

object DeployGrpcServiceV1 {

  def apply[F[_]: Monixable: Concurrent: Log: SafetyOracle: BlockStore: Span: EngineCell: RPConfAsk: ConnectionsCell: NodeDiscovery: Metrics](
      apiMaxBlocksLimit: Int,
      blockReportAPI: BlockReportAPI[F],
      cacheTransactionAPI: CacheTransactionAPI[F],
      transactionStore: Transaction.TransactionStore[F],
      triggerProposeF: Option[ProposeFunction[F]],
      devMode: Boolean = false,
      networkId: String,
      shardId: String,
      minPhloPrice: Long,
      isNodeReadOnly: Boolean,
      uploadDir: Path,
      fileUploadConf: FileUploadConf
  )(
      implicit worker: Scheduler
  ): DeployServiceV1GrpcMonix.DeployService =
    new DeployServiceV1GrpcMonix.DeployService {

      private def defer[A, R <: StacksafeMessage[R]](
          task: F[Either[String, A]]
      )(
          response: Either[ServiceError, A] => R
      ): Task[R] =
        task.toTask
          .executeOn(worker)
          .fromTask
          .logOnError("Deploy service method error.")
          .attempt
          .map(
            _.fold(
              t => response(ServiceError(t.toMessageList()).asLeft),
              r => response(r.leftMap(e => ServiceError(Seq(e))))
            )
          )
          .toTask

      private def deferCollection[A, R <: StacksafeMessage[R], Collection[_]: Applicative: Foldable](
          task: F[Collection[A]]
      )(
          response: Either[ServiceError, A] => R
      ): Task[Collection[R]] =
        task.toTask
          .executeOn(worker)
          .fromTask
          .logOnError("Deploy service method error.")
          .attempt
          .map(
            _.fold(
              t => response(ServiceError(t.toMessageList()).asLeft).pure[Collection],
              _.map(r => response(r.asRight[ServiceError]))
            )
          )
          .toTask

      def doDeploy(request: DeployDataProto): Task[DeployResponse] =
        DeployData
          .from(request)
          .fold(
            errMsg => {
              import DeployResponse.Message._
              Task({
                val error = ServiceError(Seq[String](errMsg))
                DeployResponse(Error(error))
              })
            },
            dd => {
              defer(BlockAPI.deploy[F](dd, triggerProposeF, minPhloPrice, isNodeReadOnly, shardId)) {
                r =>
                  import DeployResponse.Message
                  import DeployResponse.Message._
                  DeployResponse(r.fold[Message](Error, Result))
              }
            }
          )

      /**
        * Enriches a BlockInfo with transfer data.
        * If cached, uses cached data. If not cached, waits for extraction to complete.
        */
      private def enrichWithTransfers(
          blockInfo: coop.rchain.casper.protocol.BlockInfo
      ): F[coop.rchain.casper.protocol.BlockInfo] = {
        val blockHash = blockInfo.blockInfo.blockHash
        for {
          cachedOpt <- transactionStore.get1(blockHash)
          txResponse <- cachedOpt match {
                         case Some(cached) => cached.pure[F]
                         case None         => cacheTransactionAPI.getTransaction(blockHash)
                       }
        } yield BlockInfoEnricher.enrichBlockInfo(blockInfo, txResponse)
      }

      def getBlock(request: BlockQuery): Task[BlockResponse] =
        defer(
          BlockAPI.getBlock[F](request.hash).flatMap {
            case Right(bi) =>
              enrichWithTransfers(bi).map(
                enriched => Right(enriched): Either[String, coop.rchain.casper.protocol.BlockInfo]
              )
            case Left(e) =>
              Concurrent[F].pure(Left(e): Either[String, coop.rchain.casper.protocol.BlockInfo])
          }
        ) { r =>
          import BlockResponse.Message
          import BlockResponse.Message._
          BlockResponse(r.fold[Message](Error, BlockInfo))
        }

      def visualizeDag(request: VisualizeDagQuery): Observable[VisualizeBlocksResponse] = {
        val depth            = if (request.depth <= 0) apiMaxBlocksLimit else request.depth
        val config           = GraphConfig(request.showJustificationLines)
        val startBlockNumber = request.startBlockNumber

        Observable
          .fromTask(
            deferCollection(
              for {
                ref <- Ref[F].of(Vector[String]())
                ser = new ListSerializer(ref)
                res <- BlockAPI
                        .visualizeDag[F, Vector[String]](
                          depth,
                          apiMaxBlocksLimit,
                          startBlockNumber,
                          (ts, lfb) => GraphzGenerator.dagAsCluster[F](ts, lfb, config, ser),
                          ref.get
                        )
                        .map(x => x.getOrElse(Vector.empty[String]))
              } yield res
            ) { r =>
              import VisualizeBlocksResponse.Message
              import VisualizeBlocksResponse.Message._
              VisualizeBlocksResponse(r.fold[Message](Error, Content))
            }
          )
          .flatMap(Observable.fromIterable)
      }

      def machineVerifiableDag(request: MachineVerifyQuery): Task[MachineVerifyResponse] =
        defer(BlockAPI.machineVerifiableDag[F](apiMaxBlocksLimit, apiMaxBlocksLimit)) { r =>
          import MachineVerifyResponse.Message
          import MachineVerifyResponse.Message._
          MachineVerifyResponse(r.fold[Message](Error, Content))
        }

      def showMainChain(request: BlocksQuery): Observable[BlockInfoResponse] =
        Observable
          .fromTask(
            deferCollection(BlockAPI.showMainChain[F](request.depth, apiMaxBlocksLimit)) { r =>
              import BlockInfoResponse.Message
              import BlockInfoResponse.Message._
              BlockInfoResponse(r.fold[Message](Error, BlockInfo))
            }
          )
          .flatMap(Observable.fromIterable)

      def getBlocks(request: BlocksQuery): Observable[BlockInfoResponse] =
        Observable
          .fromTask(
            deferCollection(
              BlockAPI
                .getBlocks[F](request.depth, apiMaxBlocksLimit)
                .map(_.getOrElse(List.empty[LightBlockInfo]))
            ) { r =>
              import BlockInfoResponse.Message
              import BlockInfoResponse.Message._
              BlockInfoResponse(r.fold[Message](Error, BlockInfo))
            }
          )
          .flatMap(Observable.fromIterable)

      def listenForDataAtName(request: DataAtNameQuery): Task[ListeningNameDataResponse] =
        defer(
          BlockAPI.getListeningNameDataResponse[F](request.depth, request.name, apiMaxBlocksLimit)
        ) { r =>
          import ListeningNameDataResponse.Message
          import ListeningNameDataResponse.Message._
          ListeningNameDataResponse(
            r.fold[Message](
              Error, { case (br, l) => Payload(ListeningNameDataPayload(br, l)) }
            )
          )
        }

      def getDataAtName(request: DataAtNameByBlockQuery): Task[RhoDataResponse] =
        defer(BlockAPI.getDataAtPar[F](request.par, request.blockHash, request.usePreStateHash)) {
          r =>
            import RhoDataResponse.Message
            import RhoDataResponse.Message._
            RhoDataResponse(
              r.fold[Message](
                Error, { case (par, block) => Payload(RhoDataPayload(par, block)) }
              )
            )
        }

      def listenForContinuationAtName(
          request: ContinuationAtNameQuery
      ): Task[ContinuationAtNameResponse] =
        defer(
          BlockAPI
            .getListeningNameContinuationResponse[F](
              request.depth,
              request.names,
              apiMaxBlocksLimit
            )
        ) { r =>
          import ContinuationAtNameResponse.Message
          import ContinuationAtNameResponse.Message._
          ContinuationAtNameResponse(
            r.fold[Message](
              Error, { case (br, l) => Payload(ContinuationAtNamePayload(br, l)) }
            )
          )
        }

      def findDeploy(request: FindDeployQuery): Task[FindDeployResponse] =
        defer(BlockAPI.findDeploy[F](request.deployId)) { r =>
          import FindDeployResponse.Message
          import FindDeployResponse.Message._
          FindDeployResponse(r.fold[Message](Error, BlockInfo))
        }

      def previewPrivateNames(request: PrivateNamePreviewQuery): Task[PrivateNamePreviewResponse] =
        defer(
          BlockAPI
            .previewPrivateNames[F](request.user, request.timestamp, request.nameQty)
        ) { r =>
          import PrivateNamePreviewResponse.Message
          import PrivateNamePreviewResponse.Message._
          PrivateNamePreviewResponse(
            r.fold[Message](
              Error,
              ids => Payload(PrivateNamePreviewPayload(ids))
            )
          )
        }

      def lastFinalizedBlock(request: LastFinalizedBlockQuery): Task[LastFinalizedBlockResponse] =
        defer(
          BlockAPI.lastFinalizedBlock[F].flatMap {
            case Right(bi) =>
              enrichWithTransfers(bi).map(
                enriched => Right(enriched): Either[String, coop.rchain.casper.protocol.BlockInfo]
              )
            case Left(e) =>
              Concurrent[F].pure(Left(e): Either[String, coop.rchain.casper.protocol.BlockInfo])
          }
        ) { r =>
          import LastFinalizedBlockResponse.Message
          import LastFinalizedBlockResponse.Message._
          LastFinalizedBlockResponse(r.fold[Message](Error, BlockInfo))
        }

      def isFinalized(request: IsFinalizedQuery): Task[IsFinalizedResponse] =
        defer(BlockAPI.isFinalized[F](request.hash)) { r =>
          import IsFinalizedResponse.Message
          import IsFinalizedResponse.Message._
          IsFinalizedResponse(r.fold[Message](Error, IsFinalized))
        }

      def bondStatus(request: BondStatusQuery): Task[BondStatusResponse] =
        defer(BlockAPI.bondStatus[F](request.publicKey)) { r =>
          import BondStatusResponse.Message
          import BondStatusResponse.Message._
          BondStatusResponse(r.fold[Message](Error, IsBonded))
        }

      def exploratoryDeploy(request: ExploratoryDeployQuery): Task[ExploratoryDeployResponse] =
        defer(
          BlockAPI
            .exploratoryDeploy[F](
              request.term,
              if (request.blockHash.isEmpty) none[String] else Some(request.blockHash),
              request.usePreStateHash,
              devMode
            )
        ) { r =>
          import ExploratoryDeployResponse.Message
          import ExploratoryDeployResponse.Message._
          ExploratoryDeployResponse(r.fold[Message](Error, {
            case (par, block) => Result(DataWithBlockInfo(par, block))
          }))
        }

      override def getEventByHash(request: ReportQuery): Task[EventInfoResponse] =
        defer(
          request.hash.decodeHex
            .fold(s"Request hash: ${request.hash} is not valid hex string".asLeft[Array[Byte]])(
              Right(_)
            )
            .flatTraverse(
              hash =>
                blockReportAPI.blockReport(
                  ByteString.copyFrom(hash),
                  request.forceReplay
                )
            )
        ) { r =>
          import EventInfoResponse.Message
          import EventInfoResponse.Message._
          EventInfoResponse(r.fold[Message](Error, Result))
        }

      def getBlocksByHeights(request: BlocksQueryByHeight): Observable[BlockInfoResponse] =
        Observable
          .fromTask(
            deferCollection(
              BlockAPI
                .getBlocksByHeights[F](
                  request.startBlockNumber,
                  request.endBlockNumber,
                  apiMaxBlocksLimit
                )
                .map(_.getOrElse(List.empty[LightBlockInfo]))
            ) { r =>
              import BlockInfoResponse.Message
              import BlockInfoResponse.Message._
              BlockInfoResponse(r.fold[Message](Error, BlockInfo))
            }
          )
          .flatMap(Observable.fromIterable)

      def status(request: com.google.protobuf.empty.Empty): Task[StatusResponse] =
        (for {
          address <- RPConfAsk[F].ask
          peers   <- ConnectionsCell[F].read
          nodes   <- NodeDiscovery[F].peers
        } yield {
          // Create a set of connected peer IDs for quick lookup
          val connectedIds = peers.map(_.id.key).toSet

          // Convert PeerNode to PeerInfo protobuf message
          def peerNodeToProto(
              peerNode: coop.rchain.comm.PeerNode,
              isConnected: Boolean
          ): coop.rchain.casper.protocol.PeerInfo =
            coop.rchain.casper.protocol.PeerInfo(
              address = peerNode.toAddress,
              nodeId = peerNode.id.toString,
              host = peerNode.endpoint.host,
              protocolPort = peerNode.endpoint.tcpPort,
              discoveryPort = peerNode.endpoint.udpPort,
              isConnected = isConnected
            )

          // Combine discovered peers and active connections with deduplication
          val combinedPeers = nodes
            .map(node => node.id.key -> peerNodeToProto(node, connectedIds.contains(node.id.key)))
            .toMap
            .values
            .toSeq

          val status = Status(
            version = VersionInfo(api = 1.toString, node = coop.rchain.node.web.VersionInfo.get),
            address.local.toAddress,
            networkId,
            shardId,
            peers.length,
            nodes.length,
            minPhloPrice,
            peerList = combinedPeers
          )
          val response = StatusResponse().withStatus(status)
          response
        }).toTask

      def uploadFile(request: Observable[FileUploadChunk]): Task[FileUploadResponse] =
        FileUploadAPI
          .processFileUpload(
            request,
            shardId,
            minPhloPrice,
            isNodeReadOnly,
            uploadDir,
            fileUploadConf.phloPerStorageByte,
            fileUploadConf.baseRegisterPhlo,
            fileUploadConf.maxFileSize
          )
          .flatMap { output =>
            output.deployProto match {
              case Some(proto) =>
                // Validate client signature — same path as doDeploy
                DeployData.from(proto) match {
                  case Left(sigErr) =>
                    // Sig invalid — clean up saved file
                    val hash = output.result.fileHash
                    Task.delay {
                      java.nio.file.Files.deleteIfExists(uploadDir.resolve(hash))
                      java.nio.file.Files.deleteIfExists(uploadDir.resolve(s"$hash.meta.json"))
                    } *> Task.now(
                      FileUploadResponse(
                        FileUploadResponse.Message.Error(
                          ServiceError(List(s"Invalid deploy signature: $sigErr"))
                        )
                      )
                    )
                  case Right(signed) =>
                    val deployIdHex = signed.sig.toByteArray.map("%02x".format(_)).mkString
                    BlockAPI
                      .deploy[F](signed, triggerProposeF, minPhloPrice, isNodeReadOnly, shardId)
                      .toTask
                      .flatMap {
                        case Right(_) =>
                          val updatedResult = output.result.copy(deployId = deployIdHex)
                          Task.now(
                            FileUploadResponse(FileUploadResponse.Message.Result(updatedResult))
                          )
                        case Left(deployErr) =>
                          // Deploy rejected — clean up saved file
                          val hash = output.result.fileHash
                          Task
                            .delay {
                              java.nio.file.Files.deleteIfExists(uploadDir.resolve(hash))
                              java.nio.file.Files.deleteIfExists(
                                uploadDir.resolve(s"$hash.meta.json")
                              )
                            }
                            .map(
                              _ =>
                                FileUploadResponse(
                                  FileUploadResponse.Message.Error(
                                    ServiceError(List(s"Deploy submission failed: $deployErr"))
                                  )
                                )
                            )
                      }
                }

              case None =>
                Task.now(
                  FileUploadResponse(
                    FileUploadResponse.Message.Error(
                      ServiceError(List("Deploy proto was not constructed"))
                    )
                  )
                )
            }
          }
          .onErrorHandle { t =>
            import coop.rchain.shared.ThrowableOps._
            FileUploadResponse(
              FileUploadResponse.Message.Error(ServiceError(t.toMessageList()))
            )
          }

      // The ScalaPB-generated gRPC Monix trait does not expose request metadata
      // (no ServerCallHandler or Metadata in the method signature), so we cannot
      // extract the client IP here. The rate limiter therefore acts as a GLOBAL
      // concurrent download limit (all clients share the "unknown" key).
      // To implement true per-IP limiting, add a gRPC ServerInterceptor that
      // stores the IP in a Context.Key and read it here.
      def downloadFile(request: FileDownloadRequest): Observable[FileDownloadChunk] =
        FileDownloadAPI.streamFile(
          request,
          isNodeReadOnly,
          uploadDir,
          chunkSize = fileUploadConf.chunkSize.toInt,
          maxConcurrentPerIp = fileUploadConf.maxConcurrentDownloadsPerIp,
          devMode = devMode,
          maxCacheEntries = fileUploadConf.maxDownloadCacheEntries,
          finalizationChecker = checkFileFinalized(devMode)
        )

      /**
        * Build a finalization checker that queries
        * `FileRegistry!("lookup", hash)` on the Last Finalized Block's
        * post-state via `exploratoryDeploy`.
        * A non-Nil result means the file is registered in a finalized block.
        *
        * Defense-in-depth: the hash is re-validated here even though
        * `FileDownloadAPI.streamFile` already checks format, to prevent
        * Rholang code injection if this method is ever called from a
        * different context.
        */
      private def checkFileFinalized(devMode: Boolean): String => Task[Boolean] = {
        val log = org.slf4j.LoggerFactory.getLogger("FileDownloadAPI")
        fileHash: String =>
          require(
            fileHash.matches("^[a-f0-9]{64}$"),
            s"Invalid hash for finalization check: $fileHash"
          )
          BlockAPI
            .exploratoryDeploy[F](
              s"""new return, rl(`rho:registry:lookup`), fileRegistryCh in {
                 |  rl!(`rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt`, *fileRegistryCh) |
                 |  for(@(_, FileRegistry) <- fileRegistryCh) {
                 |    @FileRegistry!("lookup", "$fileHash", *return)
                 |  }
                 |}""".stripMargin,
              none[String], // Use LFB (no specific block hash)
              false,        // Use post-state
              devMode
            )
            .toTask
            .map {
              case Right((pars, _)) =>
                // If the result is non-empty and not Nil, the file is registered
                pars.nonEmpty && pars.exists(p => p != coop.rchain.models.Par())
              case Left(err) =>
                log.error(
                  s"[FileDownloadAPI] Finalization check failed for hash=${fileHash.take(16)}...: $err"
                )
                false
            }
      }
    }
}
