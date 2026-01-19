package coop.rchain.node

import coop.rchain.casper.{ReportStore, ReportingCasper}
import coop.rchain.casper.api.BlockReportAPI
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.protocol.{BlockInfo, DeployInfo, LightBlockInfo, TransferInfo}
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.casper.util.GenesisBuilder.{buildGenesis, GenesisContext}
import coop.rchain.casper.util.rholang.Resources
import coop.rchain.crypto.PrivateKey
import coop.rchain.crypto.signatures.Secp256k1
import coop.rchain.models.Par
import coop.rchain.node.web.{
  BlockInfoEnricher,
  PreCharge,
  Refund,
  Transaction,
  TransactionResponse,
  UserDeploy
}
import coop.rchain.rholang.interpreter.util.RevAddress
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.syntax.rspaceSyntaxKeyValueStoreManager
import coop.rchain.casper.PrettyPrinter
import monix.eval.Task
import org.scalatest.{FlatSpec, Inspectors, Matchers}
import monix.execution.Scheduler.Implicits.global
import org.scalatest._
class TransactionAPISpec extends FlatSpec with Matchers with Inspectors {
  val genesis: GenesisContext = buildGenesis()

  def checkTransactionAPI(term: String, phloLimit: Long, phloPrice: Long, deployKey: PrivateKey) =
    TestNode.networkEff(genesis, networkSize = 1, withReadOnlySize = 1).use { nodes =>
      val validator = nodes(0)
      val readonly  = nodes(1)
      import readonly._
      for {
        kvm             <- Resources.mkTestRNodeStoreManager[Task](readonly.dataDir)
        rspaceStore     <- kvm.rSpaceStores
        reportingCasper = ReportingCasper.rhoReporter[Task](rspaceStore)
        reportingStore  <- ReportStore.store[Task](kvm)
        blockReportAPI  = BlockReportAPI[Task](reportingCasper, reportingStore)
        deploy <- ConstructDeploy.sourceDeployNowF(
                   term,
                   sec = deployKey,
                   phloLimit = phloLimit,
                   phloPrice = phloPrice,
                   shardId = this.genesis.genesisBlock.shardId
                 )
        transactionAPI = Transaction[Task](
          blockReportAPI,
          Par(unforgeables = Seq(Transaction.transferUnforgeable))
        )
        transferBlock <- validator.addBlock(deploy)
        _             <- readonly.processBlock(transferBlock)
        transactions <- transactionAPI
                         .getTransaction(Blake2b256Hash.fromByteString(transferBlock.blockHash))

      } yield (transactions, transferBlock)
    }

  "transfer rev" should "be gotten in transaction api" in {
    val fromSk      = genesis.genesisVaultsSks.head
    val fromAddr    = RevAddress.fromPublicKey(Secp256k1.toPublic(fromSk)).get.toBase58
    val toPk        = genesis.genesisVaultsSks.last
    val toAddr      = RevAddress.fromPublicKey(Secp256k1.toPublic(toPk)).get.toBase58
    val amount      = 1L
    val phloPrice   = 1L
    val phloLimit   = 3000000L
    val transferRho = s"""
         #new rl(`rho:registry:lookup`), RevVaultCh, vaultCh, toVaultCh, deployerId(`rho:rchain:deployerId`), revVaultKeyCh, resultCh in {
         #  rl!(`rho:rchain:revVault`, *RevVaultCh) |
         #  for (@(_, RevVault) <- RevVaultCh) {
         #    @RevVault!("findOrCreate", "$fromAddr", *vaultCh) |
         #    @RevVault!("findOrCreate", "$toAddr", *toVaultCh) |
         #    @RevVault!("deployerAuthKey", *deployerId, *revVaultKeyCh) |
         #    for (@(true, vault) <- vaultCh; key <- revVaultKeyCh; @(true, toVault) <- toVaultCh) {
         #      @vault!("transfer", "$toAddr", $amount, *key, *resultCh) |
         #      for (_ <- resultCh) { Nil }
         #    }
         #  }
         #}""".stripMargin('#')
    (for {
      result                        <- checkTransactionAPI(transferRho, phloLimit, phloPrice, fromSk)
      (transactions, transferBlock) = result
      _                             = transactions.length should be(3)
      _ = transactions.foreach { t =>
        t.transactionType match {
          case UserDeploy(_) =>
            t.transaction.fromAddr should be(fromAddr)
            t.transaction.toAddr should be(toAddr)
            t.transaction.amount should be(amount)
            t.transaction.failReason should be(None)

          case PreCharge(_) =>
            t.transaction.fromAddr should be(fromAddr)
            t.transaction.amount should be(phloLimit * phloPrice)
            t.transaction.failReason should be(None)

          case Refund(_) =>
            t.transaction.toAddr should be(fromAddr)
            t.transaction.amount should be(
              phloLimit * phloPrice - transferBlock.body.deploys.head.cost.cost
            )
            t.transaction.failReason should be(None)
          case _ => ()
        }
      }
    } yield ()).runSyncUnsafe()
  }

  "no user deploy log" should "return only precharge and refund transaction" in {
    val fromSk    = genesis.genesisVaultsSks.head
    val fromAddr  = RevAddress.fromPublicKey(Secp256k1.toPublic(fromSk)).get.toBase58
    val phloPrice = 1L
    val phloLimit = 3000000L
    val deployRho = s"""new a in {}"""
    (for {
      result                <- checkTransactionAPI(deployRho, phloLimit, phloPrice, fromSk)
      (transactions, block) = result
      _                     = transactions.length should be(2)
      _ = transactions.foreach { t =>
        t.transactionType match {
          case PreCharge(_) =>
            t.transaction.fromAddr should be(fromAddr)
            t.transaction.amount should be(phloLimit * phloPrice)
            t.transaction.failReason should be(None)

          case Refund(_) =>
            t.transaction.toAddr should be(fromAddr)
            t.transaction.amount should be(
              phloLimit * phloPrice - block.body.deploys.head.cost.cost
            )
            t.transaction.failReason should be(None)
          case _ => ()
        }
      }

    } yield ()).runSyncUnsafe()
  }

  "preCharge failed case" should "return 1 preCharge transaction" in {
    val fromSk    = genesis.genesisVaultsSks.head
    val fromAddr  = RevAddress.fromPublicKey(Secp256k1.toPublic(fromSk)).get.toBase58
    val phloPrice = 1L
    val phloLimit = 300000000000L
    val deployRho = s"""new a in {}"""
    val (transaction, block) = (for {
      result                <- checkTransactionAPI(deployRho, phloLimit, phloPrice, fromSk)
      (transactions, block) = result
      _                     = transactions.length should be(1)
      t                     = transactions.head

      _ = t.transaction.failReason should be(Some("Insufficient funds"))

    } yield (t, block)).runSyncUnsafe()
    transaction.transactionType shouldBe a[PreCharge]
    transaction.transaction.fromAddr shouldBe fromAddr
    transaction.transaction.amount shouldBe phloLimit * phloPrice - block.body.deploys.head.cost.cost
    transaction.transaction.failReason shouldBe Some("Insufficient funds")
  }

  "BlockInfoEnricher" should "map transactions to transfers correctly" in {
    val fromSk      = genesis.genesisVaultsSks.head
    val fromAddr    = RevAddress.fromPublicKey(Secp256k1.toPublic(fromSk)).get.toBase58
    val toPk        = genesis.genesisVaultsSks.last
    val toAddr      = RevAddress.fromPublicKey(Secp256k1.toPublic(toPk)).get.toBase58
    val amount      = 1L
    val phloPrice   = 1L
    val phloLimit   = 3000000L
    val transferRho = s"""
         #new rl(`rho:registry:lookup`), RevVaultCh, vaultCh, toVaultCh, deployerId(`rho:rchain:deployerId`), revVaultKeyCh, resultCh in {
         #  rl!(`rho:rchain:revVault`, *RevVaultCh) |
         #  for (@(_, RevVault) <- RevVaultCh) {
         #    @RevVault!("findOrCreate", "$fromAddr", *vaultCh) |
         #    @RevVault!("findOrCreate", "$toAddr", *toVaultCh) |
         #    @RevVault!("deployerAuthKey", *deployerId, *revVaultKeyCh) |
         #    for (@(true, vault) <- vaultCh; key <- revVaultKeyCh; @(true, toVault) <- toVaultCh) {
         #      @vault!("transfer", "$toAddr", $amount, *key, *resultCh) |
         #      for (_ <- resultCh) { Nil }
         #    }
         #  }
         #}""".stripMargin('#')
    (for {
      result                        <- checkTransactionAPI(transferRho, phloLimit, phloPrice, fromSk)
      (transactions, transferBlock) = result

      // Create TransactionResponse from transactions
      transactionResponse = TransactionResponse(transactions)

      // Use BlockInfoEnricher to map transactions to transfers
      transfersByDeploy = BlockInfoEnricher.mapTransactionsToTransfers(transactionResponse)

      // Verify that user deploy has transfers mapped
      userDeployId = transactions.collectFirst {
        case t if t.transactionType.isInstanceOf[UserDeploy] =>
          t.transactionType.asInstanceOf[UserDeploy].deployId
      }.get
      _ = transfersByDeploy.contains(userDeployId) should be(true)

      transfers = transfersByDeploy(userDeployId)
      _         = transfers.length should be(1)
      transfer  = transfers.head
      _         = transfer.fromAddr should be(fromAddr)
      _         = transfer.toAddr should be(toAddr)
      _         = transfer.amount should be(amount)
      _         = transfer.success should be(true)
      _         = transfer.failReason should be("")
    } yield ()).runSyncUnsafe()
  }

  it should "enrich BlockInfo with transfer data" in {
    val fromSk      = genesis.genesisVaultsSks.head
    val fromAddr    = RevAddress.fromPublicKey(Secp256k1.toPublic(fromSk)).get.toBase58
    val toPk        = genesis.genesisVaultsSks.last
    val toAddr      = RevAddress.fromPublicKey(Secp256k1.toPublic(toPk)).get.toBase58
    val amount      = 1L
    val phloPrice   = 1L
    val phloLimit   = 3000000L
    val transferRho = s"""
         #new rl(`rho:registry:lookup`), RevVaultCh, vaultCh, toVaultCh, deployerId(`rho:rchain:deployerId`), revVaultKeyCh, resultCh in {
         #  rl!(`rho:rchain:revVault`, *RevVaultCh) |
         #  for (@(_, RevVault) <- RevVaultCh) {
         #    @RevVault!("findOrCreate", "$fromAddr", *vaultCh) |
         #    @RevVault!("findOrCreate", "$toAddr", *toVaultCh) |
         #    @RevVault!("deployerAuthKey", *deployerId, *revVaultKeyCh) |
         #    for (@(true, vault) <- vaultCh; key <- revVaultKeyCh; @(true, toVault) <- toVaultCh) {
         #      @vault!("transfer", "$toAddr", $amount, *key, *resultCh) |
         #      for (_ <- resultCh) { Nil }
         #    }
         #  }
         #}""".stripMargin('#')
    (for {
      result                        <- checkTransactionAPI(transferRho, phloLimit, phloPrice, fromSk)
      (transactions, transferBlock) = result

      // Create a mock BlockInfo with the deploy from our transfer block
      deployInfo = transferBlock.body.deploys.head.toDeployInfo
      lightBlockInfo = LightBlockInfo(
        blockHash = PrettyPrinter.buildStringNoLimit(transferBlock.blockHash)
      )
      blockInfo = BlockInfo(blockInfo = lightBlockInfo, deploys = Seq(deployInfo))

      // Create TransactionResponse and enrich BlockInfo
      transactionResponse = TransactionResponse(transactions)
      enrichedBlockInfo   = BlockInfoEnricher.enrichBlockInfo(blockInfo, transactionResponse)

      // Verify the enriched BlockInfo has transfers
      enrichedDeploy = enrichedBlockInfo.deploys.head
      _              = enrichedDeploy.transfers.length should be(1)
      transfer       = enrichedDeploy.transfers.head
      _              = transfer.fromAddr should be(fromAddr)
      _              = transfer.toAddr should be(toAddr)
      _              = transfer.amount should be(amount)
      _              = transfer.success should be(true)
      _              = transfer.failReason should be("")
    } yield ()).runSyncUnsafe()
  }

  it should "return empty transfers when no user transfers exist" in {
    val fromSk    = genesis.genesisVaultsSks.head
    val phloPrice = 1L
    val phloLimit = 3000000L
    val deployRho = s"""new a in {}"""
    (for {
      result                <- checkTransactionAPI(deployRho, phloLimit, phloPrice, fromSk)
      (transactions, block) = result

      // Create TransactionResponse - should have PreCharge and Refund but no UserDeploy transfers
      transactionResponse = TransactionResponse(transactions)

      // Map transactions to transfers - should have no user deploy entries
      transfersByDeploy = BlockInfoEnricher.mapTransactionsToTransfers(transactionResponse)

      // Only UserDeploy transactions get mapped, so for a deploy with no user transfers,
      // there should still be an entry but with an empty transfer (the deploy itself had no REV transfer)
      // Actually, for a non-transfer deploy, there's no UserDeploy transaction at all with transfer data
      // Let's verify the structure
      _ = transactions.length should be(2) // PreCharge and Refund only

      // Create mock BlockInfo and enrich
      deployInfo = block.body.deploys.head.toDeployInfo
      lightBlockInfo = LightBlockInfo(
        blockHash = PrettyPrinter.buildStringNoLimit(block.blockHash)
      )
      blockInfo         = BlockInfo(blockInfo = lightBlockInfo, deploys = Seq(deployInfo))
      enrichedBlockInfo = BlockInfoEnricher.enrichBlockInfo(blockInfo, transactionResponse)

      // Verify no transfers in enriched BlockInfo
      enrichedDeploy = enrichedBlockInfo.deploys.head
      _              = enrichedDeploy.transfers.length should be(0)
    } yield ()).runSyncUnsafe()
  }
}
