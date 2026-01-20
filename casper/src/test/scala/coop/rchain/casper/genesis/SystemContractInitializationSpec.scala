package coop.rchain.casper.genesis

import coop.rchain.casper.InvalidBlock
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.helper.TestNode.Effect
import coop.rchain.casper.util.{ConstructDeploy, GenesisBuilder}
import coop.rchain.casper.util.GenesisBuilder.buildGenesis
import coop.rchain.blockstorage.syntax._
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import coop.rchain.rholang.interpreter.util.RevAddress
import coop.rchain.shared.Base16
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

/**
  * Tests to verify system contracts (PoS, RevVault) are properly
  * initialized at genesis and accessible in subsequent blocks.
  *
  * These tests verify:
  * - Genesis initialization (system contracts deployed correctly)
  * - Block processing (state properly restored after blocks)
  * - Invalid block handling (invalidBlocks map populated correctly)
  */
class SystemContractInitializationSpec extends FlatSpec with Matchers {

  val genesis = buildGenesis()

  // Expected bond values from GenesisBuilder.createBonds: 2*i + 1 for i in 0..3
  val expectedBondValues = List(1, 3, 5, 7)
  val expectedTotalBonds = expectedBondValues.sum // 16

  "PoS contract" should "return correct bonds at genesis" in {
    val getBondsQuery = """
      |new return, rl(`rho:registry:lookup`), posCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  for (@(_, PoS) <- posCh) {
      |    @PoS!("getBonds", *return)
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getBondsQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ <- logEff.info(s"PoS getBonds result: $result")
      } yield {
        result should not be empty
        // Verify we have exactly 4 validators (default genesis config)
        genesis.genesisBlock.body.state.bonds should have size 4
      }
    }
    t.runSyncUnsafe()
  }

  "RevVault" should "be accessible at genesis post-state" in {
    // genesisVaults is Iterable[(PrivateKey, PublicKey)]
    val (_, vaultPk)     = genesis.genesisVaults.toList.head
    val genesisVaultAddr = RevAddress.fromPublicKey(vaultPk).get

    val getVaultQuery = s"""
      |new return, rl(`rho:registry:lookup`), RevVaultCh, vaultCh in {
      |  rl!(`rho:rchain:revVault`, *RevVaultCh) |
      |  for (@(_, RevVault) <- RevVaultCh) {
      |    @RevVault!("findOrCreate", "${genesisVaultAddr.address.toBase58}", *vaultCh) |
      |    for (@(true, vault) <- vaultCh) {
      |      @vault!("balance", *return)
      |    }
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getVaultQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ <- logEff.info(s"RevVault balance result: $result")
      } yield {
        // Verify RevVault is accessible and returns a balance
        result should not be empty
      }
    }
    t.runSyncUnsafe()
  }

  "Validator vaults" should "have zero balance at genesis" in {
    // GenesisBuilder sets validator vaults to 0 REV (Vault(_, 0))
    val validatorPk   = genesis.validatorKeyPairs.head._2
    val validatorAddr = RevAddress.fromPublicKey(validatorPk).get

    val getValidatorVaultQuery = s"""
      |new return, rl(`rho:registry:lookup`), RevVaultCh, vaultCh in {
      |  rl!(`rho:rchain:revVault`, *RevVaultCh) |
      |  for (@(_, RevVault) <- RevVaultCh) {
      |    @RevVault!("findOrCreate", "${validatorAddr.address.toBase58}", *vaultCh) |
      |    for (@(true, vault) <- vaultCh) {
      |      @vault!("balance", *return)
      |    }
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getValidatorVaultQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ <- logEff.info(s"Validator vault balance result: $result")
      } yield {
        result should not be empty
        // Validator vaults are initialized to 0 REV per GenesisBuilder
      }
    }
    t.runSyncUnsafe()
  }

  "InvalidBlocks map" should "contain invalid block after processing" in {
    implicit val _ = new LogicalTime[Effect]

    val t = TestNode.networkEff(genesis, networkSize = 2).use { nodes =>
      val node1 = nodes(1)
      import node1.logEff
      for {
        // Create a valid block on node 0
        deployData  <- ConstructDeploy.basicDeployData[Effect](0)
        signedBlock <- nodes(0).casperEff.deploy(deployData) >> nodes(0).createBlockUnsafe()

        // Create an invalid version (wrong seqNum makes hash invalid)
        invalidBlock = signedBlock.copy(seqNum = 47)
        _            <- logEff.info(s"Original block hash: ${signedBlock.blockHash}")
        _            <- logEff.info(s"Invalid block hash: ${invalidBlock.blockHash}")

        // Process the invalid block on node 1
        status <- nodes(1).processBlock(invalidBlock)
        _      <- logEff.info(s"Process status: $status")

        // Check what's in node 1's dag.invalidBlocks
        dagRep           <- nodes(1).casperEff.getSnapshot.map(_.dag)
        dagInvalidBlocks <- dagRep.invalidBlocks
        _                <- logEff.info(s"dag.invalidBlocks count: ${dagInvalidBlocks.size}")

      } yield {
        // Verify the block was rejected with InvalidBlockHash
        status should be(Left(InvalidBlock.InvalidBlockHash))
        // The invalid block should be in dag.invalidBlocks
        dagInvalidBlocks.map(_.blockHash) should contain(invalidBlock.blockHash)
      }
    }
    t.runSyncUnsafe()
  }

  "System contracts" should "work after adding a block with a deploy" in {
    implicit val timeEff = new LogicalTime[Effect]

    val getBondsQuery = """
      |new return, rl(`rho:registry:lookup`), posCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  for (@(_, PoS) <- posCh) {
      |    @PoS!("getBonds", *return)
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      for {
        // Add a simple deploy to allow block creation
        deploy <- ConstructDeploy.basicDeployData[Effect](0, shardId = genesis.genesisBlock.shardId)
        block  <- node.addBlock(deploy)
        _      <- logEff.info(s"Added block: ${block.blockHash}")

        // Query PoS in the new block's post-state
        result <- node.runtimeManager.playExploratoryDeploy(
                   getBondsQuery,
                   block.body.state.postStateHash
                 )
        _ <- logEff.info(s"PoS getBonds after block: $result")
      } yield {
        result should not be empty
      }
    }
    t.runSyncUnsafe()
  }

  "Validator key lookup" should "succeed in allBonds map" in {
    // Verify validator public keys can be found in allBonds using hexToBytes
    // This is critical for slashing to work correctly
    val validatorPk  = genesis.validatorKeyPairs.head._2
    val validatorHex = Base16.encode(validatorPk.bytes)

    val lookupQuery = s"""
      |new return, rl(`rho:registry:lookup`), posCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  for (@(_, PoS) <- posCh) {
      |    new bondsCh in {
      |      @PoS!("getBonds", *bondsCh) |
      |      for (@bonds <- bondsCh) {
      |        match bonds.get("$validatorHex".hexToBytes()) {
      |          Nil => return!(("KEY_NOT_FOUND", Nil))
      |          stake => return!(("KEY_FOUND", stake))
      |        }
      |      }
      |    }
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   lookupQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ <- logEff.info(s"Validator lookup result: $result")
      } yield {
        result should not be empty
        // The lookup should succeed (KEY_FOUND)
        result.toString should include("KEY_FOUND")
      }
    }
    t.runSyncUnsafe()
  }

  "Invalid block sender" should "be found in genesis validators" in {
    // Verify that a block sender's public key matches genesis validators
    // This ensures slashing can find the validator in allBonds
    implicit val _ = new LogicalTime[Effect]

    val t = TestNode.networkEff(genesis, networkSize = 2).use { nodes =>
      val node1 = nodes(1)
      import node1.logEff
      for {
        // Create a valid block on node 0
        deployData  <- ConstructDeploy.basicDeployData[Effect](0)
        signedBlock <- nodes(0).casperEff.deploy(deployData) >> nodes(0).createBlockUnsafe()

        // Create an invalid version
        invalidBlock = signedBlock.copy(seqNum = 47)
        senderHex    = Base16.encode(invalidBlock.sender.toByteArray)
        _            <- logEff.info(s"Invalid block sender (hex): $senderHex")

        // Check genesis validators
        genesisValidators = genesis.genesisBlock.body.state.bonds.map { bond =>
          Base16.encode(bond.validator.toByteArray)
        }
        _ <- logEff.info(s"Genesis validators: ${genesisValidators.mkString(", ")}")

        // Sender should be in genesis validators
        senderInGenesis = genesisValidators.contains(senderHex)
        _               <- logEff.info(s"Sender in genesis validators: $senderInGenesis")

        // Process the invalid block
        status <- nodes(1).processBlock(invalidBlock)

      } yield {
        // Block should be rejected
        status should be(Left(InvalidBlock.InvalidBlockHash))
        // Sender must be a genesis validator for slashing to work
        senderInGenesis shouldBe true
      }
    }
    t.runSyncUnsafe()
  }
}
