package coop.rchain.casper

import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.helper.TestNode.Effect
import coop.rchain.casper.util.{ConstructDeploy, GenesisBuilder}
import coop.rchain.casper.util.GenesisBuilder.buildGenesis
import coop.rchain.casper.InvalidBlock._
import coop.rchain.blockstorage.syntax._
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import coop.rchain.rholang.interpreter.util.RevAddress
import coop.rchain.shared.Base16
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

/**
  * Diagnostic test to verify system contracts (PoS, RevVault) are properly
  * initialized at genesis and accessible in subsequent blocks.
  *
  * This test helps isolate whether ConsumeFailed errors in system deploys
  * are due to:
  * - Genesis initialization problems (system contracts not deployed)
  * - Block processing problems (state not properly restored)
  */
class SystemContractInitializationSpec extends FlatSpec with Matchers {

  val genesis = buildGenesis()

  "PoS contract" should "be accessible at genesis post-state" in {
    val getBondsQuery = """
      |new return, rl(`rho:registry:lookup`), posCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  for (@(_, PoS) <- posCh) {
      |    @PoS!("getBonds", *return)
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getBondsQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ = println(s"PoS getBonds result: $result")
        _ = result should not be empty
        // Bonds map should have 4 validators (default genesis config)
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "RevVault" should "be accessible at genesis post-state" in {
    val genesisVaultAddr = RevAddress.fromPublicKey(genesis.genesisVaults.toList.head._2).get

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
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getVaultQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ = println(s"RevVault balance result: $result")
        _ = result should not be empty
        // Genesis vault should have 9,000,000 REV
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "Validator vaults" should "have correct balances at genesis" in {
    // GenesisBuilder sets validator vaults to 0 REV (line 94: Vault(_, 0))
    // This test verifies that and checks if it might cause slashing issues
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
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getValidatorVaultQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ = println(s"Validator vault balance result: $result")
        // According to GenesisBuilder, validator vaults have 0 REV
        _ = result should not be empty
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "PoS vault" should "hold the bonded stakes" in {
    // The PoS contract should have a vault holding all bonded stakes
    // When slashing, funds are transferred FROM this vault TO the coop vault
    val getPosVaultBalanceQuery =
      """
      |new return, rl(`rho:registry:lookup`), posCh, vaultCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  for (@(_, PoS) <- posCh) {
      |    // Try to get PoS vault info - this tests if PoS has funds to transfer during slashing
      |    @PoS!("getBonds", *return)
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getPosVaultBalanceQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ = println(s"PoS bonds (stake amounts): $result")
        _ = result should not be empty
        // Bonds should be: validator0 -> 1, validator1 -> 3, validator2 -> 5, validator3 -> 7
        // (from GenesisBuilder.createBonds: 2*i + 1)
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "PoS vault actual balance" should "have REV to cover slashing transfers" in {
    // CRITICAL: Slashing transfers from posVault to coopVault
    // If posVault has 0 REV, the transfer fails and return channel never receives!
    // This query checks if the PoS vault actually has funds

    // First, let's see what getInitialPosVault returns
    val getPosVaultQuery =
      """
      |new return, rl(`rho:registry:lookup`), posCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  for (@(_, PoS) <- posCh) {
      |    @PoS!("getInitialPosVault", *return)
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getPosVaultQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ = println(s"PoS getInitialPosVault result: $result")
        // This will show what the PoS vault object looks like
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "PoS vault balance via RevVault" should "show if vault has funds" in {
    // Get the PoS vault's REV address and check balance via RevVault
    val getPosVaultBalanceQuery =
      """
      |new return, rl(`rho:registry:lookup`), posCh, RevVaultCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  rl!(`rho:rchain:revVault`, *RevVaultCh) |
      |  for (@(_, PoS) <- posCh; @(_, RevVault) <- RevVaultCh) {
      |    new posVaultInfoCh, vaultCh, balanceCh in {
      |      @PoS!("getInitialPosVault", *posVaultInfoCh) |
      |      for (@(posRevAddr, _) <- posVaultInfoCh) {
      |        // Use the REV address to get the vault via RevVault
      |        @RevVault!("findOrCreate", posRevAddr, *vaultCh) |
      |        for (@(true, vault) <- vaultCh) {
      |          @vault!("balance", *balanceCh) |
      |          for (@balance <- balanceCh) {
      |            return!(("posVaultRevAddr", posRevAddr, "balance", balance))
      |          }
      |        }
      |      }
      |    }
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   getPosVaultBalanceQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ = println(s"PoS vault balance: $result")
        // CRITICAL: If balance is 0, slashing will fail because transfer will fail!
        // Expected: Total bonds (1+3+5+7=16) should be in PoS vault
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "InvalidBlocks map" should "contain invalid block after processing" in {
    // This test traces the invalidBlocks map flow to debug slashing issues
    implicit val timeEff = new LogicalTime[Effect]

    val t = TestNode.networkEff(genesis, networkSize = 2).use { nodes =>
      for {
        // Create a valid block on node 0
        deployData  <- ConstructDeploy.basicDeployData[Effect](0)
        signedBlock <- nodes(0).casperEff.deploy(deployData) >> nodes(0).createBlockUnsafe()

        // Create an invalid version (wrong seqNum makes hash invalid)
        invalidBlock = signedBlock.copy(seqNum = 47)
        _            = println(s"Original block hash: ${signedBlock.blockHash}")
        _            = println(s"Invalid block hash: ${invalidBlock.blockHash}")
        _            = println(s"Invalid block sender: ${invalidBlock.sender}")

        // Process the invalid block on node 1
        status <- nodes(1).processBlock(invalidBlock)
        _      = println(s"Process status: $status")

        // Check what's in node 1's dag.invalidBlocks
        dagRep              <- nodes(1).casperEff.getSnapshot.map(_.dag)
        dagInvalidBlocks    <- dagRep.invalidBlocks
        dagInvalidBlocksMap <- dagRep.invalidBlocksMap
        _                   = println(s"dag.invalidBlocks count: ${dagInvalidBlocks.size}")
        _ = dagInvalidBlocks
          .foreach(b => println(s"  Invalid block: ${b.blockHash}, sender: ${b.sender}"))
        _ = println(s"dag.invalidBlocksMap: $dagInvalidBlocksMap")

        // Check invalidLatestMessages
        invalidLM <- dagRep.invalidLatestMessages
        _         = println(s"invalidLatestMessages: $invalidLM")

        // Check what's in CasperSnapshot.invalidBlocks
        cs <- nodes(1).casperEff.getSnapshot
        _  = println(s"CasperSnapshot.invalidBlocks: ${cs.invalidBlocks}")

      } yield {
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
      for {
        // Add a simple deploy to allow block creation
        deploy <- ConstructDeploy.basicDeployData[Effect](0, shardId = genesis.genesisBlock.shardId)
        block  <- node.addBlock(deploy)
        _      = println(s"Added block: ${block.blockHash}")

        // Query PoS in the new block's post-state
        result <- node.runtimeManager.playExploratoryDeploy(
                   getBondsQuery,
                   block.body.state.postStateHash
                 )
        _ = println(s"PoS getBonds after block: $result")
        _ = result should not be empty
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "Validator key format" should "match between allBonds and block.sender" in {
    // DIAGNOSTIC: Compare byte format of validator keys in allBonds vs invalidBlocks
    // allBonds keys are created via: "$pkHex".hexToBytes() in Rholang
    // invalidBlocks values come from: block.sender (raw ByteString)
    //
    // This test verifies if they produce the same byte representation
    implicit val timeEff = new LogicalTime[Effect]

    // Get the first validator's public key bytes
    val validatorPk    = genesis.validatorKeyPairs.head._2
    val validatorBytes = validatorPk.bytes
    val validatorHex   = Base16.encode(validatorBytes)

    println(s"=== VALIDATOR KEY FORMAT DIAGNOSTIC ===")
    println(s"Validator PK bytes length: ${validatorBytes.length}")
    println(s"Validator PK hex: $validatorHex")
    println(
      s"Validator PK bytes (first 10): ${validatorBytes.take(10).map(b => f"$b%02x").mkString(" ")}"
    )

    // Query to get allBonds keys and compare
    val compareBytesQuery = s"""
      |new return, rl(`rho:registry:lookup`), posCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  for (@(_, PoS) <- posCh) {
      |    new bondsCh in {
      |      @PoS!("getBonds", *bondsCh) |
      |      for (@bonds <- bondsCh) {
      |        // Get the keys of the bonds map
      |        new keysCh in {
      |          // Test if our hex-encoded validator exists in bonds
      |          // Format matches ProofOfStake.scala: hexString.hexToBytes()
      |          match bonds.get("$validatorHex".hexToBytes()) {
      |            Nil => return!(("KEY_NOT_FOUND", "hexToBytes lookup failed", bonds.keys()))
      |            stake => return!(("KEY_FOUND", "hexToBytes lookup succeeded", stake, bonds.keys()))
      |          }
      |        }
      |      }
      |    }
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   compareBytesQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ = println(s"Validator lookup result: $result")
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "Slashing with explicit validator lookup" should "show what happens in PoS.slash" in {
    // This test simulates what PoS.slash does:
    // 1. Get invalidBlocks map (blockHash -> validator)
    // 2. Look up validator in allBonds
    // 3. Check if lookup succeeds or fails
    implicit val timeEff = new LogicalTime[Effect]

    val validatorPk  = genesis.validatorKeyPairs.head._2
    val validatorHex = Base16.encode(validatorPk.bytes)
    val blockHashHex = "deadbeef01234567890abcdef" // Dummy hash

    println(s"=== SLASHING LOOKUP DIAGNOSTIC ===")
    println(s"Testing validator hex: $validatorHex")

    // Simulate what happens in PoS.slash:
    // invalidBlocks.getOrElse(blockHash, userPk) returns validator bytes
    // Then state.get("allBonds").get(validator) should return the stake
    val slashSimQuery = s"""
      |new return, rl(`rho:registry:lookup`), posCh in {
      |  rl!(`rho:rchain:pos`, *posCh) |
      |  for (@(_, PoS) <- posCh) {
      |    new bondsCh in {
      |      @PoS!("getBonds", *bondsCh) |
      |      for (@bonds <- bondsCh) {
      |        // Simulate: invalidBlocks returns raw bytes (what RhoType.ByteArray produces)
      |        // In the test, we use hexToBytes which should produce the same result
      |        match bonds.get("$validatorHex".hexToBytes()) {
      |          Nil => {
      |            // This is the failure case that causes ConsumeFailed!
      |            // valBond becomes Nil, transfer fails, return never written
      |            return!(("LOOKUP_FAILED", "bonds.get returned Nil", "keys", bonds.keys()))
      |          }
      |          stake => {
      |            return!(("LOOKUP_SUCCESS", "Found stake", stake))
      |          }
      |        }
      |      }
      |    }
      |  }
      |}
      |""".stripMargin

    val t = TestNode.standaloneEff(genesis).use { node =>
      for {
        result <- node.runtimeManager.playExploratoryDeploy(
                   slashSimQuery,
                   genesis.genesisBlock.body.state.postStateHash
                 )
        _ = println(s"Slashing lookup simulation: $result")
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "Byte format comparison" should "show if RhoType.ByteArray matches allBonds keys" in {
    // Direct comparison of byte formats in the Rholang runtime
    // This test verifies that RhoType.ByteArray produces the same format as hexToBytes

    // Get a genesis validator
    val validatorPk    = genesis.validatorKeyPairs.head._2
    val validatorBytes = validatorPk.bytes
    val validatorHex   = Base16.encode(validatorBytes)

    println(s"=== BYTE FORMAT COMPARISON ===")
    println(s"Validator hex: $validatorHex")
    println(s"Validator bytes length: ${validatorBytes.length}")

    // Create a Par from RhoType.ByteArray and examine its structure
    import coop.rchain.rholang.interpreter.RhoType
    import coop.rchain.models.{Expr, Par}

    val parFromRhoType = RhoType.ByteArray(validatorBytes)
    println(s"RhoType.ByteArray result: $parFromRhoType")

    // Compare with what hexToBytes would produce in Rholang
    val t = TestNode.standaloneEff(genesis).use { node =>
      for {
        // Query allBonds and check if any key matches our validator
        queryResult <- node.runtimeManager.playExploratoryDeploy(
                        s"""
          |new return, rl(`rho:registry:lookup`), posCh in {
          |  rl!(`rho:rchain:pos`, *posCh) |
          |  for (@(_, PoS) <- posCh) {
          |    new bondsCh in {
          |      @PoS!("getBonds", *bondsCh) |
          |      for (@bonds <- bondsCh) {
          |        // Check multiple lookup methods
          |        new hexLookupCh, keysListCh in {
          |          // Method 1: Direct hex lookup (this works in diagnostic tests)
          |          match bonds.get("$validatorHex".hexToBytes()) {
          |            Nil => hexLookupCh!(("HEX_LOOKUP_FAILED", Nil))
          |            stake => hexLookupCh!(("HEX_LOOKUP_SUCCESS", stake))
          |          } |
          |          // Get the keys as a list for inspection
          |          keysListCh!(bonds.keys().toList()) |
          |          for (@hexResult <- hexLookupCh & @keysList <- keysListCh) {
          |            return!((hexResult, "keys", keysList))
          |          }
          |        }
          |      }
          |    }
          |  }
          |}
          |""".stripMargin,
                        genesis.genesisBlock.body.state.postStateHash
                      )
        _ = println(s"Query result: $queryResult")
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "InvalidBlocks validator bytes" should "match allBonds keys when set via Scala" in {
    // This is the REAL test - it uses the actual invalidBlocks system process
    // to verify that validators set via RhoType.ByteArray match allBonds keys
    implicit val timeEff = new LogicalTime[Effect]

    val t = TestNode.networkEff(genesis, networkSize = 2).use { nodes =>
      for {
        // Create a valid block on node 0
        deployData  <- ConstructDeploy.basicDeployData[Effect](0)
        signedBlock <- nodes(0).casperEff.deploy(deployData) >> nodes(0).createBlockUnsafe()

        // Create an invalid version (wrong seqNum makes hash invalid)
        invalidBlock = signedBlock.copy(seqNum = 47)
        _            = println(s"Invalid block hash: ${invalidBlock.blockHash}")
        _            = println(s"Invalid block sender (raw): ${invalidBlock.sender}")
        _ = println(
          s"Invalid block sender (hex): ${Base16.encode(invalidBlock.sender.toByteArray)}"
        )

        // Process the invalid block on node 1
        status <- nodes(1).processBlock(invalidBlock)
        _      = println(s"Process status: $status")

        // Now query using the rho:casper:invalidBlocks system process
        // This will show us the ACTUAL format of validator bytes after setInvalidBlocks
        queryInvalidBlocksAndCompare = s"""
          |new return, rl(`rho:registry:lookup`), posCh,
          |    getInvalidBlocks(`rho:casper:invalidBlocks`)
          |in {
          |  rl!(`rho:rchain:pos`, *posCh) |
          |  getInvalidBlocks!(*return)
          |}
          |""".stripMargin

        cs            <- nodes(1).casperEff.getSnapshot
        postStateHash = cs.parents.head.body.state.postStateHash
        result <- nodes(1).runtimeManager.playExploratoryDeploy(
                   queryInvalidBlocksAndCompare,
                   postStateHash
                 )
        _ = println(s"invalidBlocks from system process: $result")

        // Also query allBonds to compare
        queryBonds = """
          |new return, rl(`rho:registry:lookup`), posCh in {
          |  rl!(`rho:rchain:pos`, *posCh) |
          |  for (@(_, PoS) <- posCh) {
          |    @PoS!("getBonds", *return)
          |  }
          |}
          |""".stripMargin

        bondsResult <- nodes(1).runtimeManager.playExploratoryDeploy(
                        queryBonds,
                        postStateHash
                      )
        _ = println(s"allBonds keys for comparison: $bondsResult")

      } yield {
        status should be(Left(InvalidBlock.InvalidBlockHash))
      }
    }
    t.runSyncUnsafe()
  }

  "PoS.slash lookup simulation" should "trace the exact lookup path" in {
    // This test verifies that the sender's public key can be found in allBonds
    // using the SAME method that would be used during slashing
    implicit val timeEff = new LogicalTime[Effect]

    val t = TestNode.networkEff(genesis, networkSize = 2).use { nodes =>
      for {
        // Create a valid block on node 0
        deployData  <- ConstructDeploy.basicDeployData[Effect](0)
        signedBlock <- nodes(0).casperEff.deploy(deployData) >> nodes(0).createBlockUnsafe()

        // Create an invalid version
        invalidBlock = signedBlock.copy(seqNum = 47)
        blockHashHex = Base16.encode(invalidBlock.blockHash.toByteArray)
        senderHex    = Base16.encode(invalidBlock.sender.toByteArray)
        _            = println(s"=== PoS.slash simulation test ===")
        _            = println(s"Invalid block hash (hex): $blockHashHex")
        _            = println(s"Invalid block sender (hex): $senderHex")
        _            = println(s"Sender bytes length: ${invalidBlock.sender.toByteArray.length}")

        // Check genesis validators to compare
        genesisValidators = genesis.genesisBlock.body.state.bonds.map { bond =>
          Base16.encode(bond.validator.toByteArray)
        }
        _ = println(s"Genesis validators (${genesisValidators.size}):")
        _ = genesisValidators.foreach(v => println(s"  - $v"))

        // Check if sender matches any genesis validator
        senderInGenesis = genesisValidators.contains(senderHex)
        _               = println(s"Sender in genesis validators: $senderInGenesis")

        // Process the invalid block on node 1
        status <- nodes(1).processBlock(invalidBlock)
        _      = println(s"Process status: $status")

        cs            <- nodes(1).casperEff.getSnapshot
        postStateHash = cs.parents.head.body.state.postStateHash

        // Try looking up the sender in allBonds with hexToBytes - this SHOULD work
        lookupQuery  = s"""
          |new return, rl(`rho:registry:lookup`), posCh in {
          |  rl!(`rho:rchain:pos`, *posCh) |
          |  for (@(_, PoS) <- posCh) {
          |    new bondsCh in {
          |      @PoS!("getBonds", *bondsCh) |
          |      for (@bonds <- bondsCh) {
          |        return!(("sender_lookup", bonds.get("$senderHex".hexToBytes()), "bonds_keys", bonds.keys().toList()))
          |      }
          |    }
          |  }
          |}
          |""".stripMargin
        lookupResult <- nodes(1).runtimeManager.playExploratoryDeploy(lookupQuery, postStateHash)
        _            = println(s"Sender lookup in allBonds result: $lookupResult")

      } yield {
        status should be(Left(InvalidBlock.InvalidBlockHash))
      }
    }
    t.runSyncUnsafe()
  }
}
