use crate::args::*;
use crate::f1r3fly_api::F1r3flyApi;
use std::fs;
use std::time::Instant;

pub async fn exploratory_deploy_command(
    args: &ExploratoryDeployArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read the Rholang code from file
    println!("📄 Reading Rholang from: {}", args.file.display());
    let rholang_code =
        fs::read_to_string(&args.file).map_err(|e| format!("Failed to read file: {}", e))?;
    println!("📊 Code size: {} bytes", rholang_code.len());

    // Initialize the F1r3fly API client
    println!(
        "🔌 Connecting to F1r3fly node at {}:{}",
        args.host, args.port
    );
    let f1r3fly_api = F1r3flyApi::new(&args.private_key, &args.host, args.port);

    // Execute the exploratory deployment
    println!("🚀 Executing Rholang code (exploratory deploy)...");

    // Display block hash if provided
    if let Some(block_hash) = &args.block_hash {
        println!("🧱 Using block hash: {}", block_hash);
    }

    // Display state hash preference
    if args.use_pre_state {
        println!("🔍 Using pre-state hash");
    } else {
        println!("🔍 Using post-state hash");
    }

    let start_time = Instant::now();

    match f1r3fly_api
        .exploratory_deploy(
            &rholang_code,
            args.block_hash.as_deref(),
            args.use_pre_state,
        )
        .await
    {
        Ok((result, block_info)) => {
            let duration = start_time.elapsed();
            println!("✅ Execution successful!");
            println!("⏱️  Time taken: {:.2?}", duration);
            println!("🧱 {}", block_info);
            println!("📊 Result:");
            println!("{}", result);
        }
        Err(e) => {
            println!("❌ Execution failed!");
            println!("Error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

pub async fn deploy_command(args: &DeployArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Read the Rholang code from file
    println!("📄 Reading Rholang from: {}", args.file.display());
    let rholang_code =
        fs::read_to_string(&args.file).map_err(|e| format!("Failed to read file: {}", e))?;
    println!("📊 Code size: {} bytes", rholang_code.len());

    // Initialize the F1r3fly API client
    println!(
        "🔌 Connecting to F1r3fly node at {}:{}",
        args.host, args.port
    );
    let f1r3fly_api = F1r3flyApi::new(&args.private_key, &args.host, args.port);

    let phlo_limit = if args.bigger_phlo {
        "5,000,000,000"
    } else {
        "50,000"
    };
    println!("💰 Using phlo limit: {}", phlo_limit);

    // Deploy the Rholang code
    println!("🚀 Deploying Rholang code...");
    let start_time = Instant::now();

    match f1r3fly_api
        .deploy(&rholang_code, args.bigger_phlo, "rholang")
        .await
    {
        Ok(deploy_id) => {
            let duration = start_time.elapsed();
            println!("✅ Deployment successful!");
            println!("⏱️  Time taken: {:.2?}", duration);
            println!("🆔 Deploy ID: {}", deploy_id);
        }
        Err(e) => {
            println!("❌ Deployment failed!");
            println!("Error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

pub async fn propose_command(args: &ProposeArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the F1r3fly API client
    println!(
        "🔌 Connecting to F1r3fly node at {}:{}",
        args.host, args.port
    );
    let f1r3fly_api = F1r3flyApi::new(&args.private_key, &args.host, args.port);

    // Propose a block
    println!("📦 Proposing a new block...");
    let start_time = Instant::now();

    match f1r3fly_api.propose().await {
        Ok(block_hash) => {
            let duration = start_time.elapsed();
            println!("✅ Block proposed successfully!");
            println!("⏱️  Time taken: {:.2?}", duration);
            println!("🧱 Block hash: {}", block_hash);
        }
        Err(e) => {
            println!("❌ Block proposal failed!");
            println!("Error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

pub async fn full_deploy_command(args: &DeployArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Read the Rholang code from file
    println!("📄 Reading Rholang from: {}", args.file.display());
    let rholang_code =
        fs::read_to_string(&args.file).map_err(|e| format!("Failed to read file: {}", e))?;
    println!("📊 Code size: {} bytes", rholang_code.len());

    // Initialize the F1r3fly API client
    println!(
        "🔌 Connecting to F1r3fly node at {}:{}",
        args.host, args.port
    );
    let f1r3fly_api = F1r3flyApi::new(&args.private_key, &args.host, args.port);

    let phlo_limit = if args.bigger_phlo {
        "5,000,000,000"
    } else {
        "50,000"
    };
    println!("💰 Using phlo limit: {}", phlo_limit);

    // Deploy and propose
    println!("🚀 Deploying Rholang code and proposing a block...");
    let start_time = Instant::now();

    match f1r3fly_api
        .full_deploy(&rholang_code, args.bigger_phlo, "rholang")
        .await
    {
        Ok(block_hash) => {
            let duration = start_time.elapsed();
            println!("✅ Deployment and block proposal successful!");
            println!("⏱️  Time taken: {:.2?}", duration);
            println!("🧱 Block hash: {}", block_hash);
        }
        Err(e) => {
            println!("❌ Operation failed!");
            println!("Error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

pub async fn is_finalized_command(
    args: &IsFinalizedArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the F1r3fly API client
    println!(
        "🔌 Connecting to F1r3fly node at {}:{}",
        args.host, args.port
    );
    let f1r3fly_api = F1r3flyApi::new(&args.private_key, &args.host, args.port);

    // Check if the block is finalized
    println!("🔍 Checking if block is finalized: {}", args.block_hash);
    println!(
        "⏱️  Will retry every {} seconds, up to {} times",
        args.retry_delay, args.max_attempts
    );
    let start_time = Instant::now();

    match f1r3fly_api
        .is_finalized(&args.block_hash, args.max_attempts, args.retry_delay)
        .await
    {
        Ok(is_finalized) => {
            let duration = start_time.elapsed();
            if is_finalized {
                println!("✅ Block is finalized!");
            } else {
                println!(
                    "❌ Block is not finalized after {} attempts",
                    args.max_attempts
                );
            }
            println!("⏱️  Time taken: {:.2?}", duration);
        }
        Err(e) => {
            println!("❌ Error checking block finalization!");
            println!("Error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

pub async fn bond_validator_command(
    args: &BondValidatorArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Bonding new validator to the network");
    println!("💰 Stake amount: {} REV", args.stake);

    // Initialize the F1r3fly API client for deploying
    let f1r3fly_api = F1r3flyApi::new(&args.private_key, &args.host, args.port);

    // Create the bonding Rholang code
    let bonding_code = format!(
        r#"new rl(`rho:registry:lookup`), poSCh, retCh, stdout(`rho:io:stdout`) in {{
  stdout!("About to lookup PoS contract...") |
  rl!(`rho:rchain:pos`, *poSCh) |
  for(@(_, PoS) <- poSCh) {{
    stdout!("About to bond...") |
    new deployerId(`rho:rchain:deployerId`) in {{
      @PoS!("bond", *deployerId, {}, *retCh) |
      for (@(result, message) <- retCh) {{
        stdout!(("Bond result:", result, "Message:", message))
      }}
    }}
  }}
}}"#,
        args.stake
    );

    println!("🚀 Deploying bonding transaction...");
    let deploy_start_time = Instant::now();

    // Step 1: Deploy the bonding code
    let deploy_id = match f1r3fly_api.deploy(&bonding_code, true, "rholang").await {
        Ok(deploy_id) => {
            let deploy_duration = deploy_start_time.elapsed();
            println!("✅ Bonding deploy successful! Deploy ID: {}", deploy_id);
            println!("⏱️  Deploy time: {:.2?}", deploy_duration);
            deploy_id
        }
        Err(e) => {
            println!("❌ Bonding deploy failed!");
            println!("Error: {}", e);
            return Err(e);
        }
    };

    // Step 2: Wait for deploy to be included in a block
    println!("⏳ Waiting for bonding deploy to be included in a block...");
    let block_wait_start = Instant::now();
    let max_block_wait_attempts = args.max_wait / args.check_interval;
    let mut block_wait_attempts = 0;

    let block_hash = loop {
        block_wait_attempts += 1;

        // Show progress every 10 attempts or if we're at the end
        if block_wait_attempts % 10 == 0 || block_wait_attempts >= max_block_wait_attempts {
            println!(
                "   ⏱️  Checking... ({}/{} attempts)",
                block_wait_attempts, max_block_wait_attempts
            );
        }

        match f1r3fly_api
            .get_deploy_block_hash(&deploy_id, args.http_port)
            .await
        {
            Ok(Some(hash)) => {
                println!("✅ Bonding deploy found in block: {}", hash);
                break hash;
            }
            Ok(None) => {
                // Deploy not in block yet, continue waiting
            }
            Err(e) => {
                println!("❌ Error checking bonding deploy status: {}", e);
                return Err(e);
            }
        }

        if block_wait_attempts >= max_block_wait_attempts {
            println!(
                "❌ Timeout waiting for bonding deploy to be included in block after {} seconds",
                args.max_wait
            );
            return Err("Bonding deploy inclusion timeout".into());
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(args.check_interval)).await;
    };

    let block_wait_duration = block_wait_start.elapsed();
    println!("⏱️  Block inclusion time: {:.2?}", block_wait_duration);

    // Step 3: Wait for block finalization using observer node
    println!("🔍 Waiting for block finalization...");

    // Determine observer node settings (fallback to default observer or main node)
    let observer_host = args.observer_host.as_deref().unwrap_or("localhost");
    let observer_port = args.observer_port.unwrap_or(40452); // Default to port 40452 (common observer port)

    let finalization_start = Instant::now();

    // Create observer node API client for finalization checks
    let observer_api = F1r3flyApi::new(&args.private_key, observer_host, observer_port);

    // Use the same finalization logic as deploy_and_wait_command
    let finalization_max_attempts: u32 = 120; // 10 minutes (120 * 5 seconds)
    let finalization_retry_delay: u64 = 5;

    match observer_api
        .is_finalized(
            &block_hash,
            finalization_max_attempts,
            finalization_retry_delay,
        )
        .await
    {
        Ok(true) => {
            let finalization_duration = finalization_start.elapsed();
            let total_duration = deploy_start_time.elapsed();
            println!("✅ Block finalized! Bonding transaction is complete.");
            println!("⏱️  Finalization time: {:.2?}", finalization_duration);
            println!("🎉 Total bonding process time: {:.2?}", total_duration);
        }
        Ok(false) => {
            println!("⚠️  Block not yet finalized after {} attempts, but bonding deploy is in the blockchain.", finalization_max_attempts);
            println!("💡 The validator bonding is likely successful and will be finalized soon.");
        }
        Err(e) => {
            println!("❌ Error checking finalization status: {}", e);
            println!("⚠️  Could not verify finalization, but bonding deploy is in the blockchain.");
        }
    }

    // Handle propose logic if enabled
    if args.propose {
        println!("📦 Proposing block to help finalize the bonding transaction...");
        let propose_start = Instant::now();

        match f1r3fly_api.propose().await {
            Ok(block_hash) => {
                let propose_duration = propose_start.elapsed();
                println!("✅ Block proposed successfully!");
                println!("⏱️  Propose time: {:.2?}", propose_duration);
                println!("🧱 Block hash: {}", block_hash);
            }
            Err(e) => {
                println!("❌ Block proposal failed!");
                println!("Error: {}", e);
                return Err(e);
            }
        }
    }

    println!("🎯 Validator bonding process completed!");
    println!("📋 Next steps:");
    println!("   1. Verify the validator appears in the bonds list");
    println!("   2. Check that the validator is participating in consensus");
    println!("   3. Monitor for block proposals from the new validator");

    Ok(())
}

pub async fn transfer_command(args: &TransferArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("💸 Initiating REV transfer");

    // Initialize the F1r3fly API client
    println!(
        "🔌 Connecting to F1r3fly node at {}:{}",
        args.host, args.port
    );
    let f1r3fly_api = F1r3flyApi::new(&args.private_key, &args.host, args.port);

    // Generate from_address from private key using proper crypto utils
    println!("🔍 Deriving sender address from private key...");
    let from_address = {
        use crate::utils::CryptoUtils;
        let secret_key = CryptoUtils::decode_private_key(&args.private_key)?;
        let public_key = CryptoUtils::derive_public_key(&secret_key);
        let public_key_hex = CryptoUtils::serialize_public_key(&public_key, false);
        CryptoUtils::generate_rev_address(&public_key_hex)?
    };

    // Validate addresses format
    validate_rev_address(&from_address)?;
    validate_rev_address(&args.to_address)?;

    // Convert REV to dust (1 REV = 100,000,000 dust)
    let amount_dust = args.amount * 100_000_000;

    println!("📋 Transfer Details:");
    println!("   From: {}", from_address);
    println!("   To: {}", args.to_address);
    println!("   Amount: {} REV ({} dust)", args.amount, amount_dust);
    println!(
        "   Phlo limit: {}",
        if args.bigger_phlo {
            "High (recommended for transfers)"
        } else {
            "Standard"
        }
    );

    // Generate Rholang transfer contract
    let rholang_code = generate_transfer_contract(&from_address, &args.to_address, amount_dust);

    // Step 1: Deploy the transfer contract
    println!("🚀 Deploying transfer contract...");
    let deploy_start_time = Instant::now();

    let deploy_id = match f1r3fly_api
        .deploy(&rholang_code, args.bigger_phlo, "rholang")
        .await
    {
        Ok(deploy_id) => {
            let deploy_duration = deploy_start_time.elapsed();
            println!("✅ Transfer contract deployed successfully!");
            println!("⏱️  Deploy time: {:.2?}", deploy_duration);
            println!("🆔 Deploy ID: {}", deploy_id);
            deploy_id
        }
        Err(e) => {
            println!("❌ Transfer deployment failed!");
            println!("Error: {}", e);
            return Err(e);
        }
    };

    // Step 2: Wait for deploy to be included in a block
    println!("⏳ Waiting for transfer deploy to be included in a block...");
    let block_wait_start = Instant::now();
    let max_block_wait_attempts = args.max_wait / args.check_interval;
    let mut block_wait_attempts = 0;

    let block_hash = loop {
        block_wait_attempts += 1;

        // Show progress every 10 attempts or if we're at the end
        if block_wait_attempts % 10 == 0 || block_wait_attempts >= max_block_wait_attempts {
            println!(
                "   ⏱️  Checking... ({}/{} attempts)",
                block_wait_attempts, max_block_wait_attempts
            );
        }

        match f1r3fly_api
            .get_deploy_block_hash(&deploy_id, args.http_port)
            .await
        {
            Ok(Some(hash)) => {
                println!("✅ Transfer deploy found in block: {}", hash);
                break hash;
            }
            Ok(None) => {
                // Deploy not in block yet, continue waiting
            }
            Err(e) => {
                println!("❌ Error checking transfer deploy status: {}", e);
                return Err(e);
            }
        }

        if block_wait_attempts >= max_block_wait_attempts {
            println!(
                "❌ Timeout waiting for transfer deploy to be included in block after {} seconds",
                args.max_wait
            );
            return Err("Transfer deploy inclusion timeout".into());
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(args.check_interval)).await;
    };

    let block_wait_duration = block_wait_start.elapsed();
    println!("⏱️  Block inclusion time: {:.2?}", block_wait_duration);

    // Step 3: Wait for block finalization using observer node
    println!("🔍 Waiting for block finalization...");

    // Determine observer node settings (fallback to default observer or main node)
    let observer_host = args.observer_host.as_deref().unwrap_or("localhost");
    let observer_port = args.observer_port.unwrap_or(40452); // Default to port 40452 (common observer port)

    let finalization_start = Instant::now();

    // Create observer node API client for finalization checks
    let observer_api = F1r3flyApi::new(&args.private_key, observer_host, observer_port);

    // Use the same finalization logic as deploy_and_wait_command
    let finalization_max_attempts: u32 = 120; // 10 minutes (120 * 5 seconds)
    let finalization_retry_delay: u64 = 5;

    match observer_api
        .is_finalized(
            &block_hash,
            finalization_max_attempts,
            finalization_retry_delay,
        )
        .await
    {
        Ok(true) => {
            let finalization_duration = finalization_start.elapsed();
            let total_duration = deploy_start_time.elapsed();
            println!("✅ Block finalized! Transfer completed successfully.");
            println!("⏱️  Finalization time: {:.2?}", finalization_duration);
            println!("🎉 Total transfer time: {:.2?}", total_duration);
        }
        Ok(false) => {
            println!("⚠️  Block not yet finalized after {} attempts, but transfer deploy is in the blockchain.", finalization_max_attempts);
            println!("💡 The transfer is likely successful and will be finalized soon.");
        }
        Err(e) => {
            println!("❌ Error checking finalization status: {}", e);
            println!(
                "⚠️  Could not verify finalization, but transfer deploy is in the blockchain."
            );
        }
    }

    // Handle propose logic if enabled
    if args.propose {
        println!("📦 Proposing block to help finalize the transfer...");
        let propose_start = Instant::now();

        match f1r3fly_api.propose().await {
            Ok(block_hash) => {
                let propose_duration = propose_start.elapsed();
                println!("✅ Block proposed successfully!");
                println!("⏱️  Propose time: {:.2?}", propose_duration);
                println!("🧱 Block hash: {}", block_hash);
            }
            Err(e) => {
                println!("❌ Block proposal failed!");
                println!("Error: {}", e);
                return Err(e);
            }
        }
    }

    println!("🎯 Transfer process completed!");

    Ok(())
}

pub async fn deploy_and_wait_command(
    args: &DeployAndWaitArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read the Rholang code from file
    println!("📄 Reading Rholang from: {}", args.file);
    let rholang_code =
        fs::read_to_string(&args.file).map_err(|e| format!("Failed to read file: {}", e))?;
    println!("📊 Code size: {} bytes", rholang_code.len());

    // Initialize the F1r3fly API client
    println!(
        "🔌 Connecting to F1r3fly node at {}:{}",
        args.host, args.port
    );
    let private_key = args
        .private_key
        .as_deref()
        .unwrap_or("5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657");
    let f1r3fly_api = F1r3flyApi::new(private_key, &args.host, args.port);

    let phlo_limit = if args.bigger_phlo {
        "5,000,000,000"
    } else {
        "50,000"
    };
    println!("💰 Using phlo limit: {}", phlo_limit);

    // Step 1: Deploy the Rholang code
    println!("🚀 Deploying Rholang code...");
    let deploy_start_time = Instant::now();

    let deploy_id = match f1r3fly_api
        .deploy(&rholang_code, args.bigger_phlo, "rholang")
        .await
    {
        Ok(deploy_id) => {
            let deploy_duration = deploy_start_time.elapsed();
            println!("✅ Deploy successful! Deploy ID: {}", deploy_id);
            println!("⏱️  Deploy time: {:.2?}", deploy_duration);
            deploy_id
        }
        Err(e) => {
            println!("❌ Deployment failed!");
            println!("Error: {}", e);
            return Err(e);
        }
    };

    // Step 2: Wait for deploy to be included in a block
    println!("⏳ Waiting for deploy to be included in a block...");
    let block_wait_start = Instant::now();
    let max_block_wait_attempts = args.max_wait / args.check_interval;
    let mut block_wait_attempts = 0;

    let block_hash = loop {
        block_wait_attempts += 1;

        // Show progress every 10 attempts or if we're at the end
        if block_wait_attempts % 10 == 0 || block_wait_attempts >= max_block_wait_attempts {
            println!(
                "   ⏱️  Checking... ({}/{} attempts)",
                block_wait_attempts, max_block_wait_attempts
            );
        }

        match f1r3fly_api
            .get_deploy_block_hash(&deploy_id, args.http_port)
            .await
        {
            Ok(Some(hash)) => {
                println!("✅ Deploy found in block: {}", hash);
                break hash;
            }
            Ok(None) => {
                // Deploy not in block yet, continue waiting
            }
            Err(e) => {
                println!("❌ Error checking deploy status: {}", e);
                return Err(e);
            }
        }

        if block_wait_attempts >= max_block_wait_attempts {
            println!(
                "❌ Timeout waiting for deploy to be included in block after {} seconds",
                args.max_wait
            );
            return Err("Deploy inclusion timeout".into());
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(args.check_interval)).await;
    };

    let block_wait_duration = block_wait_start.elapsed();
    println!("⏱️  Block inclusion time: {:.2?}", block_wait_duration);

    // Step 3: Wait for block finalization using observer node
    println!("🔍 Waiting for block finalization...");

    // Determine observer node settings (fallback to default observer or main node)
    let observer_host = args.observer_host.as_deref().unwrap_or("localhost");
    let observer_port = args.observer_port.unwrap_or(40452); // Default to port 40452 (common observer port)

    let finalization_start = Instant::now();

    // Create observer node API client for finalization checks
    let private_key = args
        .private_key
        .as_deref()
        .unwrap_or("5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657");
    let observer_api = F1r3flyApi::new(private_key, observer_host, observer_port);

    // Calculate finalization attempts (default: 120 attempts, 5 second intervals = 10 minutes)
    let finalization_max_attempts: u32 = 120; // 10 minutes (120 * 5 seconds)
    let finalization_retry_delay: u64 = 5;

    match observer_api
        .is_finalized(
            &block_hash,
            finalization_max_attempts,
            finalization_retry_delay,
        )
        .await
    {
        Ok(true) => {
            let finalization_duration = finalization_start.elapsed();
            let total_duration = deploy_start_time.elapsed();

            println!("✅ Block finalized! Deploy completed successfully.");
            println!("⏱️  Finalization time: {:.2?}", finalization_duration);
            println!("📊 Total time: {:.2?}", total_duration);
        }
        Ok(false) => {
            println!(
                "⚠️  Block not yet finalized after {} attempts, but deploy is in the blockchain.",
                finalization_max_attempts
            );
            println!("💡 The deployment is likely successful and will be finalized soon.");
        }
        Err(e) => {
            println!("❌ Error checking finalization status: {}", e);
            println!("⚠️  Could not verify finalization, but deploy is in the blockchain.");
        }
    }

    Ok(())
}

pub async fn get_deploy_command(args: &GetDeployArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::f1r3fly_api::DeployStatus;

    println!("🔍 Looking up deploy: {}", args.deploy_id);
    println!(
        "🔌 Connecting to F1r3fly node at {}:{}",
        args.host, args.http_port
    );

    // Initialize the F1r3fly API client (private key not needed for read operations)
    let dummy_private_key = "5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657";
    let f1r3fly_api = F1r3flyApi::new(dummy_private_key, &args.host, 40412); // Port doesn't matter for HTTP queries

    let start_time = Instant::now();

    match f1r3fly_api
        .get_deploy_info(&args.deploy_id, args.http_port)
        .await
    {
        Ok(deploy_info) => {
            let duration = start_time.elapsed();

            match args.format.as_str() {
                "json" => {
                    // Output raw JSON
                    let json_output = serde_json::to_string_pretty(&deploy_info)?;
                    println!("{}", json_output);
                }
                "summary" => {
                    // One-line summary
                    match deploy_info.status {
                        DeployStatus::Included => {
                            if let Some(block_hash) = &deploy_info.block_hash {
                                println!(
                                    "✅ Deploy {} included in block {}",
                                    deploy_info.deploy_id, block_hash
                                );
                            } else {
                                println!("✅ Deploy {} included in block", deploy_info.deploy_id);
                            }
                        }
                        DeployStatus::Pending => {
                            println!(
                                "⏳ Deploy {} pending (not yet in block)",
                                deploy_info.deploy_id
                            );
                        }
                        DeployStatus::NotFound => {
                            println!("❌ Deploy {} not found", deploy_info.deploy_id);
                        }
                        DeployStatus::Error(ref err) => {
                            println!("❌ Deploy {} error: {}", deploy_info.deploy_id, err);
                        }
                    }
                }
                "pretty" | _ => {
                    // Pretty formatted output (default)
                    println!("📋 Deploy Information");
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    println!("🆔 Deploy ID: {}", deploy_info.deploy_id);

                    match deploy_info.status {
                        DeployStatus::Included => {
                            println!("✅ Status: Included in block");
                            if let Some(block_hash) = &deploy_info.block_hash {
                                println!("🧱 Block Hash: {}", block_hash);
                            }
                        }
                        DeployStatus::Pending => {
                            println!("⏳ Status: Pending (not yet in block)");
                        }
                        DeployStatus::NotFound => {
                            println!("❌ Status: Not found");
                            println!("⏱️  Query time: {:.2?}", duration);
                            return Ok(());
                        }
                        DeployStatus::Error(ref err) => {
                            println!("❌ Status: Error - {}", err);
                            println!("⏱️  Query time: {:.2?}", duration);
                            return Ok(());
                        }
                    }

                    if args.verbose || deploy_info.status == DeployStatus::Included {
                        if let Some(sender) = &deploy_info.sender {
                            println!("👤 Sender: {}", sender);
                        }
                        if let Some(seq_num) = deploy_info.seq_num {
                            println!("🔢 Sequence Number: {}", seq_num);
                        }
                        if let Some(timestamp) = deploy_info.timestamp {
                            println!("🕐 Timestamp: {}", timestamp);
                        }
                        if let Some(shard_id) = &deploy_info.shard_id {
                            println!("🌐 Shard ID: {}", shard_id);
                        }
                        if let Some(sig_algorithm) = &deploy_info.sig_algorithm {
                            println!("🔐 Signature Algorithm: {}", sig_algorithm);
                        }
                        if args.verbose {
                            if let Some(sig) = &deploy_info.sig {
                                println!("✍️  Signature: {}", sig);
                            }
                        }
                    }

                    println!("⏱️  Query time: {:.2?}", duration);
                }
            }
        }
        Err(e) => {
            println!("❌ Error retrieving deploy information: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

fn validate_rev_address(address: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !address.starts_with("1111") {
        return Err("Invalid REV address format: must start with '1111'".into());
    }

    if address.len() < 40 {
        return Err("Invalid REV address format: too short".into());
    }

    Ok(())
}

fn generate_transfer_contract(from_address: &str, to_address: &str, amount_dust: u64) -> String {
    format!(
        r#"new 
    deployerId(`rho:rchain:deployerId`),
    stdout(`rho:io:stdout`),
    rl(`rho:registry:lookup`),
    revVaultCh,
    vaultCh,
    toVaultCh,
    revVaultKeyCh,
    resultCh
in {{
  rl!(`rho:rchain:revVault`, *revVaultCh) |
  for (@(_, RevVault) <- revVaultCh) {{
    @RevVault!("findOrCreate", "{}", *vaultCh) |
    @RevVault!("findOrCreate", "{}", *toVaultCh) |
    @RevVault!("deployerAuthKey", *deployerId, *revVaultKeyCh) |
    for (@(true, vault) <- vaultCh; key <- revVaultKeyCh; @(true, toVault) <- toVaultCh) {{
      @vault!("transfer", "{}", {}, *key, *resultCh) |
      for (@result <- resultCh) {{
        match result {{
          (true, Nil) => {{
            stdout!(("Transfer successful:", {}, "REV"))
          }}
          (false, reason) => {{
            stdout!(("Transfer failed:", reason))
          }}
        }}
      }}
    }} |
    for (@(false, errorMsg) <- vaultCh) {{
      stdout!(("Sender vault error:", errorMsg))
    }} |
    for (@(false, errorMsg) <- toVaultCh) {{
      stdout!(("Destination vault error:", errorMsg))
    }}
  }}
}}"#,
        from_address, // findOrCreate sender
        to_address,   // findOrCreate recipient
        to_address,   // transfer target
        amount_dust,  // transfer amount
        amount_dust   // success message amount
    )
}
