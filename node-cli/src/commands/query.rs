use crate::args::*;
use crate::f1r3fly_api::F1r3flyApi;
use reqwest;
use serde_json;
use std::time::Instant;

pub async fn status_command(args: &HttpArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Getting node status from {}:{}", args.host, args.port);

    let url = format!("http://{}:{}/status", args.host, args.port);
    let client = reqwest::Client::new();

    let start_time = Instant::now();

    match client.get(&url).send().await {
        Ok(response) => {
            let duration = start_time.elapsed();
            if response.status().is_success() {
                let status_text = response.text().await?;
                let status_json: serde_json::Value = serde_json::from_str(&status_text)?;

                println!("✅ Node status retrieved successfully!");
                println!("⏱️  Time taken: {:.2?}", duration);
                println!("📊 Node Status:");
                println!("{}", serde_json::to_string_pretty(&status_json)?);
            } else {
                println!("❌ Failed to get node status: HTTP {}", response.status());
                println!("Error: {}", response.text().await?);
            }
        }
        Err(e) => {
            println!("❌ Connection failed!");
            println!("Error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

pub async fn blocks_command(args: &BlocksArgs) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let client = reqwest::Client::new();

    if let Some(block_hash) = &args.block_hash {
        println!("🔍 Getting specific block: {}", block_hash);
        let url = format!(
            "http://{}:{}/api/block/{}",
            args.host, args.port, block_hash
        );

        match client.get(&url).send().await {
            Ok(response) => {
                let duration = start_time.elapsed();
                if response.status().is_success() {
                    let block_text = response.text().await?;
                    let block_json: serde_json::Value = serde_json::from_str(&block_text)?;

                    println!("✅ Block retrieved successfully!");
                    println!("⏱️  Time taken: {:.2?}", duration);
                    println!("🧱 Block Details:");
                    println!("{}", serde_json::to_string_pretty(&block_json)?);
                } else {
                    println!("❌ Failed to get block: HTTP {}", response.status());
                    println!("Error: {}", response.text().await?);
                }
            }
            Err(e) => {
                println!("❌ Connection failed!");
                println!("Error: {}", e);
                return Err(e.into());
            }
        }
    } else {
        println!(
            "🔍 Getting {} recent blocks from {}:{}",
            args.number, args.host, args.port
        );
        let url = format!(
            "http://{}:{}/api/blocks/{}",
            args.host, args.port, args.number
        );

        match client.get(&url).send().await {
            Ok(response) => {
                let duration = start_time.elapsed();
                if response.status().is_success() {
                    let blocks_text = response.text().await?;
                    let blocks_json: serde_json::Value = serde_json::from_str(&blocks_text)?;

                    println!("✅ Blocks retrieved successfully!");
                    println!("⏱️  Time taken: {:.2?}", duration);
                    println!("🧱 Recent Blocks:");
                    println!("{}", serde_json::to_string_pretty(&blocks_json)?);
                } else {
                    println!("❌ Failed to get blocks: HTTP {}", response.status());
                    println!("Error: {}", response.text().await?);
                }
            }
            Err(e) => {
                println!("❌ Connection failed!");
                println!("Error: {}", e);
                return Err(e.into());
            }
        }
    }

    Ok(())
}

pub async fn bonds_command(args: &HttpArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🔍 Getting validator bonds from {}:{}",
        args.host, args.port
    );

    let url = format!("http://{}:{}/api/explore-deploy", args.host, args.port);
    let client = reqwest::Client::new();

    let rholang_query = r#"new return, rl(`rho:registry:lookup`), poSCh in { rl!(`rho:rchain:pos`, *poSCh) | for(@(_, PoS) <- poSCh) { @PoS!("getBonds", *return) } }"#;

    let body = serde_json::json!({
        "term": rholang_query
    });

    let start_time = Instant::now();

    match client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(response) => {
            let duration = start_time.elapsed();
            if response.status().is_success() {
                let bonds_text = response.text().await?;
                let bonds_json: serde_json::Value = serde_json::from_str(&bonds_text)?;

                println!("✅ Validator bonds retrieved successfully!");
                println!("⏱️  Time taken: {:.2?}", duration);
                println!();

                // Parse and display bonds data in a clean format
                if let Some(block) = bonds_json.get("block") {
                    if let Some(bonds) = block.get("bonds") {
                        if let Some(bonds_array) = bonds.as_array() {
                            let validator_count = bonds_array.len();
                            let total_stake: i64 = bonds_array
                                .iter()
                                .filter_map(|bond| bond.get("stake")?.as_i64())
                                .sum();

                            println!(
                                "🔗 Bonded Validators ({} total, {} total stake):",
                                validator_count, total_stake
                            );
                            println!();

                            for (i, bond) in bonds_array.iter().enumerate() {
                                if let (Some(validator), Some(stake)) = (
                                    bond.get("validator").and_then(|v| v.as_str()),
                                    bond.get("stake").and_then(|s| s.as_i64()),
                                ) {
                                    // Truncate long validator keys for readability
                                    let truncated_key = if validator.len() > 16 {
                                        format!(
                                            "{}...{}",
                                            &validator[..8],
                                            &validator[validator.len() - 8..]
                                        )
                                    } else {
                                        validator.to_string()
                                    };

                                    println!("  {}. {} (stake: {})", i + 1, truncated_key, stake);
                                }
                            }
                        } else {
                            println!("❌ Invalid bonds format in response");
                        }
                    } else {
                        println!("❌ No bonds data found in response");
                    }
                } else {
                    println!("❌ No block data found in response");
                }
            } else {
                println!("❌ Failed to get bonds: HTTP {}", response.status());
                println!("Error: {}", response.text().await?);
            }
        }
        Err(e) => {
            println!("❌ Connection failed!");
            println!("Error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

pub async fn active_validators_command(args: &HttpArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🔍 Getting active validators from {}:{}",
        args.host, args.port
    );

    let url = format!("http://{}:{}/api/explore-deploy", args.host, args.port);
    let client = reqwest::Client::new();

    let rholang_query = r#"new return, rl(`rho:registry:lookup`), poSCh in { rl!(`rho:rchain:pos`, *poSCh) | for(@(_, PoS) <- poSCh) { @PoS!("getActiveValidators", *return) } }"#;

    let body = serde_json::json!({
        "term": rholang_query
    });

    let start_time = Instant::now();

    match client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(response) => {
            let duration = start_time.elapsed();
            if response.status().is_success() {
                let validators_text = response.text().await?;
                let validators_json: serde_json::Value = serde_json::from_str(&validators_text)?;

                println!("✅ Active validators retrieved successfully!");
                println!("⏱️  Time taken: {:.2?}", duration);
                println!();

                // Parse and display validator data in a clean format
                if let Some(block) = validators_json.get("block") {
                    if let Some(bonds) = block.get("bonds") {
                        if let Some(bonds_array) = bonds.as_array() {
                            let validator_count = bonds_array.len();
                            let total_stake: i64 = bonds_array
                                .iter()
                                .filter_map(|bond| bond.get("stake")?.as_i64())
                                .sum();

                            println!(
                                "👥 Active Validators ({} total, {} total stake):",
                                validator_count, total_stake
                            );
                            println!();

                            for (i, bond) in bonds_array.iter().enumerate() {
                                if let (Some(validator), Some(stake)) = (
                                    bond.get("validator").and_then(|v| v.as_str()),
                                    bond.get("stake").and_then(|s| s.as_i64()),
                                ) {
                                    // Truncate long validator keys for readability
                                    let truncated_key = if validator.len() > 16 {
                                        format!(
                                            "{}...{}",
                                            &validator[..8],
                                            &validator[validator.len() - 8..]
                                        )
                                    } else {
                                        validator.to_string()
                                    };

                                    println!("  {}. {} (stake: {})", i + 1, truncated_key, stake);
                                }
                            }
                        } else {
                            println!("❌ Invalid bonds format in response");
                        }
                    } else {
                        println!("❌ No bonds data found in response");
                    }
                } else {
                    println!("❌ No block data found in response");
                }
            } else {
                println!(
                    "❌ Failed to get active validators: HTTP {}",
                    response.status()
                );
                println!("Error: {}", response.text().await?);
            }
        }
        Err(e) => {
            println!("❌ Connection failed!");
            println!("Error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

pub async fn wallet_balance_command(
    args: &WalletBalanceArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Checking wallet balance for address: {}", args.address);

    // Use F1r3fly API with gRPC (like exploratory-deploy)
    let f1r3fly_api = F1r3flyApi::new(
        "5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657", // Bootstrap private key
        &args.host,
        args.port,
    );

    let rholang_query = format!(
        r#"new return, rl(`rho:registry:lookup`), revVaultCh, vaultCh, balanceCh in {{
            rl!(`rho:rchain:revVault`, *revVaultCh) |
            for (@(_, RevVault) <- revVaultCh) {{
                @RevVault!("findOrCreate", "{}", *vaultCh) |
                for (@either <- vaultCh) {{
                    match either {{
                        (true, vault) => {{
                            @vault!("balance", *balanceCh) |
                            for (@balance <- balanceCh) {{
                                return!(balance)
                            }}
                        }}
                        (false, errorMsg) => {{
                            return!(errorMsg)
                        }}
                    }}
                }}
            }}
        }}"#,
        args.address
    );

    let start_time = Instant::now();

    match f1r3fly_api
        .exploratory_deploy(&rholang_query, None, false)
        .await
    {
        Ok((result, block_info)) => {
            let duration = start_time.elapsed();
            println!("✅ Wallet balance retrieved successfully!");
            println!("⏱️  Time taken: {:.2?}", duration);
            println!("💰 Balance for {}: {} REV", args.address, result);
            println!("📊 {}", block_info);
        }
        Err(e) => {
            println!("❌ Failed to get wallet balance!");
            println!("Error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

pub async fn bond_status_command(args: &BondStatusArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🔍 Checking bond status for public key: {}",
        args.public_key
    );

    let url = format!("http://{}:{}/api/explore-deploy", args.host, args.port);
    let client = reqwest::Client::new();

    // Get all bonds first, then check if our public key is in there
    let rholang_query = r#"new return, rl(`rho:registry:lookup`), poSCh in { rl!(`rho:rchain:pos`, *poSCh) | for(@(_, PoS) <- poSCh) { @PoS!("getBonds", *return) } }"#;

    let body = serde_json::json!({
        "term": rholang_query
    });

    let start_time = Instant::now();

    match client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(response) => {
            let duration = start_time.elapsed();
            if response.status().is_success() {
                let bonds_text = response.text().await?;
                let bonds_json: serde_json::Value = serde_json::from_str(&bonds_text)?;

                println!("✅ Bond information retrieved successfully!");
                println!("⏱️  Time taken: {:.2?}", duration);

                // Check if the public key exists in the bonds
                let is_bonded = check_if_key_is_bonded(&bonds_json, &args.public_key);

                if is_bonded {
                    println!("🔗 ✅ Validator is BONDED");
                    println!("📍 Public key: {}", args.public_key);
                } else {
                    println!("🔗 ❌ Validator is NOT BONDED");
                    println!("📍 Public key: {}", args.public_key);
                }

                println!("\n📊 Full bonds data:");
                println!("{}", serde_json::to_string_pretty(&bonds_json)?);
            } else {
                println!("❌ Failed to get bond status: HTTP {}", response.status());
                println!("Error: {}", response.text().await?);
            }
        }
        Err(e) => {
            println!("❌ Connection failed!");
            println!("Error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

fn check_if_key_is_bonded(bonds_json: &serde_json::Value, target_public_key: &str) -> bool {
    // Navigate through the JSON structure to find bonds
    // The structure is: block.bonds[].validator
    if let Some(block) = bonds_json.get("block") {
        if let Some(bonds_array) = block.get("bonds") {
            if let Some(bonds) = bonds_array.as_array() {
                // Check each bond entry
                for bond in bonds {
                    if let Some(validator) = bond.get("validator") {
                        if let Some(validator_key) = validator.as_str() {
                            if validator_key == target_public_key {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

pub async fn metrics_command(args: &HttpArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Getting node metrics from {}:{}", args.host, args.port);

    let url = format!("http://{}:{}/metrics", args.host, args.port);
    let client = reqwest::Client::new();

    let start_time = Instant::now();

    match client.get(&url).send().await {
        Ok(response) => {
            let duration = start_time.elapsed();
            if response.status().is_success() {
                let metrics_text = response.text().await?;

                println!("✅ Node metrics retrieved successfully!");
                println!("⏱️  Time taken: {:.2?}", duration);
                println!("📊 Node Metrics:");

                // Filter and display key metrics
                let lines: Vec<&str> = metrics_text
                    .lines()
                    .filter(|line| {
                        line.contains("peers")
                            || line.contains("blocks")
                            || line.contains("consensus")
                            || line.contains("casper")
                            || line.contains("rspace")
                    })
                    .collect();

                if lines.is_empty() {
                    println!("📊 All Metrics:");
                    println!("{}", metrics_text);
                } else {
                    println!("📊 Key Metrics (peers, blocks, consensus):");
                    for line in lines {
                        println!("{}", line);
                    }
                    println!("\n💡 Use --verbose flag (if implemented) to see all metrics");
                }
            } else {
                println!("❌ Failed to get metrics: HTTP {}", response.status());
                println!("Error: {}", response.text().await?);
            }
        }
        Err(e) => {
            println!("❌ Connection failed!");
            println!("Error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

pub async fn network_health_command(
    args: &NetworkHealthArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Checking F1r3fly network health");

    let mut ports_to_check = Vec::new();

    if args.standard_ports {
        // Standard F1r3fly shard ports from the docker configuration
        ports_to_check.extend_from_slice(&[
            (40403, "Bootstrap"),
            (40413, "Validator1"),
            (40423, "Validator2"),
            (40433, "Validator3"),
            (40453, "Observer"),
        ]);
    }

    // Add custom ports if specified
    if let Some(custom_ports_str) = &args.custom_ports {
        for port_str in custom_ports_str.split(',') {
            if let Ok(port) = port_str.trim().parse::<u16>() {
                ports_to_check.push((port, "Custom"));
            }
        }
    }

    if ports_to_check.is_empty() {
        println!("❌ No ports specified to check");
        return Ok(());
    }

    let client = reqwest::Client::new();
    let mut healthy_nodes = 0;
    let mut total_nodes = 0;
    let mut all_peers = Vec::new();

    println!("🔍 Checking {} nodes...\n", ports_to_check.len());

    for (port, node_type) in ports_to_check {
        total_nodes += 1;
        let url = format!("http://{}:{}/status", args.host, port);

        print!("📊 {} ({}:{}): ", node_type, args.host, port);

        match client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    match response.text().await {
                        Ok(status_text) => {
                            match serde_json::from_str::<serde_json::Value>(&status_text) {
                                Ok(status_json) => {
                                    healthy_nodes += 1;

                                    let peers = status_json
                                        .get("peers")
                                        .and_then(|p| p.as_u64())
                                        .unwrap_or(0);

                                    all_peers.push(peers);

                                    println!("✅ HEALTHY ({} peers)", peers);
                                }
                                Err(_) => println!("❌ Invalid JSON response"),
                            }
                        }
                        Err(_) => println!("❌ Failed to read response"),
                    }
                } else {
                    println!("❌ HTTP {}", response.status());
                }
            }
            Err(_) => println!("❌ Connection failed"),
        }
    }

    println!("\n📈 Network Health Summary:");
    println!("✅ Healthy nodes: {}/{}", healthy_nodes, total_nodes);

    if healthy_nodes > 0 {
        let avg_peers = all_peers.iter().sum::<u64>() as f64 / all_peers.len() as f64;
        let min_peers = all_peers.iter().min().unwrap_or(&0);
        let max_peers = all_peers.iter().max().unwrap_or(&0);

        println!(
            "👥 Peer connections: avg={:.1}, min={}, max={}",
            avg_peers, min_peers, max_peers
        );

        if healthy_nodes == total_nodes && *min_peers >= (total_nodes as u64 - 1) {
            println!("🎉 Network is FULLY CONNECTED and HEALTHY!");
        } else if healthy_nodes == total_nodes {
            println!("⚠️  All nodes healthy but some peer connections may be missing");
        } else {
            println!("⚠️  Some nodes are unhealthy - check individual node logs");
        }
    } else {
        println!("❌ No healthy nodes found - check if network is running");
    }

    Ok(())
}

pub async fn last_finalized_block_command(
    args: &HttpArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🔍 Getting last finalized block from {}:{}",
        args.host, args.port
    );

    let url = format!(
        "http://{}:{}/api/last-finalized-block",
        args.host, args.port
    );
    let client = reqwest::Client::new();

    let start_time = Instant::now();

    match client.get(&url).send().await {
        Ok(response) => {
            let duration = start_time.elapsed();
            if response.status().is_success() {
                let block_text = response.text().await?;
                let block_json: serde_json::Value = serde_json::from_str(&block_text)?;

                println!("✅ Last finalized block retrieved successfully!");
                println!("⏱️  Time taken: {:.2?}", duration);

                // Extract key information from blockInfo
                let block_info = block_json.get("blockInfo");

                let block_hash = block_info
                    .and_then(|info| info.get("blockHash"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");

                let block_number = block_info
                    .and_then(|info| info.get("blockNumber"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                let timestamp = block_info
                    .and_then(|info| info.get("timestamp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                // Get deploy count from blockInfo (it's already calculated)
                let deploy_count = block_info
                    .and_then(|info| info.get("deployCount"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                let shard_id = block_info
                    .and_then(|info| info.get("shardId"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");

                let fault_tolerance = block_info
                    .and_then(|info| info.get("faultTolerance"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                println!("🧱 Last Finalized Block Summary:");
                println!("   📋 Block Number: {}", block_number);
                println!("   🔗 Block Hash: {}", block_hash);
                println!("   ⏰ Timestamp: {}", timestamp);
                println!("   📦 Deploy Count: {}", deploy_count);
                println!("   🔧 Shard ID: {}", shard_id);
                println!("   ⚖️  Fault Tolerance: {:.6}", fault_tolerance);
            } else {
                println!(
                    "❌ Failed to get last finalized block: HTTP {}",
                    response.status()
                );
                println!("Error: {}", response.text().await?);
            }
        }
        Err(e) => {
            println!("❌ Connection failed!");
            println!("Error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

pub async fn show_main_chain_command(
    args: &ShowMainChainArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🔗 Getting main chain blocks from {}:{}",
        args.host, args.port
    );
    println!("📊 Depth: {} blocks", args.depth);

    // Initialize the F1r3fly API client
    let f1r3fly_api = F1r3flyApi::new(&args.private_key, &args.host, args.port);

    let start_time = Instant::now();

    match f1r3fly_api.show_main_chain(args.depth).await {
        Ok(blocks) => {
            let duration = start_time.elapsed();
            println!("✅ Main chain blocks retrieved successfully!");
            println!("⏱️  Time taken: {:.2?}", duration);
            println!("📋 Found {} blocks in main chain", blocks.len());
            println!();

            if blocks.is_empty() {
                println!("🔍 No blocks found in main chain");
            } else {
                println!("🧱 Main Chain Blocks:");
                for (index, block) in blocks.iter().enumerate() {
                    println!("📦 Block #{}:", block.block_number);
                    println!("   🔗 Hash: {}", block.block_hash);
                    let sender_display = if block.sender.len() >= 16 {
                        format!("{}...", &block.sender[..16])
                    } else if block.sender.is_empty() {
                        "(genesis)".to_string()
                    } else {
                        block.sender.clone()
                    };
                    println!("   👤 Sender: {}", sender_display);
                    println!("   ⏰ Timestamp: {}", block.timestamp);
                    println!("   📦 Deploy Count: {}", block.deploy_count);
                    println!("   ⚖️  Fault Tolerance: {:.6}", block.fault_tolerance);
                    if index < blocks.len() - 1 {
                        println!("   ⬇️");
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to get main chain blocks!");
            println!("Error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

pub async fn validator_status_command(
    args: &ValidatorStatusArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Checking validator status for: {}", args.public_key);

    let f1r3fly_api = F1r3flyApi::new(
        "5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657", // Bootstrap private key
        &args.host,
        args.port,
    );

    let start_time = Instant::now();

    // Query 1: Get all bonds to check if validator is bonded
    let bonds_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getBonds", *return)
        }
    }"#;

    // Query 2: Get active validators to check if validator is active
    let active_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getActiveValidators", *return)
        }
    }"#;

    // Query 3: Get quarantine length for timing calculations
    let quarantine_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getQuarantineLength", *return)
        }
    }"#;

    // Use HTTP API for PoS contract queries (like bonds/network-consensus commands)
    let client = reqwest::Client::new();
    let http_url = format!("http://{}:40453/api/explore-deploy", args.host); // Use HTTP port

    // Execute all queries and get current block
    let (bonds_result, active_result, quarantine_result, current_block) = tokio::try_join!(
        query_pos_http(&client, &http_url, bonds_query),
        query_pos_http(&client, &http_url, active_query),
        f1r3fly_api.exploratory_deploy(quarantine_query, None, false),
        f1r3fly_api.get_current_block_number()
    )?;

    let duration = start_time.elapsed();

    // Parse results using HTTP response format
    let bonds_data = bonds_result;
    let active_data = active_result;

    // Parse quarantine length
    let quarantine_length = quarantine_result.0.trim().parse::<i64>().map_err(|e| {
        format!(
            "Failed to parse quarantine length: '{}'. Error: {}",
            quarantine_result.0, e
        )
    })?;

    println!("✅ Validator status retrieved successfully!");
    println!("⏱️  Time taken: {:.2?}", duration);
    println!();

    // Parse bonded validators from HTTP response
    let bonded_validators = parse_validator_data(&bonds_data);
    let active_validators = parse_validator_data(&active_data);

    // Check bonded status
    let is_bonded = bonded_validators.contains(&args.public_key);

    if is_bonded {
        println!("✅ BONDED: Validator is bonded to the network");

        // Try to extract bond amount from JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&bonds_data) {
            if let Some(block) = json.get("block") {
                if let Some(bonds) = block.get("bonds") {
                    if let Some(bonds_array) = bonds.as_array() {
                        for bond in bonds_array {
                            if let Some(validator) = bond.get("validator").and_then(|v| v.as_str())
                            {
                                if validator == args.public_key {
                                    if let Some(stake) = bond.get("stake").and_then(|s| s.as_i64())
                                    {
                                        println!("   Stake Amount: {} REV", stake);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        println!("❌ NOT BONDED: Validator is not bonded to the network");
    }

    // Check active status
    let is_active = active_validators.contains(&args.public_key);
    if is_active {
        println!("✅ ACTIVE: Validator is actively participating in consensus");
    } else if is_bonded {
        println!("⏳ QUARANTINE: Validator is bonded but not yet active (in quarantine period)");
    } else {
        println!("❌ INACTIVE: Validator is not participating in consensus");
    }

    println!();
    println!("📊 Summary:");
    println!("   Public Key: {}", args.public_key);
    println!("   Bonded: {}", if is_bonded { "✅ Yes" } else { "❌ No" });
    println!("   Active: {}", if is_active { "✅ Yes" } else { "❌ No" });

    if is_bonded && !is_active {
        println!("   Status: ⏳ In quarantine period");
        println!("   Quarantine Length: {} blocks", quarantine_length);
        println!("   Current Block: {}", current_block);
        println!("   Next: Wait for epoch transition to become active");
    } else if is_active {
        println!("   Status: ✅ Fully operational");
    } else {
        println!("   Status: ❌ Not participating");
        println!("   Next: Bond validator to network first");
    }

    Ok(())
}

pub async fn epoch_info_command(args: &PosQueryArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🔍 Getting current epoch information from {}:{}",
        args.host, args.port
    );

    let f1r3fly_api = F1r3flyApi::new(
        "5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657", // Bootstrap private key
        &args.host,
        args.port,
    );

    let start_time = Instant::now();

    // Query epoch and quarantine lengths from PoS contract
    let epoch_length_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getEpochLength", *return)
        }
    }"#;

    let quarantine_length_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getQuarantineLength", *return)
        }
    }"#;

    // Get all data in parallel for efficiency
    let (epoch_result, quarantine_result, current_block, recent_blocks) = tokio::try_join!(
        f1r3fly_api.exploratory_deploy(epoch_length_query, None, false),
        f1r3fly_api.exploratory_deploy(quarantine_length_query, None, false),
        f1r3fly_api.get_current_block_number(),
        f1r3fly_api.show_main_chain(5)
    )?;

    let duration = start_time.elapsed();

    // Parse epoch length from PoS contract result
    let epoch_length = epoch_result.0.trim().parse::<i64>().map_err(|e| {
        format!(
            "Failed to parse epoch length from PoS contract: '{}'. Error: {}",
            epoch_result.0, e
        )
    })?;

    // Parse quarantine length from PoS contract result
    let quarantine_length = quarantine_result.0.trim().parse::<i64>().map_err(|e| {
        format!(
            "Failed to parse quarantine length from PoS contract: '{}'. Error: {}",
            quarantine_result.0, e
        )
    })?;

    // Calculate epoch information
    let current_epoch = current_block / epoch_length;
    let epoch_start_block = current_epoch * epoch_length;
    let epoch_end_block = epoch_start_block + epoch_length - 1;
    let blocks_into_epoch = current_block - epoch_start_block;
    let blocks_remaining = epoch_length - blocks_into_epoch;

    println!("✅ Epoch information retrieved successfully!");
    println!("⏱️  Time taken: {:.2?}", duration);
    println!();

    println!("📊 Current Epoch Status:");
    println!("   Current Block: {}", current_block);
    println!("   Current Epoch: {}", current_epoch);
    println!("   Epoch Length: {} blocks", epoch_length);
    println!("   Quarantine Length: {} blocks", quarantine_length);
    println!();

    println!("🎯 Epoch {} Details:", current_epoch);
    println!("   Start Block: {}", epoch_start_block);
    println!("   End Block: {}", epoch_end_block);
    println!(
        "   Progress: {}/{} blocks ({:.1}%)",
        blocks_into_epoch,
        epoch_length,
        (blocks_into_epoch as f64 / epoch_length as f64) * 100.0
    );
    println!("   Remaining: {} blocks", blocks_remaining);
    println!();

    if blocks_remaining <= 100 {
        println!(
            "⚠️  Epoch transition approaching! ({} blocks remaining)",
            blocks_remaining
        );
    } else if blocks_into_epoch <= 100 {
        println!(
            "🆕 Recently started new epoch! ({} blocks into epoch)",
            blocks_into_epoch
        );
    }

    println!("🔄 Next Epoch ({}):", current_epoch + 1);
    println!("   Will start at block: {}", epoch_end_block + 1);
    println!("   Estimated blocks until transition: {}", blocks_remaining);

    // Show recent block activity
    println!();
    println!("📈 Recent Block Activity:");
    for (_, block) in recent_blocks.iter().enumerate() {
        let block_epoch = block.block_number / epoch_length;
        let epoch_marker = if block_epoch != current_epoch {
            format!(" (Epoch {})", block_epoch)
        } else {
            String::new()
        };

        println!(
            "   Block {}: {} finalized{}",
            block.block_number,
            "✅", // All main chain blocks are considered finalized
            epoch_marker
        );
    }

    Ok(())
}

pub async fn epoch_rewards_command(args: &PosQueryArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🔍 Getting current epoch rewards from {}:{}",
        args.host, args.port
    );

    let f1r3fly_api = F1r3flyApi::new(
        "5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657",
        &args.host,
        args.port,
    );

    let rewards_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getCurrentEpochRewards", *return)
        }
    }"#;

    let start_time = Instant::now();

    match f1r3fly_api
        .exploratory_deploy(rewards_query, None, false)
        .await
    {
        Ok((result, block_info)) => {
            let duration = start_time.elapsed();
            println!("✅ Epoch rewards retrieved successfully!");
            println!("⏱️  Time taken: {:.2?}", duration);
            println!("💰 Current Epoch Rewards:");
            println!("{}", result);
            println!("📊 {}", block_info);
        }
        Err(e) => {
            println!("❌ Failed to get epoch rewards!");
            println!("Error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

// Helper function for HTTP PoS queries
async fn query_pos_http(
    client: &reqwest::Client,
    url: &str,
    query: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let body = serde_json::json!({
        "term": query
    });

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    if response.status().is_success() {
        let response_text = response.text().await?;
        let response_json: serde_json::Value = serde_json::from_str(&response_text)?;

        // Extract the actual result from the response
        if let Some(block) = response_json.get("block") {
            if let Some(result) = block.get("postBlockData") {
                return Ok(result.to_string());
            }
        }

        // Fallback to full response if structure is different
        Ok(response_text)
    } else {
        Err(format!("HTTP error: {}", response.status()).into())
    }
}

pub async fn network_consensus_command(
    args: &PosQueryArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🌐 Getting network-wide consensus overview from {}:{}",
        args.host, args.port
    );

    let f1r3fly_api = F1r3flyApi::new(
        "5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657",
        &args.host,
        args.port,
    );

    let start_time = Instant::now();

    // Get all validator info in parallel using HTTP API for PoS queries
    let client = reqwest::Client::new();
    let http_url = format!("http://{}:40453/api/explore-deploy", args.host); // Use HTTP port

    let bonds_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getBonds", *return)
        }
    }"#;

    let active_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getActiveValidators", *return)
        }
    }"#;

    let quarantine_query = r#"new return, rl(`rho:registry:lookup`), poSCh in {
        rl!(`rho:rchain:pos`, *poSCh) |
        for(@(_, PoS) <- poSCh) {
            @PoS!("getQuarantineLength", *return)
        }
    }"#;

    let (bonds_result, active_result, quarantine_result, current_block) = tokio::try_join!(
        query_pos_http(&client, &http_url, bonds_query),
        query_pos_http(&client, &http_url, active_query),
        f1r3fly_api.exploratory_deploy(quarantine_query, None, false),
        f1r3fly_api.get_current_block_number()
    )?;

    let duration = start_time.elapsed();

    println!("✅ Network consensus data retrieved successfully!");
    println!("⏱️  Time taken: {:.2?}", duration);
    println!();

    // Parse and display network health
    let bonds_data = bonds_result;
    let active_data = active_result;

    // Parse quarantine length
    let quarantine_length = quarantine_result.0.trim().parse::<i64>().map_err(|e| {
        format!(
            "Failed to parse quarantine length: '{}'. Error: {}",
            quarantine_result.0, e
        )
    })?;

    // Parse validator data from HTTP response
    let bonded_validators = parse_validator_data(&bonds_data);
    let active_validators = parse_validator_data(&active_data);

    let total_bonded = bonded_validators.len();
    let total_active = active_validators.len();
    let quarantine_count = total_bonded - total_active;

    println!("📊 Network Consensus Health:");
    println!("   Current Block: {}", current_block);
    println!("   Total Bonded Validators: {}", total_bonded);
    println!("   Active Validators: {}", total_active);
    println!("   Validators in Quarantine: {}", quarantine_count);
    println!("   Quarantine Length: {} blocks", quarantine_length);

    let consensus_health = if total_active >= 3 {
        "🟢 Healthy"
    } else if total_active >= 1 {
        "🟡 Limited"
    } else {
        "🔴 Critical"
    };

    println!("   Consensus Status: {}", consensus_health);

    if total_active > 0 {
        let participation_rate = (total_active as f64 / total_bonded as f64) * 100.0;
        println!("   Participation Rate: {:.1}%", participation_rate);
    }

    Ok(())
}

fn parse_validator_data(json_str: &str) -> Vec<String> {
    // Parse JSON response from HTTP PoS query
    let mut validators = Vec::new();

    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
        // Extract from the HTTP response structure: response.block.bonds[] or response.block (for active validators)
        if let Some(block) = json.get("block") {
            // For bonds data: extract from bonds array
            if let Some(bonds) = block.get("bonds") {
                if let Some(bonds_array) = bonds.as_array() {
                    for bond in bonds_array {
                        if let Some(validator) = bond.get("validator") {
                            if let Some(validator_str) = validator.as_str() {
                                validators.push(validator_str.to_string());
                            }
                        }
                    }
                }
            }

            // For active validators data: might be in a different format
            // The response structure may vary for getActiveValidators vs getBonds
            if validators.is_empty() {
                // Try to extract directly from block object or other possible structures
                if let Some(obj) = block.as_object() {
                    for (key, _value) in obj {
                        // Public keys are typically 64-character hex strings
                        if key.len() == 64 && key.chars().all(|c| c.is_ascii_hexdigit()) {
                            validators.push(key.clone());
                        }
                    }
                }
            }
        }
    }

    validators.sort();
    validators.dedup();
    validators
}
