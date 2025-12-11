# F1R3FLY Docker Network

## Quick Start

### Standalone Node (Recommended for Development)
The standalone setup runs a **single-validator node** optimized for fast development with instant finalization. This is the simplest way to get started.

**Start standalone node:**
```bash
docker-compose -f standalone.yml up --pull always -d
```

**Follow logs:**
```bash
docker-compose -f standalone.yml logs -f
```

**Stop standalone node:**
```bash
docker-compose -f standalone.yml down
```

### Multi-Validator Network
For testing multi-node consensus and advanced scenarios, use the full shard network with one bootstrap, 3 validators and observer.

**Start the Network:**
```bash
docker-compose -f shard-with-autopropose.yml up --pull always -d
```

**Wait for Genesis (2-3 minutes):**

All nodes need to complete the genesis ceremony and transition to running state before the network is ready.
```bash
# Monitor all nodes until they output 'Making a transition to Running state.'
docker-compose -f shard-with-autopropose.yml logs -f | grep "Making a transition to Running state"
```

Once you see this message from all validators, the network is ready. Press `Ctrl+C` to stop watching logs.

**Follow logs for all services in the shard:**
```bash
docker-compose -f shard-with-autopropose.yml logs -f
```

**Follow logs for a specific node:**
```bash
# For validator1
docker-compose -f shard-with-autopropose.yml logs -f validator1

# For validator2
docker-compose -f shard-with-autopropose.yml logs -f validator2

# For validator3
docker-compose -f shard-with-autopropose.yml logs -f validator3

# For bootstrap node
docker-compose -f shard-with-autopropose.yml logs -f bootstrap

# For observer
docker-compose -f shard-with-autopropose.yml logs -f readonly
```

**Stop the Network:**
```bash
docker-compose -f shard-with-autopropose.yml down
```

### Fresh Restart
When the network runs, a `data/` directory is created to store blockchain state and node data. 

**To completely reset the network to genesis state:**
```bash
# Remove all blockchain data 
rm -rf data/
```

**⚠️ Warning**: Removing the `data/` directory will permanently delete all blockchain history, blocks, and state.

## Adding Validator

See: https://github.com/F1R3FLY-io/rust-client/blob/main/VALIDATOR4_BONDING_GUIDE.md

## Genesis Configuration

### Wallets.txt - Funded Accounts
The following wallets are included in `genesis/wallets.txt` and **have funds available on network startup**:
- **Bootstrap Node** - Has initial REV balance for network operations
- **Validator_1** - Funded for transaction fees and operations  
- **Validator_2** - Funded for transaction fees and operations
- **Validator_3** - Funded for transaction fees and operations

### Bonds.txt - Network Validators
The following validators are included in `genesis/bonds.txt` and participate in consensus:
- **Validator_1** - Bonded with 1000 stake
- **Validator_2** - Bonded with 1000 stake  
- **Validator_3** - Bonded with 1000 stake

**Note**: Bootstrap node and Validator_4 are **not** in bonds.txt and do not participate in consensus validation.

## Interact with Node

Rust client: https://github.com/F1R3FLY-io/rust-client

## Wallet Information

### Standalone Node
Uses the same credentials as Bootstrap Node:
- **Private Key**: `5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657`
- **Public Key**: `04ffc016579a68050d655d55df4e09f04605164543e257c8e6df10361e6068a5336588e9b355ea859c5ab4285a5ef0efdf62bc28b80320ce99e26bb1607b3ad93d`
- **ETH**: `fac7dde9d0fa1df6355bd1382fe75ba0c50e8840`
- **REV**: `1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g`
- **Initial Balance**: 5,000,000,000 REV
- **Bond Amount**: 1,000 REV

### Bootstrap Node
- **Private Key**: `5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657`
- **Public Key**: `04ffc016579a68050d655d55df4e09f04605164543e257c8e6df10361e6068a5336588e9b355ea859c5ab4285a5ef0efdf62bc28b80320ce99e26bb1607b3ad93d`
- **ETH**: `fac7dde9d0fa1df6355bd1382fe75ba0c50e8840`
- **REV**: `1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g`

### Validator_1
- **Private Key**: `357cdc4201a5650830e0bc5a03299a30038d9934ba4c7ab73ec164ad82471ff9`
- **Public Key**: `04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420`
- **ETH**: `a77c116ce0ebe1331487638233bb52ba6b277da7`
- **REV**: `111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA`

### Validator_2
- **Private Key**: `2c02138097d019d263c1d5383fcaddb1ba6416a0f4e64e3a617fe3af45b7851d`
- **Public Key**: `04837a4cff833e3157e3135d7b40b8e1f33c6e6b5a4342b9fc784230ca4c4f9d356f258debef56ad4984726d6ab3e7709e1632ef079b4bcd653db00b68b2df065f`
- **ETH**: `df00c6395a23e9b2b8780de9a93c9522512947c3`
- **REV**: `111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH`

### Validator_3
- **Private Key**: `b67533f1f99c0ecaedb7d829e430b1c0e605bda10f339f65d5567cb5bd77cbcb`
- **Public Key**: `0457febafcc25dd34ca5e5c025cd445f60e5ea6918931a54eb8c3a204f51760248090b0c757c2bdad7b8c4dca757e109f8ef64737d90712724c8216c94b4ae661c`
- **ETH**: `ca778c4ecf5c6eb285a86cedd4aaf5167f4eae13`
- **REV**: `1111LAd2PWaHsw84gxarNx99YVK2aZhCThhrPsWTV7cs1BPcvHftP`

### Validator_4
- **Private Key**: `5ff3514bf79a7d18e8dd974c699678ba63b7762ce8d78c532346e52f0ad219cd`
- **Public Key**: `04d26c6103d7269773b943d7a9c456f9eb227e0d8b1fe30bccee4fca963f4446e3385d99f6386317f2c1ad36b9e6b0d5f97bb0a0041f05781c60a5ebca124a251d`
- **ETH**: `0cab9328d6d896e5159a1f70bc377e261ded7414`
- **REV**: `1111La6tHaCtGjRiv4wkffbTAAjGyMsVhzSUNzQxH1jjZH9jtEi3M`