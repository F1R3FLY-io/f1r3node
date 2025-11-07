# Genesis Files Format

Complete specification for `bonds.txt` and `wallets.txt` files used in genesis block creation.

## Table of Contents

1. [Overview](#overview)
2. [bonds.txt Format](#bondstxt-format)
3. [wallets.txt Format](#walletstxt-format)
4. [File Locations](#file-locations)
5. [Generation Strategies](#generation-strategies)
6. [Examples](#examples)

## Overview

Genesis files define the initial state of the blockchain:
- **bonds.txt**: Defines which validators exist at genesis and their stake amounts
- **wallets.txt**: Defines initial token distribution to wallet addresses

Both files must be present at `<data-dir>/genesis/` for all nodes in the shard.

**Critical**: All nodes in a shard must have **identical** genesis files, or they will create different genesis blocks and fail to reach consensus.

## bonds.txt Format

### Specification

Plain text file with one validator per line:

```
<public-key> <bond-amount>
```

- **`<public-key>`**: 130-character hex string (65 bytes, secp256k1 public key in uncompressed format)
- **`<bond-amount>`**: Integer representing bonded stake (no decimal point)
- **Separator**: Single space character
- **Line ending**: Unix LF (`\n`)

### Public Key Format

The public key is the secp256k1 public key used for **block signing** (validator key), **NOT** the TLS key.

Format: `04` prefix + 64-byte (128 hex char) public key

Example:
```
04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420
```

### Bond Amount Units

Bond amounts are in the smallest unit (comparable to Wei in Ethereum or Satoshi in Bitcoin).

- No decimal point
- Pure integer
- Common test values: `1000`, `100000000000` (100 billion), etc.

### Example bonds.txt

```
04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420 1000
04837a4cff833e3157e3135d7b40b8e1f33c6e6b5a4342b9fc784230ca4c4f9d356f258debef56ad4984726d6ab3e7709e1632ef079b4bcd653db00b68b2df065f 1000
0457febafcc25dd34ca5e5c025cd445f60e5ea6918931a54eb8c3a204f51760248090b0c757c2bdad7b8c4dca757e109f8ef64737d90712724c8216c94b4ae661c 1000
```

This defines a 3-validator shard with equal bonds of 1000 units each.

### Important Notes

1. **All validators must be listed**: Every node with `--genesis-validator` flag must have its public key in bonds.txt
2. **Bootstrap node NOT included**: If using a separate ceremony master/bootstrap node that doesn't validate, its key should NOT be in bonds.txt
3. **Order doesn't matter**: Validators can be listed in any order
4. **No empty lines**: File should not have trailing empty lines
5. **No comments**: File does not support comment syntax

### Configuration Reference

Config file setting (usually doesn't need to be changed):

```hocon
casper {
  genesis-block-data {
    bonds-file = ${casper.genesis-block-data.genesis-data-dir}/bonds.txt
    bond-minimum = 1
    bond-maximum = 9223372036854775807
  }
}
```

CLI flags:
```bash
--bonds-file=/var/lib/rnode/genesis/bonds.txt
--bond-minimum=1
--bond-maximum=9223372036854775807
```

## wallets.txt Format

### Specification

Plain text file with one wallet per line:

```
<rev-address>,<balance>
```

- **`<rev-address>`**: REV address format (51-character base58-encoded address)
- **`<balance>`**: Integer representing token balance (no decimal point)
- **Separator**: Comma (`,`) with no spaces
- **Line ending**: Unix LF (`\n`)

### REV Address Format

REV addresses are base58-encoded and start with `1111`:

Example:
```
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g
```

These addresses are derived from public keys using base58check encoding.

### Balance Units

Like bonds, balances are in the smallest unit:

- No decimal point
- Pure integer
- Example: `50000000000000000` (50 quadrillion smallest units)

### Example wallets.txt

```
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g,50000000000000000
111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA,50000000000000000
111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH,50000000000000000
1111LAd2PWaHsw84gxarNx99YVK2aZhCThhrPsWTV7cs1BPcvHftP,50000000000000
1111ocWgUJb5QqnYCvKiPtzcmMyfvD3gS5Eg84NtaLkUtRfw3TDS8,50000000000000000
```

### Important Notes

1. **Separator is comma**: No spaces around the comma
2. **Can have many wallets**: File can have thousands of entries (mainnet wallets.txt has 11,000+ entries)
3. **Order doesn't matter**: Wallets can be listed in any order
4. **No empty lines**: File should not have trailing empty lines
5. **Use for test accounts**: Create wallets for testing/demo purposes

### Configuration Reference

Config file setting:

```hocon
casper {
  genesis-block-data {
    wallets-file = ${casper.genesis-block-data.genesis-data-dir}/wallets.txt
  }
}
```

CLI flag:
```bash
--wallets-file=/var/lib/rnode/genesis/wallets.txt
```

## File Locations

### Standard Locations

**Default path:** `<data-dir>/genesis/`

For Docker profile: `/var/lib/rnode/genesis/`

```
/var/lib/rnode/
├── genesis/
│   ├── bonds.txt
│   └── wallets.txt
├── node.key.pem
└── node.certificate.pem
```

### Docker Volume Mounting

Example from docker-compose.yml:

```yaml
volumes:
  - ./genesis/wallets.txt:/var/lib/rnode/genesis/wallets.txt
  - ./genesis/bonds.txt:/var/lib/rnode/genesis/bonds.txt
```

## Generation Strategies

### For a 3-Node Demo Shard

**Strategy:**

1. Generate 3 validator key pairs (secp256k1)
2. Extract public keys → bonds.txt (equal stakes)
3. Generate 3-5 test wallet addresses
4. Assign large test balances → wallets.txt

**Example bond amounts for testing:**
- Equal stakes: `1000` each (simple)
- Realistic stakes: `100000000000` (100 billion) each
- Varied stakes: `150000000000`, `100000000000`, `100000000000`

**Example wallet balances for testing:**
- Development wallets: `50000000000000000` (50 quadrillion) each
- Small test amounts: `50000000000000` (50 trillion)

### For Production Shards

**Bonds:**
- Derive from actual validator commitments
- Ensure minimum meets `bond-minimum` config (default: 1)
- Ensure maximum below `bond-maximum` config (default: Long.MaxValue)

**Wallets:**
- Import from previous blockchain state (for migrations)
- Or define initial distribution per tokenomics model
- Include foundation, team, investor addresses

### Auto-generation Mode

RNode can auto-generate bonds if no file is provided:

```hocon
casper {
  genesis-ceremony {
    autogen-shard-size = 5  # Generate 5 random validators
  }
}
```

Private keys are saved to `<genesis-path>/<public_key>.sk`.

**Use case**: Quick local testing only. Not for production.

## Examples

### Minimal 3-Validator Shard

**bonds.txt:**
```
04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420 1000
04837a4cff833e3157e3135d7b40b8e1f33c6e6b5a4342b9fc784230ca4c4f9d356f258debef56ad4984726d6ab3e7709e1632ef079b4bcd653db00b68b2df065f 1000
0457febafcc25dd34ca5e5c025cd445f60e5ea6918931a54eb8c3a204f51760248090b0c757c2bdad7b8c4dca757e109f8ef64737d90712724c8216c94b4ae661c 1000
```

**wallets.txt:**
```
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g,50000000000000000
111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA,50000000000000000
111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH,50000000000000000
```

### Single Validator (Standalone Testing)

**bonds.txt:**
```
04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420 1000000000
```

**wallets.txt:**
```
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g,1000000000000000000
```

Use with `--standalone` flag and `required-signatures = 0`.

### Varied Stake Distribution

**bonds.txt:**
```
04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420 150500000000000
04837a4cff833e3157e3135d7b40b8e1f33c6e6b5a4342b9fc784230ca4c4f9d356f258debef56ad4984726d6ab3e7709e1632ef079b4bcd653db00b68b2df065f 150500000000000
0457febafcc25dd34ca5e5c025cd445f60e5ea6918931a54eb8c3a204f51760248090b0c757c2bdad7b8c4dca757e109f8ef64737d90712724c8216c94b4ae661c 151000000000000
04d26c6103d7269773b943d7a9c456f9eb227e0d8b1fe30bccee4fca963f4446e3385d99f6386317f2c1ad36b9e6b0d5f97bb0a0041f05781c60a5ebca124a251d 150000000000000
04fb67c987ae5670634b0a0751546df218c378f3ae371477c9e9a366b1f80b74510bd775f5857a17abb30432e2c70a686a662b915905f4ea77491954eb32061d64 151000000000000
```

Different stakes give validators different voting weights.

## Validation

### bonds.txt Validation

Check your bonds.txt:

```bash
# Each line should be: 130 hex chars + space + integer
awk '{print length($1), $2}' bonds.txt

# Should output: 130 <bond-amount> for each line
```

### wallets.txt Validation

Check your wallets.txt:

```bash
# Count addresses
wc -l wallets.txt

# Check format (should be: address,balance)
head -3 wallets.txt
```

### Common Errors

**bonds.txt:**
- Wrong public key length (must be exactly 130 hex chars)
- Missing space separator
- Decimal point in bond amount (must be integer)
- Including bootstrap node key (if it's not a validator)

**wallets.txt:**
- Spaces around comma
- Decimal point in balance
- Invalid REV address format
- Wrong separator (using space or semicolon instead of comma)

## Key Generation

See [KEY_GENERATION.md](./KEY_GENERATION.md) for detailed instructions on generating validator keys and wallet addresses for use in these files.

## Reference

- Configuration code: `/node/src/main/scala/coop/rchain/node/configuration/Configuration.scala`
- Bonds file parsing: Search for `bonds-file` in codebase
- Wallets file parsing: Search for `wallets-file` in codebase
- Example files: `/docker/genesis/bonds.txt` and `/docker/genesis/wallets.txt`
