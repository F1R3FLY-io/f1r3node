# Example Genesis Files

This directory contains example genesis files for a 3-validator shard.

## Files

- `bonds.txt` - 3 validators with equal stakes of 1000 units each
- `wallets.txt` - 5 test wallet addresses with generous test balances

## Usage

Copy these files to your deployment's genesis directory:

```bash
cp examples/genesis/*.txt /var/lib/rnode/genesis/
```

Or mount in Docker:

```yaml
volumes:
  - ./examples/genesis:/var/lib/rnode/genesis:ro
```

## Important Notes

1. **All nodes must have identical genesis files** - Same bonds.txt and wallets.txt, or they'll create different genesis blocks
2. **Validator keys must match** - Public keys in bonds.txt must match the --validator-private-key used by each validator
3. **Test addresses only** - These wallet addresses are public examples, never use for real value

## Validator Keys

The public keys in bonds.txt correspond to these test private keys:

- Validator 1: `357cdc4201a5650830e0bc5a03299a30038d9934ba4c7ab73ec164ad82471ff9`
- Validator 2: `2c02138097d019d263c1d5383fcaddb1ba6416a0f4e64e3a617fe3af45b7851d`
- Validator 3: `b67533f1f99c0ecaedb7d829e430b1c0e605bda10f339f65d5567cb5bd77cbcb`

**Warning**: These are public test keys for demo purposes only.

## Customization

To create your own genesis files:

1. Generate validator keys - see [KEY_GENERATION.md](../../KEY_GENERATION.md)
2. Create bonds.txt with your validator public keys
3. Generate/use wallet addresses for your use case
4. Update wallets.txt with appropriate balances

See [GENESIS_FILES_FORMAT.md](../../GENESIS_FILES_FORMAT.md) for format specifications.
