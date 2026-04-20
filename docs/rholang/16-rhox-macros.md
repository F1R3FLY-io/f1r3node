# Rhox Macros

`.rhox` files are Rholang templates with macro parameters. They allow the same contract to be instantiated with different configuration values at deployment time.

## Format

A `.rhox` file is standard Rholang source with parameter placeholders. The parameters are documented in comments at the top of the file and substituted before deployment.

## Example: PoS.rhox

The Proof-of-Stake contract (`casper/src/main/resources/PoS.rhox`) uses macro parameters for configuration:

```rho
// Rholang macro parameters:
// minimumBond - the minimum bond allowed by the PoS
// maximumBond - the maximum bond allowed by PoS
// initialBonds - the initial bonds map
// epochLength - the length of the validation epoch in blocks
// quarantineLength - the length of the quarantine time in blocks
// numberOfActiveValidators - max number of active validators in a shard
// posMultiSigPublicKeys - public keys
// posMultiSigQuorum - how many confirmations are necessary for multi-sig vault
```

### PoS Contract State

The PoS contract manages:

```
"allBonds"            : Map[Validator, bonds]
"activeValidators"    : Set[Validator]
"withdrawers"         : Map[Validator, (bonds, quarantine)]
"committedRewards"    : Map[Validator, rewards]
"pendingWithdrawers"  : Map[Validator, quarantine]
```

### PoS Lifecycle

1. **UserDeploy**: precharge (user pays posVault), refund (posVault returns excess)
2. **Bonding**: User transfers bond amount to posVault
3. **Withdraw (Unbond)**: Added to pendingWithdrawers
4. **EpochChange (CloseBlock)**: Calculate rewards, process withdrawals, pick new active validators
5. **Slashing**: Transfer slashed validator's bond to governance vault

## When to Use .rhox

Use `.rhox` when:
- A contract needs different configuration per deployment (e.g., testnet vs mainnet)
- Genesis contracts need chain-specific parameters
- You want to avoid hardcoding values that vary between environments

## Processing

The `.rhox` file is processed by the node during genesis or system deploy setup. The Rust shard reads the template, substitutes parameters from the node configuration, and deploys the resulting Rholang code.

Regular user contracts should use standard `.rho` files. The `.rhox` format is primarily for system-level contracts.
