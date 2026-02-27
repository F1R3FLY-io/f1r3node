# Ignored Tests in Casper Test Suite

This document catalogs all ignored tests in the Casper test suite, explaining what each test does and why it may be ignored.

## Summary

| Test File | Test Name | Reason |
|-----------|-----------|--------|
| MultiParentCasperAddBlockSpec | reject blocks not from bonded validators | Unknown |
| MultiParentCasperAddBlockSpec | reject addBlock when duplicate deploy | Needs investigation (RHOL-1048) |
| MultiParentCasperAddBlockSpec | ignore adding equivocation blocks | Unknown |
| MultiParentCasperAddBlockSpec | not ignore equivocation blocks for parents | Unknown |
| MultiParentCasperMergeSpec | not merge blocks touching same channel with joins | Unknown |
| ListeningNameAPITest | work across a chain | Multi-parent merging semantics |
| InterpreterUtilTest | merge histories in case of multiple parents | REV balance merging not done |
| MultiParentCasperCommunicationSpec | ask peers for blocks and add them | REV balance merging not done |
| MultiParentCasperCommunicationSpec | handle long chain of block requests | Test doesn't make sense with hashes (RCHAIN-3819) |
| LastFinalizedAPITest | return true for ancestors of LFB | Unknown |
| LastFinalizedAPITest | return false for children/uncles/cousins of LFB | Multi-parent merging semantics |
| MultiParentCasperBondingSpec | allow bonding | Needs rewrite for new PoS |
| BondedStatusAPITest | return true for newly bonded validator | Multi-parent merging semantics |
| RholangBuildTest | execute the genesis block | Unknown (performance test?) |
| RevIssuanceTest | Rev issued based on Ethereum inputs | Unknown |
| GenesisTest | parse wallets file and create RevVaults | Unknown |

---

## Detailed Test Descriptions

### MultiParentCasperAddBlockSpec.scala

#### "reject blocks not from bonded validators" (line 227)
**What it tests:** Verifies that blocks signed by validators who are not in the current bond set are rejected with `InvalidSender` status.

**Why ignored:** No explicit reason given in code.

---

#### "reject addBlock when there exist deploy by the same (user, millisecond timestamp) in the chain" (line 272)
**What it tests:** Ensures that duplicate deploys (same user + same millisecond timestamp) are rejected when a deploy with identical identifiers already exists in the chain.

**Why ignored:** Related to RHOL-1048. Comment says "ticket is closed but this test will not pass, investigate further."

---

#### "ignore adding equivocation blocks" (line 305)
**What it tests:** Verifies that when a validator creates two different blocks with the same sequence number (equivocation), only the first block is accepted and the equivocating block is ignored by other nodes.

**Why ignored:** No explicit reason given in code.

---

#### "not ignore equivocation blocks that are required for parents of proper nodes" (line 328)
**What it tests:** Tests the edge case where an equivocating block must still be accepted because it's a parent of a legitimate block from another validator. References `/docs/casper/images/minimal_equivocation_neglect.png`.

**Why ignored:** No explicit reason given in code.

---

### MultiParentCasperMergeSpec.scala

#### "not merge blocks that touch the same channel involving joins" (line 164)
**What it tests:** Verifies that blocks which access the same channel through join patterns are not merged together, resulting in a single-parent block instead of a multi-parent merge.

**Why ignored:** No explicit reason given in code.

---

### ListeningNameAPITest.scala

#### "work across a chain" (line 54)
**What it tests:** Tests that `getListeningNameDataResponse` API works correctly across multiple blocks in a chain, returning data from the appropriate blocks as the chain grows.

**Why ignored:** Multi-parent merging changes the "main chain" concept. With multi-parent blocks, all validators' blocks are merged, changing how data is queried across the chain.

---

### InterpreterUtilTest.scala

#### "merge histories in case of multiple parents" (line 166)
**What it tests:** Verifies that when a block has multiple parents (b1 and b2), the histories are correctly merged so that data from both parent branches is accessible. Tests a DAG where b3 merges b1 and b2.

**Why ignored:** "TODO reenable when merging of REV balances is done"

---

### MultiParentCasperCommunicationSpec.scala

#### "ask peers for blocks it is missing and add them" (line 77)
**What it tests:** Complex test verifying that when a node receives a block with missing dependencies, it correctly requests and receives all missing ancestor blocks from peers. Tests a scenario where node2 misses several rounds of blocks and must catch up.

**Why ignored:** Two reasons:
1. "TODO reenable when merging of REV balances is done"
2. "TODO: investigate why this test is so brittle - in presence of hashes it starts to pass only when hashes are synchronized precisely"

---

#### "handle a long chain of block requests appropriately" (line 127)
**What it tests:** Tests handling of a long chain (10+ blocks) of missing dependencies when syncing between nodes, including simulated network failures.

**Why ignored:** "TODO: investigate this test - it doesn't make much sense in the presence of hashes (see RCHAIN-3819) and why on earth does it test logs?"

---

### LastFinalizedAPITest.scala

#### "return true for ancestors of last finalized block" (line 71)
**What it tests:** Verifies that `isFinalized` API returns `true` for blocks that are ancestors of the last finalized block (LFB), including the LFB itself and its parents.

**Why ignored:** No explicit reason given in code.

---

#### "return false for children, uncles and cousins of last finalized block" (line 118)
**What it tests:** Verifies that `isFinalized` API returns `false` for blocks that are not yet finalized: children of LFB, uncle blocks (siblings of ancestors), and cousin blocks.

**Why ignored:** "TODO: Review finalization semantics with multi-parent merging" - the DAG structure and finalization timing changes with multi-parent merging.

---

### MultiParentCasperBondingSpec.scala

#### "allow bonding" (line 8)
**What it tests:** Originally tested the bonding mechanism allowing new validators to join.

**Why ignored:** "TODO rewrite this test for the new PoS" - the Proof of Stake mechanism has changed.

---

### BondedStatusAPITest.scala

#### "return true for newly bonded validator" (line 65)
**What it tests:** Verifies that after a validator bonds and the bonding block is finalized, the `bondStatus` API returns `true` for that validator.

**Why ignored:** "TODO: Review finalization semantics with multi-parent merging" - finalization timing changes with multi-parent merging, affecting when bonding takes effect.

---

### RholangBuildTest.scala

#### "execute the genesis block" (line 65)
**What it tests:** Performance/stress test that creates a genesis block with 16,000 REV vaults and verifies no warnings are produced during execution.

**Why ignored:** No explicit reason given. May be ignored due to long execution time or resource requirements.

---

### RevIssuanceTest.scala

#### "Rev issued and accessible based on inputs from Ethereum" (line 7)
**What it tests:** Tests the REV issuance mechanism based on Ethereum bridge inputs.

**Why ignored:** No explicit reason given in code.

---

### GenesisTest.scala

#### "parse the wallets file and create corresponding RevVault-s" (line 232)
**What it tests:** Verifies that the genesis process correctly parses a wallets file and creates RevVault instances for each entry.

**Why ignored:** No explicit reason given in code. Test body is empty (`{}`).

---

## Categories of Ignored Tests

### Multi-Parent Merging Related (4 tests)
Tests that need review due to changes in DAG structure and finalization semantics with multi-parent merging:
- ListeningNameAPITest: "work across a chain"
- LastFinalizedAPITest: "return false for children, uncles and cousins of last finalized block"
- BondedStatusAPITest: "return true for newly bonded validator"

### REV Balance Merging (2 tests)
Tests waiting for REV balance merging implementation:
- InterpreterUtilTest: "merge histories in case of multiple parents"
- MultiParentCasperCommunicationSpec: "ask peers for blocks it is missing and add them"

### Needs Investigation (4 tests)
Tests with unclear reasons or needing further investigation:
- MultiParentCasperAddBlockSpec: "reject blocks not from bonded validators"
- MultiParentCasperAddBlockSpec: "ignore adding equivocation blocks"
- MultiParentCasperAddBlockSpec: "not ignore equivocation blocks that are required for parents"
- MultiParentCasperMergeSpec: "not merge blocks that touch the same channel involving joins"

### Obsolete/Needs Rewrite (3 tests)
Tests that need significant changes due to system changes:
- MultiParentCasperAddBlockSpec: "reject addBlock when duplicate deploy" (RHOL-1048)
- MultiParentCasperCommunicationSpec: "handle long chain of block requests" (RCHAIN-3819)
- MultiParentCasperBondingSpec: "allow bonding" (new PoS)

### Unknown/Empty (3 tests)
Tests with no clear reason for being ignored:
- LastFinalizedAPITest: "return true for ancestors of last finalized block"
- RholangBuildTest: "execute the genesis block"
- RevIssuanceTest: "Rev issued and accessible based on Ethereum inputs"
- GenesisTest: "parse the wallets file and create corresponding RevVault-s"

---

*Last updated: December 2025*
