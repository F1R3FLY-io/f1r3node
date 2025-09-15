# RGB CLI Bug Report: State/Pay Commands Failing with clap Validation Error

## **Summary**

RGB CLI commands `state`, `pay`, `seals`, and `sync` are completely broken due to a clap argument group validation failure. This prevents users from checking token balances, making payments, or synchronizing with the blockchain.

## **Error Message**

```
Error: Command state: Argument group 'ResolverOpt' contains non-existent argument 'mempool'
Happened in clap_builder-4.5.41/src/builder/debug_asserts.rs:298:13
```

## **Root Cause**

**File:** `cli/src/opts.rs:33`  
**Issue:** Clap argument group references a non-existent struct field

```rust
// Line 33: References "mempool" but field doesn't exist
#[group(args = ["electrum", "esplora", "mempool"])]  // ❌ "mempool" doesn't exist
pub struct ResolverOpt {
    pub electrum: Option<String>,  // ✅ exists
    pub esplora: Option<String>,   // ✅ exists  
    // ❌ pub mempool: Option<String> was REMOVED but group still references it
}
```

## **Bug History**

### **Commit 1: Correct Implementation** ✅
**Commit:** `0898105e` - "cli: add options for resolver" (July 7, 2025)  
**Author:** Dr Maxim Orlovsky

- ✅ **CORRECTLY** added: `pub mempool: Option<String>` field
- ✅ **CORRECTLY** added: `#[group(args = ["electrum", "esplora", "mempool"])]`
- ✅ **CORRECTLY** added: `DEFAULT_MEMPOOL` constant
- ✅ **CORRECTLY** updated: `ResolverOpt { electrum: None, esplora: None, mempool: None }`

### **Commit 2: Introduced Bug** ❌ 
**Commit:** `5a91402d` - "resolvers: complete constructors" (July 8, 2025)  
**Author:** Dr Maxim Orlovsky

- ❌ **INCORRECTLY** removed: `pub mempool: Option<String>` field
- ❌ **INCORRECTLY** removed: `DEFAULT_MEMPOOL` constant  
- ❌ **INCORRECTLY** left: `#[group(args = ["electrum", "esplora", "mempool"])]` 
- ✅ **CORRECTLY** updated: `DEFAULT_ESPLORA` to point to mempool.space
- ✅ **CORRECTLY** updated: `ResolverOpt { electrum: None, esplora: None }`

**Analysis:** The developer consolidated mempool functionality into esplora (both now use mempool.space) but forgot to update the clap group declaration.

## **Impact**

### **Broken Commands:**
- `rgb state` - Cannot check token balances ❌
- `rgb pay` - Cannot make payments ❌  
- `rgb seals` - Cannot list wallet seals ❌
- `rgb sync` - Cannot sync with blockchain ❌

### **Working Commands:**
- `rgb contracts` - List contracts ✅
- `rgb issue` - Issue tokens ✅
- `rgb import` - Import issuers ✅  
- `rgb backup` - Export contracts ✅
- `rgb invoice` - Generate invoices ✅

## **Fix**

**Simple One-Line Fix:**

**File:** `cli/src/opts.rs`  
**Line 33:** Change from:
```rust
#[group(args = ["electrum", "esplora", "mempool"])]
```
**To:**
```rust
#[group(args = ["electrum", "esplora"])]
```

## **Testing**

After applying the fix:

1. **Before Fix:**
   ```bash
   ./target/debug/rgb -d examples/data state -go -w alice
   # Error: Command state: Argument group 'ResolverOpt' contains non-existent argument 'mempool'
   ```

2. **After Fix:**
   ```bash  
   ./target/debug/rgb -d examples/data state -go -w alice  
   # Should work correctly and display contract state
   ```

## **Verification Steps**

1. Apply the one-line fix to `cli/src/opts.rs:33`
2. Rebuild: `cargo build --bin rgb`
3. Test state command: `./target/debug/rgb -d examples/data state -go -w alice`
4. Test payment workflow with existing demo scripts

## **Additional Notes**

- The mempool functionality appears to have been intentionally consolidated into esplora resolver
- Both `DEFAULT_ESPLORA` and the removed `DEFAULT_MEMPOOL` pointed to mempool.space
- The consolidation makes sense but the clap group wasn't updated accordingly
- This is purely a configuration issue, not a functional problem with the resolver logic

## **Repository Information**

- **Repository:** https://github.com/RGB-WG/rgb
- **Branch:** master  
- **Affected File:** `cli/src/opts.rs:33`
- **Bug Introduced:** July 8, 2025 (commit 5a91402d)
- **Severity:** High (breaks core wallet functionality)
- **Complexity:** Trivial fix (one line change)
