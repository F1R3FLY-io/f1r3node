# String and URI Escape Sequence Fix

## Summary
Fixed a critical bug in the Rholang interpreter where escape sequences in string and URI literals were not being properly processed during normalization.

## The Problem
When saving strings containing escape sequences to channels, the escape characters were being preserved literally instead of being unescaped. For example:
- Input: `"\""`
- Expected: `"`
- Actual (before fix): `\"`

This affected all escape sequences defined in the Rholang grammar including quotes, backslashes, newlines, and tabs.

## The Solution

### Modified Files
1. **`rholang/src/main/scala/coop/rchain/rholang/interpreter/compiler/normalizer/GroundNormalizeMatcher.scala`**
   - Updated `stripString` method to call new `unescapeString` function
   - Updated `stripUri` method to call new `unescapeUri` function
   - Added `unescapeString` method to handle: `\"`, `\\`, `\n`, `\t`
   - Added `unescapeUri` method to handle: `` \` ``, `\\`

2. **`rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/GroundMatcherSpec.scala`**
   - Added 5 new test cases for string escape sequences
   - Added 3 new test cases for URI escape sequences
   - Tests verify proper unescaping of all supported sequences

3. **`docs/specifications/SPEC-string-escape-handling.md`** (new)
   - Complete technical specification of the fix
   - Documents grammar rules, implementation details, and test coverage

4. **`test_string_escape.rho`** (new)
   - Rholang test file demonstrating proper escape sequence handling
   - Includes test cases for all supported escape sequences

## Impact
- **Security**: Prevents potential injection attacks through malformed strings
- **Correctness**: Ensures strings are stored and retrieved with proper escape sequence interpretation
- **Compatibility**: Maintains backwards compatibility - strings without escape sequences work identically

## Examples

### Before Fix
```rholang
new ch in {
  ch!("\"Hello\"") |  // Would store literally: \"Hello\"
  ch!("Line1\nLine2") | // Would store literally: Line1\nLine2
  for (@msg <- ch) { ... }
}
```

### After Fix
```rholang
new ch in {
  ch!("\"Hello\"") |  // Now correctly stores: "Hello"
  ch!("Line1\nLine2") | // Now correctly stores: Line1<newline>Line2
  for (@msg <- ch) { ... }
}
```

## Testing
Run the test suite to verify the fix:
```bash
# In a Nix development shell
nix develop
sbt "project rholang" "testOnly *GroundMatcherSpec"
```

## Related Issues
- Fixes the issue where saving `"\""` string to a channel changes its value to `"\\\""`
- Ensures compliance with the Rholang grammar specification for string and URI literals