# SPEC-string-escape-handling

## String and URI Escape Sequence Handling in Rholang

### Problem Statement

Previously, when saving string literals to channels in Rholang, escape sequences were not being properly processed. For example, the string `"\""` would incorrectly become `"\\\""` instead of being properly unescaped to just `"`.

### Root Cause

The `stripString` and `stripUri` methods in `GroundNormalizeMatcher.scala` were only removing the outer delimiters (quotes for strings, backticks for URIs) without processing the escape sequences defined in the Rholang grammar.

### Grammar Definition

According to the Rholang grammar (`rholang_mercury.cf`):

```bnfc
token StringLiteral ( '"' ((char - ["\"\\"]) | ('\\' ["\"\\nt"]))* '"' );
token UriLiteral ('`' ((char - ["\\`"]) | ('\\' ["`\\"]))* '`') ;
```

#### String Escape Sequences
- `\"` - Escaped double quote
- `\\` - Escaped backslash
- `\n` - Newline character
- `\t` - Tab character

#### URI Escape Sequences
- `` \` `` - Escaped backtick
- `\\` - Escaped backslash

### Solution

Implemented proper unescape functions that process escape sequences after removing the outer delimiters:

1. **String Processing**: Added `unescapeString` method that converts:
   - `\"` → `"`
   - `\\` → `\`
   - `\n` → newline character
   - `\t` → tab character

2. **URI Processing**: Added `unescapeUri` method that converts:
   - `` \` `` → `` ` ``
   - `\\` → `\`

### Implementation Details

The unescape functions use a character-by-character scanning approach with a StringBuilder for efficiency:

1. Scan through the string character by character
2. When a backslash is encountered, check the next character
3. If it's a valid escape sequence, append the unescaped character and skip both characters
4. If it's not a valid escape sequence, treat the backslash as a literal character
5. Regular characters are appended as-is

### Test Coverage

Added comprehensive test cases in `GroundMatcherSpec.scala`:

- Single escaped quote: `"\""` → `"`
- Single escaped backslash: `"\\"` → `\`
- Escaped newline: `"Hello\nWorld"` → `Hello` + newline + `World`
- Escaped tab: `"Hello\tWorld"` → `Hello` + tab + `World`
- Multiple escape sequences in one string
- URI with escaped backtick
- URI with escaped backslash
- URI with multiple escape sequences

### Backwards Compatibility

This change maintains backwards compatibility:
- Strings without escape sequences are processed identically to before
- Invalid escape sequences (backslash followed by non-escape character) are preserved as-is
- The change only affects the interpretation of valid escape sequences as defined in the grammar

### Performance Considerations

The implementation uses StringBuilder for efficient string building and processes each character only once (O(n) complexity where n is the string length).

### Security Implications

Proper escape sequence handling is crucial for security:
- Prevents injection attacks through malformed strings
- Ensures data integrity when storing and retrieving string values
- Maintains consistency between what developers write and what gets executed

### Future Considerations

If additional escape sequences are added to the Rholang grammar in the future (e.g., `\r` for carriage return, `\b` for backspace, Unicode escapes), the unescape functions will need to be updated accordingly.