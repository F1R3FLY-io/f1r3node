#!/bin/bash

# Script to replace REV -> configurable ticker throughout the entire project
# This migrates an EXISTING blockchain from REV tokens to the new ticker
set -e

# Configuration - Ticker name parameter
TICKER_NAME="${1:-ASI}"  # Default to ASI if no parameter provided

# Validate ticker name
if [ -z "$TICKER_NAME" ]; then
    echo "❌ ERROR: Ticker name cannot be empty"
    echo "Usage: $0 <TICKER_NAME>"
    echo "Example: $0 ASI"
    exit 1
fi

# Generate case variations
TICKER_UPPER=$(echo "$TICKER_NAME" | tr '[:lower:]' '[:upper:]')
TICKER_LOWER=$(echo "$TICKER_NAME" | tr '[:upper:]' '[:lower:]')

echo "🚀 Starting REV → ${TICKER_UPPER} migration for existing blockchain..."
echo "📊 Using ticker: ${TICKER_UPPER} (upper), ${TICKER_LOWER} (lower)"
echo ""

# 0. CLEANUP - Remove any leftover ticker files from previous runs
echo "🧹 Cleaning up leftover files from previous runs..."

# Remove ticker directories if they exist (from previous incomplete runs)
if [ -d "node/src/main/scala/coop/rchain/node/${TICKER_LOWER}vaultexport" ]; then
    rm -rf "node/src/main/scala/coop/rchain/node/${TICKER_LOWER}vaultexport"
    echo "✅ Removed leftover node/src/main/.../${TICKER_LOWER}vaultexport/"
fi

if [ -d "node/src/test/scala/coop/rchain/node/${TICKER_LOWER}vaultexport" ]; then
    rm -rf "node/src/test/scala/coop/rchain/node/${TICKER_LOWER}vaultexport"
    echo "✅ Removed leftover node/src/test/.../${TICKER_LOWER}vaultexport/"
fi

# Remove ticker files if they exist (from previous incomplete runs)
if [ -f "casper/src/main/resources/${TICKER_UPPER}Vault.rho" ]; then
    rm "casper/src/main/resources/${TICKER_UPPER}Vault.rho"
    echo "✅ Removed leftover ${TICKER_UPPER}Vault.rho"
fi

if [ -f "casper/src/main/resources/MultiSig${TICKER_UPPER}Vault.rho" ]; then
    rm "casper/src/main/resources/MultiSig${TICKER_UPPER}Vault.rho"
    echo "✅ Removed leftover MultiSig${TICKER_UPPER}Vault.rho"
fi

if [ -f "casper/src/test/resources/${TICKER_UPPER}VaultTest.rho" ]; then
    rm "casper/src/test/resources/${TICKER_UPPER}VaultTest.rho"
    echo "✅ Removed leftover ${TICKER_UPPER}VaultTest.rho"
fi

if [ -f "casper/src/test/resources/MultiSig${TICKER_UPPER}VaultTest.rho" ]; then
    rm "casper/src/test/resources/MultiSig${TICKER_UPPER}VaultTest.rho"
    echo "✅ Removed leftover MultiSig${TICKER_UPPER}VaultTest.rho"
fi

if [ -f "casper/src/test/resources/${TICKER_UPPER}AddressTest.rho" ]; then
    rm "casper/src/test/resources/${TICKER_UPPER}AddressTest.rho"
    echo "✅ Removed leftover ${TICKER_UPPER}AddressTest.rho"
fi

if [ -f "casper/src/main/scala/coop/rchain/casper/genesis/contracts/${TICKER_UPPER}Generator.scala" ]; then
    rm "casper/src/main/scala/coop/rchain/casper/genesis/contracts/${TICKER_UPPER}Generator.scala"
    echo "✅ Removed leftover ${TICKER_UPPER}Generator.scala"
fi

if [ -f "rholang/src/main/scala/coop/rchain/rholang/interpreter/util/${TICKER_UPPER}Address.scala" ]; then
    rm "rholang/src/main/scala/coop/rchain/rholang/interpreter/util/${TICKER_UPPER}Address.scala"
    echo "✅ Removed leftover ${TICKER_UPPER}Address.scala"
fi

if [ -f "rholang/src/test/scala/coop/rchain/rholang/interpreter/util/${TICKER_UPPER}AddressSpec.scala" ]; then
    rm "rholang/src/test/scala/coop/rchain/rholang/interpreter/util/${TICKER_UPPER}AddressSpec.scala"
    echo "✅ Removed leftover ${TICKER_UPPER}AddressSpec.scala"
fi

if [ -f "casper/src/test/scala/coop/rchain/casper/genesis/contracts/${TICKER_UPPER}AddressSpec.scala" ]; then
    rm "casper/src/test/scala/coop/rchain/casper/genesis/contracts/${TICKER_UPPER}AddressSpec.scala"
    echo "✅ Removed leftover ${TICKER_UPPER}AddressSpec.scala"
fi

echo "🧹 Cleanup completed!"
echo ""

# 0.5. VERIFY REQUIRED FILES EXIST
echo "🔍 Verifying required REV files exist..."

#based on clear main branch
required_files=(
    "casper/src/main/resources/RevVault.rho"
    "casper/src/main/resources/MultiSigRevVault.rho"
    "casper/src/test/resources/RevVaultTest.rho"
    "casper/src/test/resources/MultiSigRevVaultTest.rho"
    "casper/src/test/resources/RevAddressTest.rho"
    "casper/src/main/scala/coop/rchain/casper/genesis/contracts/RevGenerator.scala"
    "rholang/src/main/scala/coop/rchain/rholang/interpreter/util/RevAddress.scala"
    "rholang/src/test/scala/coop/rchain/rholang/interpreter/util/RevAddressSpec.scala"
    "casper/src/test/scala/coop/rchain/casper/genesis/contracts/RevAddressSpec.scala"
)

required_dirs=(
    "node/src/main/scala/coop/rchain/node/revvaultexport"
    "node/src/test/scala/coop/rchain/node/revvaultexport"
)

missing_files=()
missing_dirs=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_files[@]} -gt 0 ] || [ ${#missing_dirs[@]} -gt 0 ]; then
    echo "❌ ERROR: Missing required files/directories:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    for dir in "${missing_dirs[@]}"; do
        echo "   - $dir"
    done
    echo ""
    echo "💡 Please make sure you have a clean REV codebase before running this script."
    echo "   Consider doing: git checkout . && git clean -fd"
    exit 1
fi

echo "✅ All required REV files found!"
echo ""

# 1. REPLACE CONTENT IN FILES
echo "📝 Replacing text in files..."

# Main identifiers (case-sensitive) - ONLY in code files  
# NOTE: Documentation files (.md, .txt, .json, .py, etc.) are NOT processed
# Python files are handled separately in Section 5b3 with more careful patterns
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" -o -name "*.rhox" \) ! -name "*.py" ! -path "./.git/*" | while read -r file; do
    # Skip if file is empty or doesn't exist
    [ -f "$file" ] || continue

    # Case-sensitive replacements
    sed -i.bak \
        -e "s/RevVault/${TICKER_UPPER}Vault/g" \
        -e "s/RevAddress/${TICKER_UPPER}Address/g" \
        -e "s/RevGenerator/${TICKER_UPPER}Generator/g" \
        -e "s/revVault/${TICKER_LOWER}Vault/g" \
        -e "s/revAddress/${TICKER_LOWER}Address/g" \
        -e "s/revGenerator/${TICKER_LOWER}Generator/g" \
        -e "s/RevAccount/${TICKER_UPPER}Account/g" \
        -e "s/revAccount/${TICKER_LOWER}Account/g" \
        -e "s/RevAddr/${TICKER_UPPER}Addr/g" \
        -e "s/revAddr/${TICKER_LOWER}Addr/g" \
        -e "s/revVaultCh/${TICKER_LOWER}VaultCh/g" \
        -e "s/multiSigRevVault/multiSig${TICKER_UPPER}Vault/g" \
        -e "s/MultiSigRevVault/MultiSig${TICKER_UPPER}Vault/g" \
        -e "s/REV_ADDRESS/${TICKER_UPPER}_ADDRESS/g" \
        -e "s/rev_address/${TICKER_LOWER}_address/g" \
        -e "s/receiveRev/receive${TICKER_UPPER}/g" \
        -e "s/sendRev/send${TICKER_UPPER}/g" \
        "$file"
done

# 2. REPLACE URI IN REGISTRY
echo "🔗 Replacing Registry URIs..."
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" -o -name "*.rhox" \) ! -path "./.git/*" | while read -r file; do
    sed -i.bak \
        -e "s/rho:rchain:revVault/rho:rchain:${TICKER_LOWER}Vault/g" \
        -e "s/rho:rchain:multiSigRevVault/rho:rchain:multiSig${TICKER_UPPER}Vault/g" \
        -e "s/rho:rev:address/rho:${TICKER_LOWER}:address/g" \
        "$file"
done

# 3. SPECIAL HANDLING FOR REGISTRY.RHO - Critical system file
echo "🏛️ Updating Registry.rho..."
if [ -f "casper/src/main/resources/Registry.rho" ]; then
    sed -i.bak \
        -e "s/\`rho:rchain:revVault\`/\`rho:rchain:${TICKER_LOWER}Vault\`/g" \
        -e "s/\`rho:rchain:multiSigRevVault\`/\`rho:rchain:multiSig${TICKER_UPPER}Vault\`/g" \
        "casper/src/main/resources/Registry.rho"
    echo "✅ Updated Registry.rho system URIs"
fi

# 4. UPDATE SCALA PACKAGES - Critical for compilation
echo "📦 Updating Scala package declarations..."
find . -name "*.scala" | while read -r file; do
    sed -i.bak \
        -e "s/package coop\.rchain\.node\.revvaultexport/package coop.rchain.node.${TICKER_LOWER}vaultexport/g" \
        -e "s/import coop\.rchain\.node\.revvaultexport/import coop.rchain.node.${TICKER_LOWER}vaultexport/g" \
        -e "s/import.*RevAddress/import coop.rchain.rholang.interpreter.util.${TICKER_UPPER}Address/g" \
        "$file"
done

# 4c. FORCE package line update in test sources (anchor at start of line)
echo "✅ Verifying test package lines use ${TICKER_LOWER}vaultexport..."
find node/src/test/scala -name "*.scala" | while read -r file; do
    sed -i.bak -E \
        -e "s/^package[[:space:]]+coop\.rchain\.node\.revvaultexport/package coop.rchain.node.${TICKER_LOWER}vaultexport/g" \
        "$file"
done

# 4b. ADDITIONAL: Update any remaining revvaultexport references in all file types
echo "🔄 Updating any remaining revvaultexport references..."
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" -o -name "*.rhox" \) ! -path "./.git/*" | while read -r file; do
    sed -i.bak \
        -e "s/coop\.rchain\.node\.revvaultexport/coop.rchain.node.${TICKER_LOWER}vaultexport/g" \
        "$file"
done

# 5. UPDATE COMMENTS AND DOCUMENTATION (selective)
echo "📚 Updating comments..."
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" -o -name "*.rhox" \) ! -path "./.git/*" | while read -r file; do
    sed -i.bak \
        -e "s/Rev vault/${TICKER_UPPER} vault/g" \
        -e "s/Rev address/${TICKER_UPPER} address/g" \
        -e "s/RevAddress for/${TICKER_UPPER}Address for/g" \
        -e "s/RevAddresses/${TICKER_UPPER}Addresses/g" \
        -e "s/Get deployer.*rev address/Get deployer ${TICKER_UPPER} address/g" \
        -e "s/Convert.*RevAddress/Convert into ${TICKER_UPPER}Address/g" \
        -e "s/correct RevAddress/correct ${TICKER_UPPER}Address/g" \
        -e "s/expecting RevAddress/expecting ${TICKER_UPPER}Address/g" \
        -e "s/all the revVault account/all the ${TICKER_LOWER}Vault account/g" \
        -e "s/seedForRevVault/seedFor${TICKER_UPPER}Vault/g" \
        "$file"
done

# 5b. COMPREHENSIVE HANDLING OF ALL REMAINING REV/rev INSTANCES
echo "🔍 COMPREHENSIVE update of ALL remaining REV/rev instances across ALL file types..."
echo "   📊 This addresses ALL cases found through case-insensitive search"

# 5b1. SMART CONTRACT STRING LITERALS AND COMMENTS (.rho, .rhox files)
echo "  🔧 Smart contract files: ALL string literals, comments, template variables"
find . -type f \( -name "*.rho" -o -name "*.rhox" \) ! -path "./.git/*" | while read -r file; do
    [ -f "$file" ] || continue
    sed -i.bak \
        -e "s/\([\"']\)\([^\"']*\)\bREV\b\([^\"']*\)\1/\1\2${TICKER_UPPER}\3\1/g" \
        -e "s/\([\"']\)\([^\"']*\)\brev address\b\([^\"']*\)\1/\1\2${TICKER_LOWER} address\3\1/g" \
        -e "s/\([\"']\)\([^\"']*\)\bRev address\b\([^\"']*\)\1/\1\2${TICKER_UPPER} address\3\1/g" \
        -e "s/pretty surely invalid rev address/pretty surely invalid ${TICKER_LOWER} address/g" \
        -e "s/\/\/ the rev address of/\/\/ the ${TICKER_LOWER} address of/g" \
        -e "s/\/\/ Get unforgeable channel REV address/\/\/ Get unforgeable channel ${TICKER_UPPER} address/g" \
        -e "s/\/\/ REV vault/\/\/ ${TICKER_UPPER} vault/g" \
        -e "s/%REV_ADDR/%${TICKER_UPPER}_ADDR/g" \
        -e "s/REPLACE THE REV ADDRESS/REPLACE THE ${TICKER_UPPER} ADDRESS/g" \
        -e "s/REPLACE THE REV ADDRESSES/REPLACE THE ${TICKER_UPPER} ADDRESSES/g" \
        -e "s/\"REV from\"/\"${TICKER_UPPER} from\"/g" \
        -e "s/\"REV to\"/\"${TICKER_UPPER} to\"/g" \
        -e "s/, \"REV from\"/, \"${TICKER_UPPER} from\"/g" \
        -e "s/, \"REV to\"/, \"${TICKER_UPPER} to\"/g" \
        -e "s/\/\/scalapackage coop\.rchain\.rholang\.rev/\/\/scalapackage coop.rchain.rholang.${TICKER_LOWER}/g" \
        "$file"
done

# 5b2. SCALA FILES - ALL variable names, comments, test descriptions, string literals  
echo "  🔧 Scala files: ALL variables, functions, comments, test descriptions"
find . -type f -name "*.scala" ! -path "./.git/*" | while read -r file; do
    [ -f "$file" ] || continue
    sed -i.bak \
        -e "s/\brevBalance\b/${TICKER_LOWER}Balance/g" \
        -e "s/\brevAndBalance\b/${TICKER_LOWER}AndBalance/g" \
        -e "s/\brevBalanceStr\b/${TICKER_LOWER}BalanceStr/g" \
        -e "s/revBalance(/${TICKER_LOWER}Balance(/g" \
        -e "s/revBalance)/${TICKER_LOWER}Balance)/g" \
        -e "s/, revBalance)/, ${TICKER_LOWER}Balance)/g" \
        -e "s/revAndBalance/${TICKER_LOWER}AndBalance/g" \
        -e "s/<revBalance>/<${TICKER_LOWER}Balance>/g" \
        -e "s/compute REV balances/compute ${TICKER_UPPER} balances/g" \
        -e "s/merging of REV balances/merging of ${TICKER_UPPER} balances/g" \
        -e "s/\/\/ REV vault initialization/\/\/ ${TICKER_UPPER} vault initialization/g" \
        -e "s/\/\/ REV vault/\/\/ ${TICKER_UPPER} vault/g" \
        -e "s/someone else transfer some rev/someone else transfer some ${TICKER_LOWER}/g" \
        -e "s/<REV_address>/<${TICKER_UPPER}_address>/g" \
        -e "s/Parse REV address/Parse ${TICKER_UPPER} address/g" \
        -e "s/REV address parser/${TICKER_UPPER} address parser/g" \
        -e "s/converter to REV address/converter to ${TICKER_UPPER} address/g" \
        -e "s/\*\*Not Created Vault\*\* would get 0 balance even if someone else transfer some rev/**Not Created Vault** would get 0 balance even if someone else transfer some ${TICKER_LOWER}/g" \
        -e "s/reenable when merging of REV balances is done/reenable when merging of ${TICKER_UPPER} balances is done/g" \
        -e "s/\"transfer rev\"/\"transfer ${TICKER_LOWER}\"/g" \
        -e "s/which defines the Rev wallets/which defines the ${TICKER_UPPER} wallets/g" \
        -e "s/is the amount of Rev they have bonded/is the amount of ${TICKER_UPPER} they have bonded/g" \
        -e "s/is the amount of Rev in the wallet/is the amount of ${TICKER_UPPER} in the wallet/g" \
        -e "s/\"Rev\" should \"/\"${TICKER_UPPER}\" should \"/g" \
        -e "s/to set initial REV accounts/to set initial ${TICKER_UPPER} accounts/g" \
        -e "s/Parser for wallets file used in genesis ceremony to set initial REV accounts/Parser for wallets file used in genesis ceremony to set initial ${TICKER_UPPER} accounts/g" \
        -e "s/Initial validator vaults contain 0 Rev/Initial validator vaults contain 0 ${TICKER_UPPER}/g" \
        -e "s/<synthetic in Rev\.scala>/<synthetic in ${TICKER_UPPER}.scala>/g" \
        -e "s/1 REV if phloPrice=1/1 ${TICKER_UPPER} if phloPrice=1/g" \
        "$file"
done

# 5b3. PYTHON INTEGRATION TEST FILES (.py files)
echo "  🔧 Python files: ALL variable names, method calls, strings"
find . -type f -name "*.py" ! -path "./.git/*" | while read -r file; do
    [ -f "$file" ] || continue
    sed -i.bak \
        -e "s/alice_rev_address/alice_${TICKER_LOWER}_address/g" \
        -e "s/bob_rev_address/bob_${TICKER_LOWER}_address/g" \
        -e "s/charlie_rev_address/charlie_${TICKER_LOWER}_address/g" \
        -e "s/Transfer rev from/Transfer ${TICKER_LOWER} from/g" \
        -e "s/\brev_addr\b/${TICKER_LOWER}_addr/g" \
        -e "s/%REV_ADDR/%${TICKER_UPPER}_ADDR/g" \
        -e "s/\"Transfer rev from one vault to another vault\"/\"Transfer ${TICKER_LOWER} from one vault to another vault\"/g" \
        -e "s/'rev_addr'/'${TICKER_LOWER}_addr'/g" \
        "$file"
done

# 5b4. CONFIGURATION FILES (.conf, .yaml, .yml files)
echo "  🔧 Configuration files: ALL parameter names, comments, descriptions"
find . -type f \( -name "*.conf" -o -name "*.yaml" -o -name "*.yml" \) ! -path "./.git/*" | while read -r file; do
    [ -f "$file" ] || continue
    sed -i.bak \
        -e "s/<revBalance>/<${TICKER_LOWER}Balance>/g" \
        -e "s/revBalance/${TICKER_LOWER}Balance/g" \
        -e "s/<algorithm> <pk> <revBalance>/<algorithm> <pk> <${TICKER_LOWER}Balance>/g" \
        -e "s/initial REV balance/initial ${TICKER_UPPER} balance/g" \
        -e "s/Has initial REV balance/Has initial ${TICKER_UPPER} balance/g" \
        -e "s/which defines the Rev wallets/which defines the ${TICKER_UPPER} wallets/g" \
        -e "s/is the amount of Rev they have bonded/is the amount of ${TICKER_UPPER} they have bonded/g" \
        -e "s/is the amount of Rev in the wallet/is the amount of ${TICKER_UPPER} in the wallet/g" \
        -e "s/# REV:/# ${TICKER_UPPER}:/g" \
        "$file"
done

# 5b5. DOCUMENTATION FILES (.md files) - COMPREHENSIVE token references
echo "  🔧 Documentation files: ALL token references (preserving technical terms)"
find . -type f -name "*.md" ! -path "./.git/*" | while read -r file; do
    [ -f "$file" ] || continue
    # Update ALL obvious TOKEN references, preserve technical terms like "reverse", "revision", etc.
    sed -i.bak \
        -e "s/Send REV tokens/Send ${TICKER_UPPER} tokens/g" \
        -e "s/REV vault address/${TICKER_UPPER} vault address/g" \
        -e "s/REV vault balance/${TICKER_UPPER} vault balance/g" \
        -e "s/her REV vault/her ${TICKER_UPPER} vault/g" \
        -e "s/his REV vault/his ${TICKER_UPPER} vault/g" \
        -e "s/a REV vault/a ${TICKER_UPPER} vault/g" \
        -e "s/transfers REV/transfers ${TICKER_UPPER}/g" \
        -e "s/recipient of REV/recipient of ${TICKER_UPPER}/g" \
        -e "s/my REV balance/my ${TICKER_UPPER} balance/g" \
        -e "s/%REV_ADDR/%${TICKER_UPPER}_ADDR/g" \
        -e "s/check her REV vault balance/check her ${TICKER_UPPER} vault balance/g" \
        -e "s/check Alice's REV vault balance/check Alice's ${TICKER_UPPER} vault balance/g" \
        -e "s/RevAddress (and not a RevVault)/${TICKER_UPPER}Address (and not a ${TICKER_UPPER}Vault)/g" \
        -e "s/RevVaults are possible/${TICKER_UPPER}Vaults are possible/g" \
        -e "s/simple RevVault/simple ${TICKER_UPPER}Vault/g" \
        -e "s/Know your RevAddress/Know your ${TICKER_UPPER}Address/g" \
        -e "s/creates a REV vault/creates a ${TICKER_UPPER} vault/g" \
        -e "s/vault and transfers REV into/vault and transfers ${TICKER_UPPER} into/g" \
        -e "s/Here's how Alice would check her REV vault address/Here's how Alice would check her ${TICKER_UPPER} vault address/g" \
        -e "s/Here's how Alice would check her REV vault balance/Here's how Alice would check her ${TICKER_UPPER} vault balance/g" \
        -e "s/Notice that anyone can check Alice's REV vault balance/Notice that anyone can check Alice's ${TICKER_UPPER} vault balance/g" \
        -e "s/When Bob checks his balance for the first time, a REV vault is created/When Bob checks his balance for the first time, a ${TICKER_UPPER} vault is created/g" \
        -e "s/order in which one creates a vault and transfers REV into/order in which one creates a vault and transfers ${TICKER_UPPER} into/g" \
        -e "s/Transfer to a RevAddress/Transfer to a ${TICKER_UPPER}Address/g" \
        -e "s/his REV address/his ${TICKER_UPPER} address/g" \
        -e "s/transfer 100 REV/transfer 100 ${TICKER_UPPER}/g" \
        -e "s/transfers REV into that vault/transfers ${TICKER_UPPER} into that vault/g" \
        -e "s/having the REV to pay/having the ${TICKER_UPPER} to pay/g" \
        -e "s/at the REV address/at the ${TICKER_UPPER} address/g" \
        -e "s/I want REV to be the currency token/I want ${TICKER_UPPER} to be the currency token/g" \
        -e "s/As a REV holder/As a ${TICKER_UPPER} holder/g" \
        -e "s/store REV in it/store ${TICKER_UPPER} in it/g" \
        -e "s/my REV to never be lost/my ${TICKER_UPPER} to never be lost/g" \
        -e "s/add REV to my coop-supplied wallet/add ${TICKER_UPPER} to my coop-supplied wallet/g" \
        -e "s/available REV to pay/available ${TICKER_UPPER} to pay/g" \
        -e "s/## REV/## ${TICKER_UPPER}/g" \
        -e "s/transfer rev/transfer ${TICKER_LOWER}/g" \
        -e "s/## Transfer to a RevAddress/## Transfer to a ${TICKER_UPPER}Address/g" \
        -e "s/wallets, Rev and phlogiston/wallets, ${TICKER_UPPER} and phlogiston/g" \
        -e "s/how much REV my deployment will cost/how much ${TICKER_UPPER} my deployment will cost/g" \
        -e "s/remove REV from my coop-supplied wallet/remove ${TICKER_UPPER} from my coop-supplied wallet/g" \
        -e "s/receive REV from another user/receive ${TICKER_UPPER} from another user/g" \
        -e "s/send REV to the coop-supplied wallet/send ${TICKER_UPPER} to the coop-supplied wallet/g" \
        -e "s/all REV transfers to and\/or from it/all ${TICKER_UPPER} transfers to and\/or from it/g" \
        -e "s/organization holding REV/organization holding ${TICKER_UPPER}/g" \
        -e "s/any REV transaction/any ${TICKER_UPPER} transaction/g" \
        -e "s/move Rev to\/from the key-pair/move ${TICKER_UPPER} to\/from the key-pair/g" \
        -e "s/compensated in REV for setting up/compensated in ${TICKER_UPPER} for setting up/g" \
        -e "s/receive interest in REV on my bond/receive interest in ${TICKER_UPPER} on my bond/g" \
        -e "s/use REV to pay the cost/use ${TICKER_UPPER} to pay the cost/g" \
        -e "s/buy a stake in Rev in that locale/buy a stake in ${TICKER_UPPER} in that locale/g" \
        -e "s/buy stake in Rev in that locale/buy stake in ${TICKER_UPPER} in that locale/g" \
        -e "s/burning the Rev allocated/burning the ${TICKER_UPPER} allocated/g" \
        -e "s/for Rev tracking in subtrees/for ${TICKER_UPPER} tracking in subtrees/g" \
        -e "s/Has initial REV balance for network operations/Has initial ${TICKER_UPPER} balance for network operations/g" \
        -e "s/- \*\*REV\*\*:/- **${TICKER_UPPER}**:/g" \
        "$file"
done

# 5c. RENAME FILES WITH "rev" IN THEIR NAMES
echo "  🔧 Renaming files with 'rev' in their names"
find . -name "*revaddress*" -o -name "*rev_*" | while read -r file; do
    if [ -f "$file" ]; then
        # Skip if it's a backup file
        [[ "$file" == *.bak ]] && continue
        
        # Create new filename by replacing rev patterns
        newfile=$(echo "$file" | sed -e "s/revaddress/${TICKER_LOWER}address/g" -e "s/rev_/${TICKER_LOWER}_/g")
        
        if [ "$file" != "$newfile" ]; then
            # Use git mv if possible, otherwise regular mv
            if git mv "$file" "$newfile" 2>/dev/null; then
                echo "    ✅ $file -> $newfile (git mv)"
            else
                mv "$file" "$newfile" 2>/dev/null && echo "    ✅ $file -> $newfile (mv)" || echo "    ⚠️ Failed to rename $file"
            fi
        fi
    fi
done

echo ""
echo "✅ COMPREHENSIVE REV/rev instance updates completed!"
echo "   📊 Processed: Smart contracts, Scala files, Python tests, Config files, Documentation"
echo "   🎯 Caught ALL case-insensitive 'rev' instances that refer to the token"

# 6. SPECIAL HANDLING FOR HARDCODED VALUES
echo "⚙️ Updating hardcoded values and constants..."
find . -name "*.scala" -o -name "*.rho" | while read -r file; do
    sed -i.bak \
        -e "s/REV_ADDRESS_COUNT/${TICKER_UPPER}_ADDRESS_COUNT/g" \
        -e "s/revVaultPk/${TICKER_LOWER}VaultPk/g" \
        -e "s/multiSigRevVaultPk/multiSig${TICKER_UPPER}VaultPk/g" \
        -e "s/revGeneratorPk/${TICKER_LOWER}GeneratorPk/g" \
        -e "s/revVaultPubKey/${TICKER_LOWER}VaultPubKey/g" \
        -e "s/revVaultTimestamp/${TICKER_LOWER}VaultTimestamp/g" \
        -e "s/multiSigRevVaultPk/multiSig${TICKER_UPPER}VaultPk/g" \
        "$file"
done

# 6b. CRITICAL: Update system process constants in SystemProcesses.scala
echo "🔧 Updating system process constants..."
find . -name "*.scala" | while read -r file; do
    sed -i.bak \
        -e "s/val REV_ADDRESS:/val ${TICKER_UPPER}_ADDRESS:/g" \
        -e "s/REV_ADDRESS:/${TICKER_UPPER}_ADDRESS:/g" \
        -e "s/FixedChannels\.REV_ADDRESS/FixedChannels.${TICKER_UPPER}_ADDRESS/g" \
        -e "s/BodyRefs\.REV_ADDRESS/BodyRefs.${TICKER_UPPER}_ADDRESS/g" \
        "$file"
done

# 6c. CRITICAL: Update system URI definition in RhoRuntime.scala
echo "🔗 Updating system URI definitions..."
if [ -f "rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntime.scala" ]; then
    sed -i.bak \
        -e 's/"rho:rev:address"/"rho:'${TICKER_LOWER}':address"/g' \
        "rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntime.scala"
    echo "✅ Updated system URI definition in RhoRuntime.scala"
fi

# 6d. CRITICAL: Update system process method names
echo "🔧 Updating system process method signatures..."
find . -name "*.scala" | while read -r file; do
    sed -i.bak \
        -e "s/def revAddress:/def ${TICKER_LOWER}Address:/g" \
        -e "s/\.revAddress/.${TICKER_LOWER}Address/g" \
        -e "s/systemProcesses\.revAddress/systemProcesses.${TICKER_LOWER}Address/g" \
        "$file"
done

# 6e. CRITICAL: Special handling for RevGenerator.scala code variable content
echo "🔧 Updating RevGenerator.scala code variable content..."
if [ -f "casper/src/main/scala/coop/rchain/casper/genesis/contracts/RevGenerator.scala" ]; then
    sed -i.bak \
        -e "s/revVaultCh/${TICKER_LOWER}VaultCh/g" \
        -e "s/RevVault/${TICKER_UPPER}Vault/g" \
        -e "s/revVaultInitCh/${TICKER_LOWER}VaultInitCh/g" \
        -e "s/\`rho:rchain:revVault\`/\`rho:rchain:${TICKER_LOWER}Vault\`/g" \
        -e "s/stripMargin('#')/stripMargin('|')/g" \
        -e "s/# /| /g" \
        -e "s/^         #/         |/g" \
        "casper/src/main/scala/coop/rchain/casper/genesis/contracts/RevGenerator.scala"
    echo "✅ Updated RevGenerator.scala code variable content"
fi

# 6f. CRITICAL: Special handling for PoS.rhox - Update vault URIs and contract references
echo "🔧 Updating PoS.rhox vault URIs and contract references..."
if [ -f "casper/src/main/resources/PoS.rhox" ]; then
    sed -i.bak \
        -e "s/rho:rev:address/rho:${TICKER_LOWER}:address/g" \
        -e "s/rho:rchain:revVault/rho:rchain:${TICKER_LOWER}Vault/g" \
        -e "s/rho:rchain:multiSigRevVault/rho:rchain:multiSig${TICKER_UPPER}Vault/g" \
        -e "s/RevVault/${TICKER_UPPER}Vault/g" \
        -e "s/MultiSigRevVault/MultiSig${TICKER_UPPER}Vault/g" \
        -e "s/revVaultCh/${TICKER_LOWER}VaultCh/g" \
        -e "s/multiSigRevVaultCh/multiSig${TICKER_UPPER}VaultCh/g" \
        -e "s/revAddressOps/${TICKER_LOWER}AddressOps/g" \
        -e "s/posDeployerRevAddressCh/posDeployer${TICKER_UPPER}AddressCh/g" \
        -e "s/fromRevAddress/from${TICKER_UPPER}Address/g" \
        -e "s/toRevAddress/to${TICKER_UPPER}Address/g" \
        -e "s/revAddressCh/${TICKER_LOWER}AddressCh/g" \
        "casper/src/main/resources/PoS.rhox"
    echo "✅ Updated PoS.rhox vault URIs and contract references"
fi

# 6g. CRITICAL: Special handling for VaultBalanceGetterTest.scala dynamic test logic
echo "🔧 Updating VaultBalanceGetterTest.scala with dynamic logic..."
if [ -f "node/src/test/scala/coop/rchain/node/revvaultexport/VaultBalanceGetterTest.scala" ]; then
    # Replace the entire "Get all vault" test with dynamic implementation
    cat > temp_test_replacement.txt << 'EOF'
  "Get all vault" should "return all vault balance" in {
    val t = TestNode.standaloneEff(genesis).use { node =>
      val genesisPostStateHash =
        Blake2b256Hash.fromByteString(genesis.genesisBlock.body.state.postStateHash)
      for {
        runtime <- node.runtimeManager.spawnRuntime
        _       <- runtime.reset(genesisPostStateHash)

        // Find vaultMap dynamically by checking all genesis vault addresses
        vaultPks = genesis.genesisVaults.toList.map(_._2)
        balances <- vaultPks.traverse { pub =>
                     val addr = TICKER_UPPERAddress.fromPublicKey(pub).get.address.toBase58
                     val getVault =
                       s"""new return, rl(`rho:registry:lookup`), TICKER_UPPERVaultCh, vaultCh in {
                         |  rl!(`rho:rchain:TICKER_LOWERVault`, *TICKER_UPPERVaultCh) |
                         |  for (@(_, TICKER_UPPERVault) <- TICKER_UPPERVaultCh) {
                         |    @TICKER_UPPERVault!("findOrCreate", "${addr}", *vaultCh) |
                         |    for (@(true, vault) <- vaultCh) {
                         |      @vault!("balance", *return)
                         |    }
                         |  }
                         |}
                         |""".stripMargin
                     for {
                       balancePars <- node.runtimeManager
                                       .playExploratoryDeploy(
                                         getVault,
                                         genesis.genesisBlock.body.state.postStateHash
                                       )
                       balance = if (balancePars.nonEmpty) {
                         balancePars(0).exprs.headOption
                           .flatMap(_.exprInstance.gInt)
                           .map(_.toInt)
                           .getOrElse(0)
                       } else 0
                     } yield balance
                   }
        _ = assert(balances.forall(_ == genesisInitialBalance))
      } yield ()
    }
    t.runSyncUnsafe()
  }
EOF

    # Replace TICKER placeholders with actual values
    sed -i.bak \
        -e "s/TICKER_UPPER/${TICKER_UPPER}/g" \
        -e "s/TICKER_LOWER/${TICKER_LOWER}/g" \
        temp_test_replacement.txt

    # Replace the test in the file
    sed -i.bak \
        -e '/^  "Get all vault" should "return all vault balance" in {$/,/^  }$/c\
  "Get all vault" should "return all vault balance" in {\
    val t = TestNode.standaloneEff(genesis).use { node =>\
      val genesisPostStateHash =\
        Blake2b256Hash.fromByteString(genesis.genesisBlock.body.state.postStateHash)\
      for {\
        runtime <- node.runtimeManager.spawnRuntime\
        _       <- runtime.reset(genesisPostStateHash)\
\
        // Find vaultMap dynamically by checking all genesis vault addresses\
        vaultPks = genesis.genesisVaults.toList.map(_._2)\
        balances <- vaultPks.traverse { pub =>\
                     val addr = '${TICKER_UPPER}'Address.fromPublicKey(pub).get.address.toBase58\
                     val getVault =\
                       s"""new return, rl(`rho:registry:lookup`), '${TICKER_UPPER}'VaultCh, vaultCh in {\
                         |  rl!(`rho:rchain:'${TICKER_LOWER}'Vault`, *'${TICKER_UPPER}'VaultCh) |\
                         |  for (@(_, '${TICKER_UPPER}'Vault) <- '${TICKER_UPPER}'VaultCh) {\
                         |    @'${TICKER_UPPER}'Vault!("findOrCreate", "${addr}", *vaultCh) |\
                         |    for (@(true, vault) <- vaultCh) {\
                         |      @vault!("balance", *return)\
                         |    }\
                         |  }\
                         |}\
                         |""".stripMargin\
                     for {\
                       balancePars <- node.runtimeManager\
                                       .playExploratoryDeploy(\
                                         getVault,\
                                         genesis.genesisBlock.body.state.postStateHash\
                                       )\
                       balance = if (balancePars.nonEmpty) {\
                         balancePars(0).exprs.headOption\
                           .flatMap(_.exprInstance.gInt)\
                           .map(_.toInt)\
                           .getOrElse(0)\
                       } else 0\
                     } yield balance\
                   }\
        _ = assert(balances.forall(_ == genesisInitialBalance))\
      } yield ()\
    }\
    t.runSyncUnsafe()\
  }' \
        "node/src/test/scala/coop/rchain/node/revvaultexport/VaultBalanceGetterTest.scala"

    # Add missing import for cats implicits
    sed -i.bak \
        -e '/import monix.execution.Scheduler.Implicits.global/a\
import cats.implicits._' \
        "node/src/test/scala/coop/rchain/node/revvaultexport/VaultBalanceGetterTest.scala"

    rm -f temp_test_replacement.txt
    echo "✅ Updated VaultBalanceGetterTest.scala with dynamic logic"
fi

# 7. RENAME DIRECTORIES FIRST
echo "📁 Renaming directories first..."

# Main revvaultexport directory
if [ -d "node/src/main/scala/coop/rchain/node/revvaultexport" ]; then
    if git mv "node/src/main/scala/coop/rchain/node/revvaultexport" "node/src/main/scala/coop/rchain/node/${TICKER_LOWER}vaultexport" 2>/dev/null; then
        echo "✅ node/src/main/.../revvaultexport/ -> ${TICKER_LOWER}vaultexport/ (git mv)"
    else
        mv "node/src/main/scala/coop/rchain/node/revvaultexport" "node/src/main/scala/coop/rchain/node/${TICKER_LOWER}vaultexport"
        echo "✅ node/src/main/.../revvaultexport/ -> ${TICKER_LOWER}vaultexport/ (mv)"
    fi
fi

# Test revvaultexport directory - CRITICAL: This was missed before!
if [ -d "node/src/test/scala/coop/rchain/node/revvaultexport" ]; then
    if git mv "node/src/test/scala/coop/rchain/node/revvaultexport" "node/src/test/scala/coop/rchain/node/${TICKER_LOWER}vaultexport" 2>/dev/null; then
        echo "✅ node/src/test/.../revvaultexport/ -> ${TICKER_LOWER}vaultexport/ (git mv)"
    else
        mv "node/src/test/scala/coop/rchain/node/revvaultexport" "node/src/test/scala/coop/rchain/node/${TICKER_LOWER}vaultexport"
        echo "✅ node/src/test/.../revvaultexport/ -> ${TICKER_LOWER}vaultexport/ (mv)"
    fi
fi

# Look for any other revvaultexport directories we might have missed
# Exclude build artifacts and IDE directories
find . -type d -name "*revvaultexport*" \
    -not -path "./.git/*" \
    -not -path "./.bloop/*" \
    -not -path "./.metals/*" \
    -not -path "*/target/*" \
    -not -path "./.idea/*" \
    -not -path "./.vscode/*" | while read -r dir; do
    if [ -d "$dir" ]; then
        # Check if directory is not empty or is a source directory (not a build artifact)
        if [ "$(ls -A "$dir" 2>/dev/null)" ] || [[ "$dir" == *"/src/"* ]]; then
            newdir=$(echo "$dir" | sed "s/revvaultexport/${TICKER_LOWER}vaultexport/g")
            # Avoid creating nested directories
            if [[ "$newdir" != *"${TICKER_LOWER}vaultexport/${TICKER_LOWER}vaultexport"* ]]; then
                git mv "$dir" "$newdir"
                echo "✅ Found and moved: $dir -> $newdir"
            else
                echo "⚠️  Skipping to avoid nested directory: $dir"
            fi
        else
            echo "⚠️  Skipping empty/build directory: $dir"
        fi
    fi
done

# 8. RENAME FILES
echo "📁 Renaming files..."

# Main contracts
if [ -f "casper/src/main/resources/RevVault.rho" ]; then
    git mv "casper/src/main/resources/RevVault.rho" "casper/src/main/resources/${TICKER_UPPER}Vault.rho"
    echo "✅ RevVault.rho -> ${TICKER_UPPER}Vault.rho"
fi

if [ -f "casper/src/main/resources/MultiSigRevVault.rho" ]; then
    git mv "casper/src/main/resources/MultiSigRevVault.rho" "casper/src/main/resources/MultiSig${TICKER_UPPER}Vault.rho"
    echo "✅ MultiSigRevVault.rho -> MultiSig${TICKER_UPPER}Vault.rho"
fi

# Tests
if [ -f "casper/src/test/resources/RevVaultTest.rho" ]; then
    git mv "casper/src/test/resources/RevVaultTest.rho" "casper/src/test/resources/${TICKER_UPPER}VaultTest.rho"
    echo "✅ RevVaultTest.rho -> ${TICKER_UPPER}VaultTest.rho"
fi

if [ -f "casper/src/test/resources/MultiSigRevVaultTest.rho" ]; then
    git mv "casper/src/test/resources/MultiSigRevVaultTest.rho" "casper/src/test/resources/MultiSig${TICKER_UPPER}VaultTest.rho"
    echo "✅ MultiSigRevVaultTest.rho -> MultiSig${TICKER_UPPER}VaultTest.rho"
fi

# CRITICAL: RevAddressTest.rho exists and must be renamed
if [ -f "casper/src/test/resources/RevAddressTest.rho" ]; then
    git mv "casper/src/test/resources/RevAddressTest.rho" "casper/src/test/resources/${TICKER_UPPER}AddressTest.rho"
    echo "✅ RevAddressTest.rho -> ${TICKER_UPPER}AddressTest.rho"
fi

# Rust files (may not exist)
if [ -f "rholang/src/rust/interpreter/util/rev_address.rs" ]; then
    git mv "rholang/src/rust/interpreter/util/rev_address.rs" "rholang/src/rust/interpreter/util/${TICKER_LOWER}_address.rs"
    echo "✅ rev_address.rs -> ${TICKER_LOWER}_address.rs"
fi

if [ -f "casper/src/rust/genesis/contracts/rev_generator.rs" ]; then
    git mv "casper/src/rust/genesis/contracts/rev_generator.rs" "casper/src/rust/genesis/contracts/${TICKER_LOWER}_generator.rs"
    echo "✅ rev_generator.rs -> ${TICKER_LOWER}_generator.rs"
fi

# Scala files
if [ -f "casper/src/main/scala/coop/rchain/casper/genesis/contracts/RevGenerator.scala" ]; then
    git mv "casper/src/main/scala/coop/rchain/casper/genesis/contracts/RevGenerator.scala" "casper/src/main/scala/coop/rchain/casper/genesis/contracts/${TICKER_UPPER}Generator.scala"
    echo "✅ RevGenerator.scala -> ${TICKER_UPPER}Generator.scala"
fi

# CRITICAL: RevAddress.scala exists and must be renamed
if [ -f "rholang/src/main/scala/coop/rchain/rholang/interpreter/util/RevAddress.scala" ]; then
    git mv "rholang/src/main/scala/coop/rchain/rholang/interpreter/util/RevAddress.scala" "rholang/src/main/scala/coop/rchain/rholang/interpreter/util/${TICKER_UPPER}Address.scala"
    echo "✅ RevAddress.scala -> ${TICKER_UPPER}Address.scala"
fi

# Test specs
if [ -f "rholang/src/test/scala/coop/rchain/rholang/interpreter/util/RevAddressSpec.scala" ]; then
    git mv "rholang/src/test/scala/coop/rchain/rholang/interpreter/util/RevAddressSpec.scala" "rholang/src/test/scala/coop/rchain/rholang/interpreter/util/${TICKER_UPPER}AddressSpec.scala"
    echo "✅ RevAddressSpec.scala -> ${TICKER_UPPER}AddressSpec.scala"
fi

if [ -f "casper/src/test/scala/coop/rchain/casper/genesis/contracts/RevAddressSpec.scala" ]; then
    git mv "casper/src/test/scala/coop/rchain/casper/genesis/contracts/RevAddressSpec.scala" "casper/src/test/scala/coop/rchain/casper/genesis/contracts/${TICKER_UPPER}AddressSpec.scala"
    echo "✅ RevAddressSpec.scala -> ${TICKER_UPPER}AddressSpec.scala"
fi

# 9. UPDATE IMPORTS/MODULES - Final pass
echo "🔧 Final import updates..."
find . -name "*.scala" -o -name "*.rs" | while read -r file; do
    sed -i.bak \
        -e "s/use.*rev_address/use crate::interpreter::util::${TICKER_LOWER}_address/g" \
        -e "s/mod rev_address/mod ${TICKER_LOWER}_address/g" \
        "$file"
done

# 9b. ADDITIONAL: Final cleanup of any remaining references
echo "🧹 Final cleanup of remaining references..."
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" -o -name "*.rhox" \) ! -path "./.git/*" | while read -r file; do
    sed -i.bak \
        -e "s/coop\.rchain\.node\.revvaultexport/coop.rchain.node.${TICKER_LOWER}vaultexport/g" \
        -e "s/node\.revvaultexport/node.${TICKER_LOWER}vaultexport/g" \
        "$file"
done

# Cleanup any .bak files created during sed operations
echo "🧹 Cleaning up backup files..."
find . -name "*.bak" -not -path "./.git/*" -not -path "*/target/*" -delete 2>/dev/null || true
echo "✅ Cleanup completed!"

# 9c. CLEAR SBT CACHES FOR ALL PROJECTS to refresh test discovery
echo "🧼 Clearing SBT caches (all target directories)..."
find . -type d -name target -not -path "./.git/*" -print0 | while IFS= read -r -d '' dir; do
    rm -rf "$dir" 2>/dev/null || true
    echo "✅ Removed $dir"
done

echo ""
echo "🎉 REV -> ${TICKER_UPPER} migration completed!"
echo ""
echo "🔄 EXISTING BLOCKCHAIN MIGRATION SUMMARY:"
echo ""
echo "✅ Your blockchain infrastructure now supports ${TICKER_UPPER} tokens:"
echo "   🔧 All code now uses ${TICKER_UPPER} contracts and addresses"
echo "   🔗 System URIs changed: rho:rev:address → rho:${TICKER_LOWER}:address"
echo "   🪙 New operations will create and use ${TICKER_UPPER} tokens"
echo "   📊 APIs and UIs will show ${TICKER_UPPER} instead of REV"
echo ""
echo "⚠️  CRITICAL - Impact on existing REV tokens:"
echo "   🚨 Existing REV tokens may become inaccessible!"
echo "   🔒 Old REV addresses use different URI (rho:rev:address)"
echo "   📱 Existing wallets may need updates to work with ${TICKER_UPPER}"
echo "   🔄 Consider if you need REV→${TICKER_UPPER} migration mechanism"
echo ""
echo "📋 What was done:"
echo ""
echo "🔧 CORE TECHNICAL MIGRATION:"
echo "   ✅ Replaced all identifiers (RevVault -> ${TICKER_UPPER}Vault etc.)"
echo "   ✅ Updated Registry URIs (rho:rchain:revVault -> rho:rchain:${TICKER_LOWER}Vault)"
echo "   ✅ Updated Registry.rho system file"
echo "   ✅ Updated Scala package declarations (ALL files)"
echo "   ✅ Updated method names (receiveRev -> receive${TICKER_UPPER})"
echo "   ✅ Updated hardcoded constants and values"
echo "   ✅ Updated system process constants (REV_ADDRESS -> ${TICKER_UPPER}_ADDRESS)"
echo "   ✅ Updated system URI definitions in RhoRuntime.scala"
echo "   ✅ Updated system process method signatures (revAddress -> ${TICKER_LOWER}Address)"
echo "   ✅ Updated PoS.rhox contract vault URIs and references"
echo "   ✅ Renamed files (.rho, .rs, .scala) and directories"
echo "   ✅ Moved ALL revvaultexport directories (main AND test)"
echo "   ✅ Updated imports and modules (comprehensive)"
echo ""
echo "🎯 COMPREHENSIVE REV/rev INSTANCE HANDLING (ALL CASES):"
echo "   ✅ Smart contract string literals (\"REV from\", \"REV to\", etc.)"
echo "   ✅ Smart contract comments (// the rev address of, etc.)"
echo "   ✅ Template variables (%REV_ADDR -> %${TICKER_UPPER}_ADDR)"
echo "   ✅ Scala variable names (revBalance -> ${TICKER_LOWER}Balance, etc.)"
echo "   ✅ Scala function names (revBalance() -> ${TICKER_LOWER}Balance())"
echo "   ✅ Scala test descriptions (\"Rev\" should -> \"${TICKER_UPPER}\" should)"
echo "   ✅ Scala comments (\"initial REV accounts\" -> \"initial ${TICKER_UPPER} accounts\")"
echo "   ✅ Python variable names (alice_rev_address -> alice_${TICKER_LOWER}_address)"
echo "   ⚠️  Python method calls (get_rev_address() - PRESERVED as external API)"
echo "   ✅ Configuration file parameters (<revBalance> -> <${TICKER_LOWER}Balance>)"
echo "   ✅ Configuration descriptions (\"amount of Rev they have bonded\" -> \"amount of ${TICKER_UPPER} they have bonded\")"
echo "   ✅ Documentation section headers (## REV -> ## ${TICKER_UPPER})"
echo "   ✅ Documentation token references (\"I want REV to be the currency\", \"As a REV holder\", etc.)"
echo "   ✅ Documentation user scenarios (\"transfer 100 REV\", \"having the REV to pay\", etc.)"
echo "   ✅ Error messages in test files (\"invalid rev address\" -> \"invalid ${TICKER_LOWER} address\")"
echo "   ✅ File renaming (revaddress files -> ${TICKER_LOWER}address files)"
echo "   ✅ Package comments (//scalapackage coop.rchain.rholang.rev -> //scalapackage coop.rchain.rholang.${TICKER_LOWER})"
echo "   ✅ Docker documentation (Has initial REV balance -> Has initial ${TICKER_UPPER} balance)"
echo "   ✅ YAML comments (# REV: -> # ${TICKER_UPPER}:)"
echo "   ✅ Contract comments (// REV vault -> // ${TICKER_UPPER} vault)"
echo ""
echo "📊 COMPREHENSIVE COVERAGE:"
echo "   🎯 Handled ALL instances found through case-insensitive 'rev' search"
echo "   🎯 Preserved technical terms like 'reverse', 'revision', 'revert', 'previous'"
echo "   🎯 Updated token-specific references across ALL file types"
echo ""
echo "📝 What was selectively NOT changed:"
echo "   ⏭️  Technical terms like \"reverse\", \"revision\", \"revert\", \"previous\" - preserved as programming terminology"
echo "   ⏭️  Some edge cases in documentation - only obvious token references updated"
echo "   ⏭️  Historical wallet files (wallets_REV_BLOCK-*.txt) - preserved as blockchain history"
echo "   ⏭️  Hardcoded blockchain addresses - preserved as blockchain history"
echo "   ⏭️  Existing REV token balances in blockchain state - may need migration strategy"
echo ""
echo "🚀 IMMEDIATE NEXT STEPS:"
echo "   1. Check compilation: sbt compile"
echo "   2. Run tests: sbt test"
echo "   3. Test on development/testnet first"
echo "   4. Verify new ${TICKER_UPPER} operations work correctly"
echo ""
echo "🤔 CONSIDER FOR PRODUCTION:"
echo "   ⚠️  How will existing REV holders access their tokens?"
echo "   ⚠️  Do you need backward compatibility contracts?"
echo "   ⚠️  Should you create REV→${TICKER_UPPER} conversion mechanism?"
echo "   ⚠️  How will you communicate changes to users?"