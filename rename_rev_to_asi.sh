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
# Ticker team should update documentation themselves based on their future features
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" \) ! -path "./.git/*" | while read -r file; do
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
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" \) ! -path "./.git/*" | while read -r file; do
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

# 4b. ADDITIONAL: Update any remaining revvaultexport references in all file types
echo "🔄 Updating any remaining revvaultexport references..."
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" \) ! -path "./.git/*" | while read -r file; do
    sed -i.bak \
        -e "s/coop\.rchain\.node\.revvaultexport/coop.rchain.node.${TICKER_LOWER}vaultexport/g" \
        "$file"
done

# 5. UPDATE COMMENTS AND DOCUMENTATION (selective)
echo "📚 Updating comments..."
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" \) ! -path "./.git/*" | while read -r file; do
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

# 7. RENAME DIRECTORIES FIRST
echo "📁 Renaming directories first..."

# Main revvaultexport directory
if [ -d "node/src/main/scala/coop/rchain/node/revvaultexport" ]; then
    git mv "node/src/main/scala/coop/rchain/node/revvaultexport" "node/src/main/scala/coop/rchain/node/${TICKER_LOWER}vaultexport"
    echo "✅ node/src/main/.../revvaultexport/ -> ${TICKER_LOWER}vaultexport/"
fi

# Test revvaultexport directory - CRITICAL: This was missed before!
if [ -d "node/src/test/scala/coop/rchain/node/revvaultexport" ]; then
    git mv "node/src/test/scala/coop/rchain/node/revvaultexport" "node/src/test/scala/coop/rchain/node/${TICKER_LOWER}vaultexport"
    echo "✅ node/src/test/.../revvaultexport/ -> ${TICKER_LOWER}vaultexport/"
fi

# Look for any other revvaultexport directories we might have missed
find . -type d -name "*revvaultexport*" -not -path "./.git/*" -not -path "*/target/*" | while read -r dir; do
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
find . -type f \( -name "*.scala" -o -name "*.rs" -o -name "*.rho" \) ! -path "./.git/*" | while read -r file; do
    sed -i.bak \
        -e "s/coop\.rchain\.node\.revvaultexport/coop.rchain.node.${TICKER_LOWER}vaultexport/g" \
        -e "s/node\.revvaultexport/node.${TICKER_LOWER}vaultexport/g" \
        "$file"
done

# Cleanup any .bak files created during sed operations
echo "🧹 Cleaning up backup files..."
find . -name "*.bak" -not -path "./.git/*" -not -path "*/target/*" -delete 2>/dev/null || true
echo "✅ Cleanup completed!"

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
echo "   ✅ Replaced all identifiers (RevVault -> ${TICKER_UPPER}Vault etc.)"
echo "   ✅ Updated Registry URIs (rho:rchain:revVault -> rho:rchain:${TICKER_LOWER}Vault)"
echo "   ✅ Updated Registry.rho system file"
echo "   ✅ Updated Scala package declarations (ALL files)"
echo "   ✅ Updated method names (receiveRev -> receive${TICKER_UPPER})"
echo "   ✅ Updated hardcoded constants and values"
echo "   ✅ Updated system process constants (REV_ADDRESS -> ${TICKER_UPPER}_ADDRESS)"
echo "   ✅ Updated system URI definitions in RhoRuntime.scala"
echo "   ✅ Updated system process method signatures (revAddress -> ${TICKER_LOWER}Address)"
echo "   ✅ Renamed files (.rho, .rs, .scala)"
echo "   ✅ Renamed directories and test specs"
echo "   ✅ Moved ALL revvaultexport directories (main AND test)"
echo "   ✅ Updated imports and modules (comprehensive)"
echo "   ✅ Updated comments in code files"
echo "   ✅ Final cleanup of all remaining references"
echo ""
echo "📝 What was NOT changed:"
echo "   ⏭️  Documentation files (.md, .txt, README, etc.) - ${TICKER_UPPER} team should update these"
echo "      based on their specific features and requirements"
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