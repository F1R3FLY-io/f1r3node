#!/bin/bash

set -e

mkdir -p ./rust_libraries/release

cargo build --release -p rspace_plus_plus_rhotypes
cp -r ./target/release/librspace_plus_plus_rhotypes.* ./rust_libraries/release

cargo build --release -p models

cargo build --release -p rholang
cp -r ./target/release/librholang.* ./rust_libraries/release

cargo build --release -p crypto
cp -r ./target/release/libcrypto.* ./rust_libraries/release
