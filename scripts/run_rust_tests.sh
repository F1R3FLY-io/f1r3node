#!/bin/bash

set -e

cd rspace++/
cargo test --release

cd ../rholang
cargo test --release

cd ../casper
cargo test --release

cd ../models
cargo test --release

cd ../crypto
cargo test --release

cd ../shared
cargo test --release

cd ../graphz
cargo test --release

cd ../block-storage
cargo test --release

cd ../comm
cargo test --release

cd ../node
cargo test --release
