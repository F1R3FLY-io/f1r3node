#!/bin/bash

set -e

rm -rf rust_libraries/

cd rspace++/
cargo clean

cd ../rholang/
cargo clean

cd ../crypto/
cargo clean

cd ../models/
cargo clean

cd ../casper/
cargo clean

cd ../shared/
cargo clean

