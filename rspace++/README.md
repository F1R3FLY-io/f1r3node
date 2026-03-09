# rspace++

The **R**Chain Tuple **Space** in **Rust**

### Overview

`rspace++` is a direct port of the `rspace` Scala library to Rust. See [rspace documentation](../rspace/README.md) for more information about rspace and how to use the Scala library.

`JNA` is used to load the `rspace++` Rust library into the Scala project.

## Development

### Prerequisites

* [Environment set up](../README.md#installation).

### Building

To build the `rspace++` Rust library, run `cargo build --release -p rspace_plus_plus_rhotypes`
  - `cargo build --profile dev -p rspace_plus_plus_rhotypes` will build the library in debug mode

## Testing

### Rust

Run all tests: `cargo test`

Run all tests in release mode: `cargo test --release`
  
Run specific test file: `cargo test --test <test_file_name>`

Run specific test file in specific folder: `cargo test --test <test_folder_name>::<test_file_name>`

### Scala

The following tests all use `rspace++` and must be run at root project directory:

- Run basic RSpace-Bench Benchmark: `sbt "rspaceBench/jmh:run -i 10 -wi 10 -f1 -t1 .BasicBench."`
- Run Casper Genesis tests: `sbt "casper/testOnly coop.rchain.casper.genesis.GenesisTest"`
- Run Casper Rholang tests: `sbt ";casper/testOnly coop.rchain.casper.batch1.MultiParentCasperRholangSpec"`
- Run Casper Block tests: `sbt ";casper/testOnly coop.rchain.casper.addblock.MultiParentCasperAddBlockSpec"`
- Run Casper Propose test: `sbt ";casper/testOnly coop.rchain.casper.addblock.ProposerSpec"`


<!-- ## Quickstart -->

<!-- - For setting up `nix` and `direnv`, see [project overview](../docs/paul_brain_dump.md)
- Make sure you have [protobuf](https://grpc.io/docs/protoc-installation/) installed -->

<!-- To get in and out of `direnv` you can use the following:
- `direnv allow` in root project directory
- `direnv revoke` then exit root project directory. Coming back into root project directory you will be out of nix shell -->

<!-- Starting standalone node using RSpace++
1. `sbt ";clean ;compile ;stage"`
2. `./node/target/universal/stage/bin/rnode -Djna.library.path=./rust_libraries/debug run --standalone` in one terminal
3. In a another terminal, execute rholang: `./node/target/universal/stage/bin/rnode -Djna.library.path=./rust_libraries/debug eval rholang/examples/stdout.rho` -->

<!-- Standing up network using RSpace++
1. Follow these instructions on setting up `.rnode` directory [setting up rnode directory](../docs/paul_brain_dump.md#an-example-tying-the-above-together-hopefully) stopping just before you execute the java command that starts the node
2. `sbt ";clean ;compile ;project node ;assembly ;project rchain"`
3. `java -Djna.library.path=./rust_libraries/debug --add-opens java.base/sun.security.util=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED -jar node/target/scala-2.12/rnode-assembly-0.0.0-unknown.jar run -s --no-upnp --allow-private-addresses --synchrony-constraint-threshold=0.0 --validator-private-key <your_validator_key>`
4. (Optional) Run this command to ensure node performs genesis ceremony: `rm -rf ~/.rnode/casperbuffer/ ~/.rnode/dagstorage/ ~/.rnode/deploystorage/ ~/.rnode/blockstorage/ ~/.rnode/rnode.log ~/.rnode/rspace++/ ~/.rnode/node.certificate.pem ~/.rnode/node.key.pem`

Standing up network using RSpace++ (Under Docker)
1. `docker context use default`
2. `sbt ";clean ;compile ;project node ;Docker/publishLocal ;project rchain"`
3. `docker compose -f docker/shard.yml up`
 
 - (Optional) Run this command to ensure nodes preform genesis ceremony: `scripts/delete_data.sh`<br>
 - Sometimes the nodes will not reach a fully complete state so it is recommended to delete the container after every restart -->