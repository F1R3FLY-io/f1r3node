# Brain Dump

* Reproducible development environment
  * [Nix](https://github.com/DeterminateSystems/nix-installer)
  * [typelevel-nix](https://github.com/typelevel/typelevel-nix)
  * [direnv](https://direnv.net/)
  * [nix-direnv](https://github.com/nix-community/nix-direnv)
  * The structure of `flake.nix`
    * 90% of everything comes from whatever nixpkgs typelevel-nix uses. That's very much the point.
    * JVM 17 is considered appropriate for "application," vs. "library," projects.
    * I decided to use GraalVM to have access to its JavaScript and native-image systems.
    * Current nixpkgs only gives us JVM 21 (because that's all the upstream does), so I had to refer to a specific recent nixpkgs that offers the older "Community Edition" GraalVM versions.
    * To avoid confusion, I set the "jdk" and "jre" attributes to use GraalVM, so anything using a JVM from nixpkgs will also use GraalVM.
    * I wanted the [Ammonite REPL](https://ammonite.io/#Ammonite-REPL) to be available, but the one in current nixpkgs is out of date _and_ relies on a more recent version of Scala than we use. My inclusion of the Ammonite REPL addresses those issues.
    * Rust support is provided by a very popular and well-maintained overlay.
    * The rest of the setup is straightforward, following the [devshell](https://numtide.github.io/devshell/) documentation.
    * At one point I had included GraalVM's `js` and `native-image` commands. I took them out because we weren't using them and I wanted to reduce noisy distractions. But be aware that, e.g. it's entirely possible to write Scala code that embeds JavaScript and executes it, which might be useful in e.g. some testing scenarios.
* Building
  * [`sbt`](https://www.scala-sbt.org/) is the standard Scala build tool. It's important that you know and understand this.
    * You can invoke sbt commands from your shell in one go, e.g. `sbt ';compile ;project node ;Docker/publishLocal'`, but this incurs the overhead of launching `sbt` each time, so I recommend just having a terminal with `sbt` running.
    * There are several subprojects in the system. The root one is named `rchain`, as you will see at the `sbt:` prompt.
    * The key commands you can use are `project <name>` to switch to a given subproject, `clean` to clean the build, `compile` to compile the (sub)project, and some subproject-specific commands provided by plugins.
    * The main subproject is `node`, which is the F1r3fly blockchain node project.
    * In the `sbt` shell, you can issue multiple commands, _prepended_ with `;`.
    * In the `node` subproject, there are two important commands: `assembly` builds a "fat `.jar`" file you can run with `java -jar`, `Docker/publishLocal` publishes a Docker container locally, and `Docker/publish` publishes a Docker container to whatever repository is configured.
    * The project is configured to use [sbt-release](https://github.com/sbt/sbt-release) to manage versioning and publishing (in both the sbt/Maven and Docker senses) formal (pre-)releases. It's important that you know and understand this.
    * As of this writing, `compile` sometimes fails due to an apparent bug in Scala 2.12.x. The only known solution is to try again until it works.
* Running
  * Prepare a `data-dir` with a subdirectory, `genesis`, containing a file, `bonds.txt`, with one line containing the public key of an secp256k1 key pair followed by a space and then a small integer. 1 will do fine.
  * `java --add-opens java.base/sun.security.util=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED -jar node/target/scala-2.12/rnode-assembly-1.0.0-SNAPSHOT.jar run -s --no-upnp --allow-private-addresses --synchrony-constraint-threshold=0.0 --validator-private-key <private key> --data-dir <data-dir>`
  * `<private key> is the key matching the public key in the `bonds.txt` file; `<data-dir>` is the `data-dir` you prepared.
  * This runs a standalone (`-s`) node, which means it won't try to bootstrap from another node, requires 0 signatures for block approval, and will act as the genesis ceremony master if it fails to find an existing approved block. The other argument are mostly conveniences for local use, e.g. don't bother trying to forward ports with UPnP.
  * The `data-dir` being empty apart from the `bonds.txt` file will cause the node to go through the genesis ceremony, self-approving the genesis block. Because the number of required signatures (0) is less than the number of active validators (1), this will succeed. Because the node is running with the validator-private-key that matches the bond in `bonds.txt`, the result is a "network" of a single bonded validator.
  * `docker/shard.yml` provides a Docker Compose file that stands up a network of 4 nodes, one "bootstrap" observer and three bonded validators, for experiments that benefit from having > 1 node.
* Philosophy
  * The fat "`.jar`" result of `;compile ;project node ;assembly` is ground truth. It is your top priority to maintain. It is the first thing to try to see whether something works or not. If it doesn't do what you expect, that's a drop-everything-else-you're-doing-and-fix-it event.
  * Ruthlessly eliminate variables. Docker, Kubernetes, etc. add variables. Running commands in another instance of the `node` `.jar` adds variables. Don't do that.
* Client-Side
  * F1r3fly's API is provided via [gRPC](https://grpc.io/). It's important that you know and understand this.  
  * The reproducible development environment provides [grpcurl](https://github.com/fullstorydev/grpcurl). It's important that you know and understand this.
  * All clients must have an implementation of `signDeploy`, which takes a Protobuf `DeployDataProto`. serializes several of its fields, hashes those bytes with Blake2b256, signs the hash with a private secp256k1 key, DER encodes the signature, and returns a new `DeployDataProto` with the `sig` field containing the encoded signature, the `deployer` field containing the compressed public key inferred from the private key, and the `sigAlgorithm` field containing "secp256k1".
  * See `scripts/playground.sc` for the most perspicuous (if I do say so myself) implementation of `signDeploy`.
* Deployment
  * The reality is you'll deploy on Kubernetes indefinitely—at the barest minimum, two years. It's important that you know and understand this.
  * Public cloud infrastructure, customer infrastructure, it doesn't matter, except insofar as, if you want these networks to talk to each other, you'll need to master [Kubernetes multi-clustering](https://www.tigera.io/learn/guides/kubernetes-networking/kubernetes-multi-cluster/).
  * [Rancher](https://www.rancher.com/) is extremely good for provisioning Kubernetes clusters, and in particular supports your first target, OKE.
  * Eventually, to deal with exotic hardware F1r3fly can take the best advantage of, you'll need to master [bare-metal Kubernetes](https://deploy.equinix.com/blog/guide-to-running-kubernetes-on-bare-metal/).
  * The reproducible development environment provides [Minikube](https://minikube.sigs.k8s.io/docs/), a _very_ good local single-node Kubernetes "cluster."
  * The reproducible development environment provides [Dhall](ihttps://dhall-lang.org/) and its YAML-generating CLI.
  * The project repository includes [dhall-kubernetes](https://github.com/dhall-lang/dhall-kubernetes) as a Git submodule. This may be useful for developing rich deployments.
  * Study other stateful services that can be deployed to Kubernetes, including other blockchains.
    * Strongly consider developing a F1r3fly [Operator](https://operatorframework.io/)
      * Using the [kubernetes-client](https://github.com/joan38/kubernetes-client)
    * Strongly consider using the [Operator Lifecycle Manager](https://olm.operatorframework.io/)
    * Ideally, the F1r3fly Operator should be listed in the [Operator Hub](https://operatorhub.io/)
    * Study the [Crunchy Data PostgreSQL Operator](https://access.crunchydata.com/documentation/postgres-operator/latest/architecture)
    * Study the [Hyperledger Besu Operator](https://github.com/hyperledger-labs/besu-operator)
    * Study the [Hyperledger Fabric Operator](https://hyperledger.github.io/bevel-operator-fabric/docs/)
    * Study the [Cosmos Operator](https://github.com/strangelove-ventures/cosmos-operator)
    * Consider integrating into [Kotal](https://github.com/kotalco/kotal)
* The Code
  * A time capsule of multiple teams with no engineering management and varying backgrounds and skills banging away.
  * Using a badly outdated version of Scala, at least one major unsupported library, and libraries that have been dramatically improved upon since.
  * Priorities (it's important that you know and understand this):
    * Upgrade ScalaTest to a version supported by [cats-effect-testing](https://github.com/typelevel/cats-effect-testing) and [testcontainers-scala](https://github.com/testcontainers/testcontainers-scala).
      * _Very_ strongly consider migrating from ScalaTest to [Weaver](https://disneystreaming.github.io/weaver-test/)
    * Upgrade to Scala 2.13.x.
    * Upgrade dependencies to those based on [cats-effect 3.x](https://typelevel.org/cats-effect/docs/getting-started), whose scheduler is [50-60x faster](https://gist.github.com/djspiewak/f4cfc08e0827088f17032e0e9099d292) than cats-effect 2.x's.
    * Remove the unmaintained [Monix](https://monix.io/) dependency and replace all the Monix-gRPC machinery with [fs2-grpc](https://github.com/typelevel/fs2-grpc).
    * For the love of all that is holy, remove Scapegoat, add [sbt-tpolecat](https://github.com/typelevel/sbt-tpolecat), and strictly use the [unsafe Wart list](https://www.wartremover.org/doc/install-setup.html) from WartRemover.
    * Generally, refactor over time to reflect style from [Scala With Cats](https://underscore.io/books/scala-with-cats/) and [Practical FP in Scala](https://leanpub.com/pfp-scala).
* The Programming Model
  * Re-architect the API to be almost, if not entirely, streaming. Be sure to eradicate polling.
    * See [this fs2-grpc example](https://github.com/fiadliel/fs2-grpc-example/tree/master) of both an RPC and a streaming API.
    * See [this TypeScript example](https://floydfajones.medium.com/creating-your-grpc-service-in-nodejs-typescript-part-2-19464c73320b) of gRPC with streaming.
    * See e.g. [web3j's managed filter approach for Ethereum](https://docs.web3j.io/4.11.0/advanced/filters_and_events/), but don't actually do a client-side polling implementation.
    * See [Functional Event-Driven Architecture](https://leanpub.com/feda) for a great book on modern distributed systems architecture.
    * See [RxDB](https://rxdb.info/) for an example of a database system designed along async/streaming principles.
    * See [React-RxJS](https://react-rxjs.org/) for an example of building React UIs with an async/streaming architecture.
    * [RethinkDB](https://rethinkdb.com/) is another example of an async/streaming database system.
  * It must not be necessary to "deploy Rholang" just to invoke a smart contract and get results. See [web3.eth.Contract](https://web3js.readthedocs.io/en/v1.2.11/web3-eth-contract.html) for the right way to approach this.

## An example tying the above together, hopefully

Assumption: you are in a terminal at the project's root with a successfully build fat `.jar` as described above, and as a consequence, in the reproducible development environment. Monospaced examples starting with `$` below are command invocations in your shell.

`$ rm -fr ~/.rnode`

By default, the node uses `~/.rnode` as its `data-dir`. Let's take advantage of this, and start fresh.

`$ amm`

We need to generate an secp256k1 key pair, getting both keys in hexadecimal form. The Ammonite REPL prompt is `@`.

```
@ import $exec.scripts.playground
@ ECPrivateKey.freshPrivateKey
SLF4J: No SLF4J providers were found.
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See https://www.slf4j.org/codes.html#noProviders for further details.
res1: ECPrivateKey = ECPrivateKey(Chunk(View(scodec.bits.ByteVector$AtArray@574b7f4a, 1L, 32L)))

@ res1.publicKey.decompressedHex
res3: String = "04458c90e163dcb143b7e358b160a794a6eb85301518a7f46aa5c6f9514188f876809fbd5607d1d338f9056da9e07026c6b5e9ff56d057797fc4c8b3f2f8be76b3"

@ res1.hex
res4: String = "f004ccc5955cef0b7ae52953d861f749bbc8e18b1484217209777af1b4fced4c"

@ <ctl-d>
```

`$ mkdir -p ~/.rnode/genesis`

For convenience, copy the public key hex value above, and create a file in `~/.rnode/genesis` with that as its name, with `.sk` appended:

`$ hx ~/.rnode/genesis/04458c90e163dcb143b7e358b160a794a6eb85301518a7f46aa5c6f9514188f876809fbd5607d1d338f9056da9e07026c6b5e9ff56d057797fc4c8b3f2f8be76b3.sk`

In the file you're now editing, copy and paste the private key hex value above. Save the file.

Now edit `~/.rnode/genesis/bonds.txt`, and copy and paste the public key above. Put a space and a `1` after it on the same line. Save the file:

`04458c90e163dcb143b7e358b160a794a6eb85301518a7f46aa5c6f9514188f876809fbd5607d1d338f9056da9e07026c6b5e9ff56d057797fc4c8b3f2f8be76b3 1`

Now run the node as described above, not specifying `data-dir` explicitly and providing the above private key:

`$ java --add-opens java.base/sun.security.util=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED -jar node/target/scala-2.12/rnode-assembly-1.0.0-SNAPSHOT.jar run -s --no-upnp --allow-private-addresses --synchrony-constraint-threshold=0.0 --validator-private-key f004ccc5955cef0b7ae52953d861f749bbc8e18b1484217209777af1b4fced4c`

The logs should include:

`Approved block not found, taking part in ceremony as ceremony master`

and one "Bond loaded" line, e.g.

`Bond loaded 04458c90e163dcb143b7e358b160a794a6eb85301518a7f46aa5c6f9514188f876809fbd5607d1d338f9056da9e07026c6b5e9ff56d057797fc4c8b3f2f8be76b3 => 1`

You should also see:

`Starting execution of ApprovedBlockProtocol. Waiting for 0 approvals from genesis validators.`
`Self-approving genesis block.`
`Finished execution of ApprovedBlockProtocol`
`Making a transition to Running state. Approved Block #0 (644386340c...) with empty parents (supposedly genesis)`

These indicate that the node is not waiting for other validators, is unilaterally approving the genesis block, and is now ready to do work to be validated by the one bonded validator in the network—itself.

Open another terminal. Make sure your new shell is also in the project root, i.e. the reproducible development environment.

```
$ amm
...
@ import $exec.scripts.playground
...
```

Copy the contents of `rholang/examples/tut-registry.rho` to your clipboard. Then:

```
@ val rho = """
<paste clipboard contents here>
"""
...
@ val vpk = "aebb63dc0d50e4dd29ddd94fb52103bfe0dc4941fa0c2c8a9082a191af35ffa1"
...
@ val ddp = DeployDataProto(
    term = rho,
    timestamp = 0,
    phloPrice = 1,
    phloLimit = 1000000,
    shardId = "root",
    validAfterBlockNumber = 0
  )
...
@ signDeployJSON(vpk, ddp)
```

The value of `vpk` can be any valid secp256k1 private key. It has no meaning other than to let F1r3fly validate the signature against the `deployer` field, which is just the public key inferred from this private key.

Copy the resulting JSON object to your clipboard. Then:

`$ grpcurl -plaintext -d '<paste here>' --import-path ./node/target/protobuf_external --import-path ./models/src/main/protobuf --proto DeployServiceV1.proto 127.0.0.1:40402 casper.v1.DeployService.doDeploy`

This should result in something like:

```
{
  "result": "Success!\nDeployId is: 30450221009e2ef79896ccdb40532a489ddaeaf3ec7854051d9c05f2f83d65f87e6e78ec340220558b6c349d2fcbee48969d7539413dc29a14ed440600b5b026d4c8363db0fd0b"
}
```

Then:

```
$ grpcurl -plaintext --import-path ./node/target/protobuf_external --import-path ./models/src/main/protobuf --proto ProposeServiceV1.proto 127.0.0.1:40402 casper.v1.ProposeService.propose
```

This should result in something like:

```
{
  "result": "Success! Block aebe28462b4bfc6fb7cbdf3bea2364931f03a1fd6b6545c046beb32d2db36bac created and added."
}
```

Then copy the DeployId above, from _before_ the `propose`, and:

```
$ amm
...
@ import $exec.scripts.playground
...
@
ByteVector.fromHex("<paste here>").get.toBase64
...
@ <ctl-d>
```

Copy the base 64 output from the above to your clipboard. Then:

```
$ grpcurl -plaintext -d '{"deployId": "<paste here>"}' --import-path ./node/target/protobuf_external --import-path ./models/src/main/protobuf --proto DeployServiceV1.proto 127.0.0.1:40402 casper.v1.DeployService.findDeploy
```

This should result in something like:

```
{
  "blockInfo": {
    "blockHash": "aebe28462b4bfc6fb7cbdf3bea2364931f03a1fd6b6545c046beb32d2db36bac",
    "sender": "04458c90e163dcb143b7e358b160a794a6eb85301518a7f46aa5c6f9514188f876809fbd5607d1d338f9056da9e07026c6b5e9ff56d057797fc4c8b3f2f8be76b3",
    "seqNum": "1",
    "sig": "3045022100c9853232c94a60e85478569014b994b1317e1ec8cbae2d857c039991bd2a383702205677183b8accbd8d457371ca1b2f3bba7a760caa5ae60b22b8bb46522f5209ca",
    "sigAlgorithm": "secp256k1",
    "shardId": "root",
    "version": "1",
    "timestamp": "1708617943958",
    "parentsHashList": [
      "644386340cc4ef011e757313e9a95720db0c0ef1f06d0f202e392e9fbe627d62"
    ],
    "blockNumber": "1",
    "preStateHash": "a3af96b667e00198761bed1356079a54172852e965f56bc516c51acc17e94b1c",
    "postStateHash": "4903d1ed03dc4255c710a1501b98ce47221173914f7cb4d1b3726b6879f782ae",
    "bonds": [
      {
        "validator": "04458c90e163dcb143b7e358b160a794a6eb85301518a7f46aa5c6f9514188f876809fbd5607d1d338f9056da9e07026c6b5e9ff56d057797fc4c8b3f2f8be76b3",
        "stake": "1"
      }
    ],
    "blockSize": "69387",
    "deployCount": 1,
    "faultTolerance": 1,
    "justifications": [
      {
        "validator": "04458c90e163dcb143b7e358b160a794a6eb85301518a7f46aa5c6f9514188f876809fbd5607d1d338f9056da9e07026c6b5e9ff56d057797fc4c8b3f2f8be76b3",
        "latestBlockHash": "644386340cc4ef011e757313e9a95720db0c0ef1f06d0f202e392e9fbe627d62"
      }
    ]
  }
}
```

Copy the `blockHash` to your clipboard. Then:

```
$ grpcurl -plaintext -d '{"hash": "<paste here>"}' --import-path ./node/target/protobuf_external --import-path ./models/src/main/protobuf --proto DeployServiceV1.proto 127.0.0.1:40402 casper.v1.DeployService.isFinalized
```

This should result in:

```
{
  "isFinalized": true
}
```

In reality, you'll need to loop over the "is finalized" check until it either succeeds or you give up.

This is what it takes to do anything with this blockchain. It doesn't even get to the next step of asking for data.

As a client programmer, of course you need to do all of this programmatically, with the gRPC client library for your language and whatever else you can bring to bear. Here's what this looks like with the [Mutiny](https://smallrye.io/smallrye-mutiny/latest/) library in Java:

```java
      	String rhoCode = loadStringResource( onChainVolumeCode );
      	// Make deployment
      	DeployDataProto deployment = DeployDataProto.newBuilder()
              .setTerm(rhoCode)
              .setTimestamp(0)
              .setPhloPrice(1)
              .setPhloLimit(1000000)
              .setShardId("root")
              .build();
      
      	// Sign deployment
      	DeployDataProto signed = signDeploy(deployment);
      
      	// Deploy
      	Uni<Void> deployVolumeContract =
        Uni.createFrom().future(deployService.doDeploy(signed))
        .flatMap(deployResponse -> {
            if (deployResponse.hasError()) {
                return this.<String>fail(deployResponse.getError());
            } else {
                return succeed(deployResponse.getResult());
            }
        })
        .flatMap(deployResult -> {
            String      deployId   = deployResult.substring(deployResult.indexOf("DeployId is: ") + 13, deployResult.length());
            return Uni.createFrom().future(proposeService.propose(ProposeQuery.newBuilder().setIsAsync(false).build()))
            .flatMap(proposeResponse -> {
                if (proposeResponse.hasError()) {
                    return this.<String>fail(proposeResponse.getError());
                } else {
                    return succeed(deployId);
                }
            });
        })
        .flatMap(deployId -> {
            ByteString  b64        = ByteString.copyFrom(Hex.decode(deployId));
            return Uni.createFrom().future(deployService.findDeploy(FindDeployQuery.newBuilder().setDeployId(b64).build()))
            .flatMap(findResponse -> {
                if (findResponse.hasError()) {
                    return this.<String>fail(findResponse.getError());
                } else {
                    return succeed(findResponse.getBlockInfo().getBlockHash());
                }
            });
        })
        .flatMap(blockHash -> {
            return Uni.createFrom().future(deployService.isFinalized(IsFinalizedQuery.newBuilder().setHash(blockHash).build()))
            .flatMap(isFinalizedResponse -> {
                if (isFinalizedResponse.hasError() || !isFinalizedResponse.getIsFinalized()) {
                    return fail(isFinalizedResponse.getError());
                } else {
                    return Uni.createFrom().voidItem();
                }
            })
            .onFailure().retry()
            .withBackOff(INIT_DELAY, MAX_DELAY)
            .atMost(RETRIES);
        });

        // Drummer Hoff Fired It Off
        deployVolumeContract.await().indefinitely();
```
