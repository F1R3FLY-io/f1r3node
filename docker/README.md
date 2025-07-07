### Quickstart:

```
sbt ";clean ;compile ;project node ;Docker/publishLocal"
cd docker
docker compose -f ./shard.yml up # to start shard
docker compose -f ./shard.yml down # to stop shard
```

Test deploy (execute the following with running container):

```
grpcurl -plaintext  -d '{"deployer":"BChaWFkQyEoVg1gVvpccZyX07jjo2U741fCFbqmbn2wHdTNnN1eDSKEr4qqfJ0u6oJ9+V3INID3MTmc3RQ7aDcg=","term":"new return(`rho:rchain:deployId`) in { return!((42, true, \"Hello from blockchain!\")) }","sig":"MEQCIBPMx5E+15R+UM9ZV3STGtYegpwGmaizAtq8QtQshZK9AiBvShfjac6p3INXp4j6t9Nl66OrfK5KjRAcd+J4EmJdmA==","sigAlgorithm":"secp256k1","phloPrice":"1","phloLimit":"500000","shardId":"root"}' --import-path ./node/target/protobuf_external --import-path ./models/src/main/protobuf --proto DeployServiceV1.proto 127.0.0.1:40401 casper.v1.DeployService.doDeploy
```

Should see something like this:

```
{
  "result": "Success! DeployId is: 3044022013ccc7913ed7947e50cf595774931ad61e829c0699a8b302dabc42d42c8592bd02206f4a17e369cea9dc8357a788fab7d365eba3ab7cae4a8d101c77e27812625d98"
}
```

# Observer (Readonly) Node
The additional observer node that will connect to the shard network and will be able to read the state of the network:
```sh
docker compose -f ./observer.yml up # start observer
docker compose -f ./observer.yml down # stop observer
```

# Configuration needed for 4 nodes to run in shard mode 

- `standalone = false` in `conf/*.conf` files to run in shard mode
- put the keys (from comment from `shard.yaml`) to `casper.validator-public-key` and `casper.validator-private-key` in each `conf/*.conf` files
- empty value `protocol-client.bootstrap` in `conf/boostrap.conf` file
- `protocol-client.bootstrap = "rnode://138410b5da898936ec1dc13fafd4893950eb191b@rnode.bootstrap?protocol=40400&discovery=40404"` in `conf/validator1.conf`, `conf/validator2.conf`, `conf/validator2.conf` files. `138410b5da898936ec1dc13fafd4893950eb191b` refers to Bootstrap certificate and `rnode.bootstrap` to the hostname of the Bootstrap node in the docker compose configuration
- `casper.fork-choice-stale-threshold = 1 minutes` and `casper.fork-choice-check-if-stale-interval = 3 minutes` in `conf/*.conf` files to more frequent fork-choice checks
- `casper.synchrony-constraint-threshold = 0.33` in `conf/*.conf` files (or 0.5 for more strict synchrony constraint). *DON'T USE 0.0* as it will cause the node to not finalize any blocks.
- put list of public keys of all nodes into `pos-multi-sig-public-keys` in `conf/*.conf` files
- `casper.genesis-block-data.pos-multi-sig-quorum = 2` in `conf/*.conf` files to set the quorum for the shard
- `casper.genesis-ceremony.required-signatures = 2` in `conf/*.conf` files to set the number of signatures required for the genesis ceremony
- `casper.genesis-ceremony.approve-interval = 10 seconds` in `conf/*.conf` files for more frequent approval intervals
- `casper.genesis-ceremony.genesis-validator-mode = false` in `conf/bootstrap.conf` file
- `casper.genesis-ceremony.genesis-validator-mode = true` in `conf/validator1.conf`, `conf/validator2.conf`, `conf/validator2.conf` files
- `casper.genesis-ceremony.ceremony-master-mode = true` in `conf/bootstrap.conf` file
- `casper.genesis-ceremony.ceremony-master-mode = false` in `conf/validator1.conf`, `conf/validator2.conf`, `conf/validator2.conf` files
- put all public keys and same staking number to `genesis/validators.txt` file in the docker compose configuration
- put clients and nodes wallet to `genesis/wallet.txt` file in the docker compose configuration
- bind all configuration, genesis and wallets to the nodes in the docker compose configuration. Example for bootstrap node:
```yaml
      - ./data/$BOOTSTRAP_HOST:/var/lib/rnode/ # $BOOTSTRAP_HOST is defined in .env file
      - ./genesis/wallets.txt:/var/lib/rnode/genesis/wallets.txt 
      - ./genesis/bonds.txt:/var/lib/rnode/genesis/bonds.txt
      - ./conf/bootstrap.conf:/var/lib/rnode/rnode.conf 
      - ./conf/bootstrap.certificate.pem:/var/lib/rnode/node.certificate.pem # Node can regenerate cert if this line missed. Bootstrap address depends on a cert, validator won't be able to connect to the bootstrap node if cert is changed.
      - ./conf/bootstrap.key.pem:/var/lib/rnode/node.key.pem
      - ./conf/logback.xml:/var/lib/rnode/logback.xml
```
# Check status of each node:
CURL each node by GET /api/status
```shell
for port in 40403 40413 40423 40433 40443; do echo "Node: localhost:$port" && curl -s "http://localhost:${port}/api/status" | jq '.' && echo ; done
```
## Get blocks info of each node:
```shell
for port in 40403 40413 40423 40433 40443; do echo "Node: localhost:$port" && curl -s "http://localhost:${port}/api/blocks" |  jq '.[0] | {number: .blockNumber, faultTolerance: .faultTolerance, hash: .blockHash}' && echo ; done
```

## Check a fault tolerance of last finalized block of each node:
```shell
for port in 40403 40413 40423 40433 40443; do echo "Node: localhost:$port" && curl -s "http://localhost:${port}/api/last-finalized-block" | jq '{ number: .blockInfo.seqNum, faultTolerance: .blockInfo.faultTolerance }' && echo ; done
```