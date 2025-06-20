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
  "result": "Success!\nDeployId is: 3044022013ccc7913ed7947e50cf595774931ad61e829c0699a8b302dabc42d42c8592bd02206f4a17e369cea9dc8357a788fab7d365eba3ab7cae4a8d101c77e27812625d98"
}
```

Cleanup.

```bash
clean() {
rm -rf ./casperbuffer/ ./dagstorage/ ./deploystorage/ ./blockstorage/ ./rnode.log ./rspace++/ ./node.certificate.pem ./node.key.pem ./eval/ ./rspace/
}

cd data/rnode.bootstrap
clean
cd ../rnode.validator1/
clean
cd ../rnode.validator2/
clean
cd ../rnode.validator3/
clean
cd ../..
docker compose -f shard.yml up -d
```
