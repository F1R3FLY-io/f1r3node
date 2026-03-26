> Last updated: 2026-03-23

# Data Flow: Block Lifecycle

```
1. DEPLOY ARRIVAL
   Client -> DeployGrpcServiceV1::do_deploy(signed_deploy)
   -> Validate signature, format, phlogiston limits
   -> Store in KeyValueDeployStorage
   -> If autopropose: trigger proposal

2. BLOCK CREATION (Validator)
   ProposerInstance receives proposal request
   -> CasperSnapshot acquired (fails if finalization in progress)
   -> Constraint checks (bonded, enough new blocks, height limits)
   -> prepare_user_deploys(): select valid deploys from storage
   -> Execute each deploy via RhoRuntime (play runtime):
      - Rholang parse -> normalize -> reduce (with phlogiston budget)
      - RSpace produce/consume operations
      - create_soft_checkpoint() between deploys
   -> System deploys (slashing, close block)
   -> create_checkpoint() -> final state hash
   -> Assemble BlockMessage (header, body, justifications, bonds)
   -> Sign with validator Secp256k1 key
   -> Self-validate
   -> Insert into DAG

3. BLOCK PROPAGATION
   TransportLayer::broadcast(peers, Packet(BlockHashMessage))
   -> Peers request full block via BlockRequest
   -> TransportLayer::stream(peer, Blob(BlockMessage))

4. BLOCK PROCESSING (Receiver)
   BlockProcessorInstance receives block from P2P
   -> Check interest (shard, version, not old)
   -> Format + signature validation
   -> Dependency resolution (fetch missing parents)
   -> Full validation:
      - Replay all deploys via ReplayRSpace
      - Verify state hash matches
      - Check equivocations, justifications, bonds cache
   -> If valid: insert into DAG, record metrics
   -> If invalid: track equivocation record if applicable

5. FINALIZATION
   Finalizer (triggered periodically):
   -> Scope: blocks between LFB and tips
   -> Find candidates with >50% stake agreement
   -> Clique Oracle computes fault tolerance
   -> If exceeds threshold: mark as finalized
   -> Update LFB, clean up deploy storage
   -> Emit BlockFinalised event
   -> Background: proactive transfer extraction via CacheTransactionAPI
```

---

# Data Flow: Deploy Execution

```
DeployData (Rholang source + metadata)
    |
    | Compiler::source_to_adt()
    v
Par (normalized AST)
    |
    | DebruijnInterpreter::eval(par, env, rand)
    v
+-------------------------------------+
| Evaluation Loop                     |
|                                     |
|  Send  -> RSpace.produce(chan, data) |
|           If match: dispatch cont   |
|                                     |
|  For   -> RSpace.consume(chans,     |
|           patterns, cont)           |
|           If match: dispatch cont   |
|                                     |
|  New   -> Blake2b512Random.next()   |
|           Create unforgeable name   |
|           Eval body with new binding|
|                                     |
|  Match -> Eval target, try cases    |
|           SpatialMatcher matches    |
|                                     |
|  Expr  -> Evaluate arithmetic/logic |
|           Charge phlogiston         |
|                                     |
|  System-> Dispatch to Rust handler  |
|   Chan    (crypto, I/O, registry,   |
|            external services)       |
+-------------------------------------+
    |
    | ChargingRSpace wraps all ops with cost metering
    | OutOfPhlogistonsError if budget exhausted
    v
EvaluateResult { cost, errors, mergeable_channels }
    |
    | create_soft_checkpoint() after each deploy
    | create_checkpoint() at block boundary
    v
Checkpoint { root: Blake2b256Hash, log: Vec<Event> }
```

[<- Back to overview](./README.md)
