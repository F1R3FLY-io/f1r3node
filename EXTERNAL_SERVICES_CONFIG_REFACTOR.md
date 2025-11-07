# External Services Configuration Refactor Plan

## Problem Statement

All external services (OpenAI, Ollama, Nunet) currently determine if they're enabled by calling `ConfigFactory.load()` directly in their singleton initialization. This happens **before** the node's configuration system has:

1. Loaded custom configuration files (via `-c` flag)
2. Merged environment variables
3. Applied command-line options

This causes the enabled flags to not be detected, resulting in errors like:
```
Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true
```

Even when the configuration IS properly set in the node's config.

## Root Cause Analysis

### Current Flow (Broken)

```
1. JVM starts
2. RealExternalServices object created
3. NunetServiceImpl.instance accessed
4. isNunetEnabled calls ConfigFactory.load()
   ├─ Loads default application.conf
   ├─ Loads defaults.conf from resources
   └─ Missing: Custom config files, processed env vars
5. Returns false (can't find nunet.enabled)
6. Creates DisabledNunetService
7. Later: Node loads actual configuration
8. Too late! Singleton already created
```

### Where Each Service Checks Configuration

**OpenAI** (`rholang/src/main/scala/coop/rchain/rholang/externalservices/OpenAIService.scala`):
```scala
object OpenAIServiceImpl {
  lazy val instance: OpenAIService = {
    if (isOpenAIEnabled) new OpenAIServiceImpl
    else noOpInstance
  }
}
```

**Ollama** (`rholang/src/main/scala/coop/rchain/rholang/externalservices/OllamaService.scala`):
```scala
object OllamaServiceImpl {
  lazy val instance: OllamaService = {
    import scala.concurrent.ExecutionContext.Implicits.global
    if (isOllamaEnabled) new OllamaServiceImpl
    else new DisabledOllamaService
  }
}
```

**Nunet** (`rholang/src/main/scala/coop/rchain/rholang/externalservices/NunetService.scala`):
```scala
object NunetServiceImpl {
  lazy val instance: NunetService = {
    import scala.concurrent.ExecutionContext.Implicits.global
    if (isNunetEnabled) new NunetServiceImpl
    else new DisabledNunetService
  }
}
```

All use the same broken pattern!

## Solution: Configuration Injection

Pass the node's `NodeConf` to external services so they use the configuration that has already been properly loaded and processed.

### Architecture Changes

```
┌─────────────────────────────────────────────────────────┐
│ Node Main                                               │
│  1. Load config (files + env + CLI)                     │
│  2. Build NodeConf                                      │
│  3. Create ExternalServices(nodeConf) ← PASS CONFIG     │
│  4. Create Runtime with ExternalServices                │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ ExternalServices                                        │
│  - Receives NodeConf                                    │
│  - Checks nodeConf.openai, nodeConf.ollama, etc.       │
│  - Creates enabled/disabled implementations             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Service Implementations                                 │
│  - Receive their specific config (OpenAIConf, etc.)    │
│  - No more ConfigFactory.load() calls                  │
│  - No more isXEnabled() checks                         │
└─────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Modify ExternalServices.scala

**File**: `rholang/src/main/scala/coop/rchain/rholang/externalservices/ExternalServices.scala`

```scala
package coop.rchain.rholang.externalservices

import coop.rchain.node.configuration.{NodeConf, OpenAIConf, OllamaConf, NunetConf}

trait ExternalServices {
  def openAIService: OpenAIService
  def grpcClient: GrpcClientService
  def ollamaService: OllamaService
  def nunetService: NunetService
}

/** Factory for creating ExternalServices with configuration */
object ExternalServices {

  /**
    * Create ExternalServices based on node type and configuration.
    *
    * @param isValidator true if node has validator keys, false for observer
    * @param nodeConf Node configuration with all service settings
    */
  def apply(isValidator: Boolean, nodeConf: NodeConf): ExternalServices = {
    if (isValidator)
      new RealExternalServices(nodeConf)
    else
      new ObserverExternalServices(nodeConf)
  }

  /** Legacy method for backwards compatibility */
  @deprecated("Use apply(isValidator, nodeConf) instead", "1.0.0")
  def forNodeType(isValidator: Boolean): ExternalServices = {
    // Fallback to old behavior - load config directly
    import com.typesafe.config.ConfigFactory
    val config = ConfigFactory.load()

    // This won't work properly but maintains compilation
    if (isValidator) RealExternalServices else ObserverExternalServices
  }
}

/** External services for validator nodes - all services enabled based on config */
class RealExternalServices(nodeConf: NodeConf) extends ExternalServices {
  import scala.concurrent.ExecutionContext.Implicits.global

  override val openAIService: OpenAIService = {
    nodeConf.openai match {
      case Some(conf) if conf.enabled => new OpenAIServiceImpl(conf)
      case _ => OpenAIServiceImpl.noOpInstance
    }
  }

  override val grpcClient: GrpcClientService = GrpcClientService.instance

  override val ollamaService: OllamaService = {
    nodeConf.ollama match {
      case Some(conf) if conf.enabled => new OllamaServiceImpl(conf)
      case _ => new DisabledOllamaService
    }
  }

  override val nunetService: NunetService = {
    nodeConf.nunet match {
      case Some(conf) if conf.enabled => new NunetServiceImpl(conf)
      case _ => new DisabledNunetService
    }
  }
}

/** External services for observer nodes - all services disabled */
class ObserverExternalServices(nodeConf: NodeConf) extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
  override val ollamaService: OllamaService  = new DisabledOllamaService
  override val nunetService: NunetService    = new DisabledNunetService
}

// Keep old singletons for backwards compatibility during transition
@deprecated("Use RealExternalServices(nodeConf) instead", "1.0.0")
case object RealExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.instance
  override val grpcClient: GrpcClientService = GrpcClientService.instance
  override val ollamaService: OllamaService  = OllamaServiceImpl.instance
  override val nunetService: NunetService    = NunetServiceImpl.instance
}

@deprecated("Use ObserverExternalServices(nodeConf) instead", "1.0.0")
case object NoOpExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
  override val ollamaService: OllamaService  = new DisabledOllamaService
  override val nunetService: NunetService    = new DisabledNunetService
}

@deprecated("Use ObserverExternalServices(nodeConf) instead", "1.0.0")
case object ObserverExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
  override val ollamaService: OllamaService  = new DisabledOllamaService
  override val nunetService: NunetService    = new DisabledNunetService
}
```

### Step 2: Update Service Implementations

**OpenAIService.scala**:
```scala
class OpenAIServiceImpl(config: OpenAIConf) extends OpenAIService {
  private val apiKey = config.apiKey
  // Use config directly, no ConfigFactory.load()

  // ... implementation
}
```

**OllamaService.scala**:
```scala
class OllamaServiceImpl(config: OllamaConf)(implicit ec: ExecutionContext) extends OllamaService {
  private val baseUrl = config.baseUrl
  private val defaultModel = config.defaultModel
  private val timeoutSec = config.timeoutSec

  // ... implementation
}
```

**NunetService.scala**:
```scala
class NunetServiceImpl(config: NunetConf)(implicit ec: ExecutionContext) extends NunetService {
  private val cliPath = config.cliPath
  private val context = config.context
  private val timeout = config.timeout

  // ... implementation
}
```

### Step 3: Update Node Runtime

**File**: `node/src/main/scala/coop/rchain/node/runtime/NodeRuntime.scala`

Find where `ExternalServices.forNodeType(...)` is called and update to:

```scala
val externalServices = ExternalServices(isValidator, nodeConf)
```

### Step 4: Update Tests

All test files that create `ExternalServices` need to be updated to pass a `NodeConf`.

**Example**:
```scala
// Old
val services = RealExternalServices

// New
val testNodeConf = NodeConf(
  // ... all required fields
  openai = Some(OpenAIConf(enabled = false, apiKey = "", validateApiKey = false, validationTimeoutSec = 15)),
  ollama = Some(OllamaConf(enabled = false, baseUrl = "", defaultModel = "", validateConnection = false, timeoutSec = 30)),
  nunet = Some(NunetConf(enabled = false, cliPath = "nunet", context = "user", timeout = 600)),
  // ... rest
)
val services = ExternalServices(isValidator = true, testNodeConf)
```

### Step 5: Remove Deprecated Code (Future)

After all call sites are updated:
- Remove singleton objects `RealExternalServices`, `NoOpExternalServices`, `ObserverExternalServices`
- Remove `isOpenAIEnabled`, `isOllamaEnabled`, `isNunetEnabled` from package.scala
- Remove `instance` lazy vals from service implementations

## Files to Modify

### Rholang Module
1. ✅ `rholang/src/main/scala/coop/rchain/rholang/externalservices/ExternalServices.scala`
2. ✅ `rholang/src/main/scala/coop/rchain/rholang/externalservices/OpenAIService.scala`
3. ✅ `rholang/src/main/scala/coop/rchain/rholang/externalservices/OllamaService.scala`
4. ✅ `rholang/src/main/scala/coop/rchain/rholang/externalservices/NunetService.scala`
5. ✅ `rholang/src/main/scala/coop/rchain/rholang/externalservices/package.scala` (optional: remove isXEnabled)

### Node Module
6. ✅ `node/src/main/scala/coop/rchain/node/runtime/NodeRuntime.scala`
7. ⚠️  `node/src/main/scala/coop/rchain/node/revvaultexport/StateBalances.scala`
8. ⚠️  `node/src/main/scala/coop/rchain/node/revvaultexport/mainnet1/reporting/MergeBalanceMain.scala`

### Tests (Find all with grep)
```bash
grep -r "RealExternalServices\|NoOpExternalServices\|ObserverExternalServices" --include="*.scala" rholang/src/test
grep -r "RealExternalServices\|NoOpExternalServices\|ObserverExternalServices" --include="*.scala" node/src/test
```

## Testing Strategy

1. **Compile tests**: Ensure all code compiles
2. **Unit tests**: Run existing test suites
3. **Integration test**: Start node with nunet enabled
4. **Manual test**: Execute `test-nunet-list.rho` contract

## Benefits

1. ✅ **Correct configuration loading**: Uses node's processed config
2. ✅ **Cleaner architecture**: Single source of truth for config
3. ✅ **Better testability**: Easy to inject test configurations
4. ✅ **No timing issues**: Config loaded before services created
5. ✅ **Consistent pattern**: All services configured the same way

## Migration Path

1. Implement new classes alongside old ones (both exist)
2. Update node runtime to use new pattern
3. Update tests incrementally
4. Remove deprecated code in separate commit

This ensures the codebase keeps working during migration.

## Next Steps

1. Implement ExternalServices changes
2. Update service implementations to take config in constructor
3. Update NodeRuntime to pass NodeConf
4. Find and update all test files
5. Test thoroughly
6. Remove deprecated code
