# Nunet Configuration Fix Required

## Problem

The Nunet service is not detecting when it's enabled, even when:
1. Environment variable `NUNET_ENABLED=true` is set
2. Configuration file has `nunet.enabled = true`

Error message:
```
Non deterministic process failed. Cause: Nunet service is not enabled.
Set NUNET_ENABLED=true or configure nunet.enabled=true in application.conf
```

## Root Cause

The issue is in the initialization order:

1. `RealExternalServices` is created with `NunetServiceImpl.instance`
2. `NunetServiceImpl.instance` calls `isNunetEnabled`
3. `isNunetEnabled` calls `ConfigFactory.load()`
4. **BUT** at this point, the node hasn't loaded the custom config file yet!

The rholang module loads the configuration **before** the node's configuration system has processed:
- Custom config files (like `nunet-test.conf`)
- Merged environment variables
- Command-line options

## Current Implementation

```scala
// In NunetServiceImpl
lazy val instance: NunetService = {
  import scala.concurrent.ExecutionContext.Implicits.global
  if (isNunetEnabled) new NunetServiceImpl
  else new DisabledNunetService
}

// In package.scala
private[rholang] def isNunetEnabled: Boolean = {
  val config = ConfigFactory.load()  // <-- PROBLEM: Loads too early!

  val envEnabled = Option(System.getenv("NUNET_ENABLED")).flatMap { ... }
  val configEnabled = if (config.hasPath("nunet.enabled")) {
    Some(config.getBoolean("nunet.enabled"))
  } else None

  envEnabled.getOrElse(configEnabled.getOrElse(false))
}
```

## Solution Options

### Option 1: Pass NodeConf to ExternalServices (Recommended)

This is how it should work - the node's configuration should be passed through to the external services.

**Changes needed:**

1. **Modify ExternalServices.scala:**
```scala
trait ExternalServices {
  def openAIService: OpenAIService
  def grpcClient: GrpcClientService
  def ollamaService: OllamaService
  def nunetService: NunetService
}

// Add factory method that takes config
object ExternalServices {
  def forNodeType(isValidator: Boolean, nodeConf: NodeConf): ExternalServices = {
    if (isValidator) new RealExternalServices(nodeConf)
    else new ObserverExternalServices(nodeConf)
  }
}

class RealExternalServices(nodeConf: NodeConf) extends ExternalServices {
  override val nunetService: NunetService = {
    if (nodeConf.nunet.exists(_.enabled)) {
      new NunetServiceImpl(nodeConf.nunet.get)
    } else {
      new DisabledNunetService
    }
  }
  // ... other services
}
```

2. **Modify NunetServiceImpl:**
```scala
class NunetServiceImpl(nunetConf: NunetConf)(implicit ec: ExecutionContext) extends NunetService {
  private val cliPath = nunetConf.cliPath
  private val context = nunetConf.context
  private val timeout = nunetConf.timeout

  // ... rest of implementation
}
```

3. **Update all call sites** that create ExternalServices to pass NodeConf

### Option 2: Use System Properties (Quick Fix)

Set Java system properties before the rholang module loads:

```scala
// In node Main.scala, BEFORE creating runtime
System.setProperty("nunet.enabled", nodeConf.nunet.map(_.enabled.toString).getOrElse("false"))
System.setProperty("nunet.cli-path", nodeConf.nunet.map(_.cliPath).getOrElse("nunet"))
// ... etc
```

Then update `isNunetEnabled` to check system properties first.

### Option 3: Delay Singleton Creation

Make the instance creation happen later, after config is loaded:

```scala
object NunetServiceImpl {
  @volatile private var _instance: NunetService = null

  def instance: NunetService = {
    if (_instance == null) {
      synchronized {
        if (_instance == null) {
          _instance = if (isNunetEnabled) new NunetServiceImpl
                     else new DisabledNunetService
        }
      }
    }
    _instance
  }
}
```

But this still has the timing issue with `Config Factory.load()`.

## Recommendation

**Use Option 1** - it's the cleanest architectural solution. The node's configuration system should be the source of truth, and external services should receive their configuration from the node, not try to load it themselves.

This matches how other systems work (the node loads config once, then passes it to all subsystems).

## Temporary Workaround

Until the proper fix is implemented, you can work around this by:

1. Setting environment variables **before** starting the JVM:
```bash
export NUNET_ENABLED=true
export NUNET_CLI_PATH=/path/to/nunet
./node/target/universal/stage/bin/rnode run --standalone
```

2. OR using system properties in the launch script:
```bash
./node/target/universal/stage/bin/rnode \
  -Dnunet.enabled=true \
  -Dnunet.cli-path=/path/to/nunet \
  run --standalone
```

The `-D` properties are loaded by `ConfigFactory.load()` and will work.

## Files to Modify (for Option 1)

1. `rholang/src/main/scala/coop/rchain/rholang/externalservices/ExternalServices.scala`
2. `rholang/src/main/scala/coop/rchain/rholang/externalservices/NunetService.scala`
3. `node/src/main/scala/coop/rchain/node/runtime/NodeRuntime.scala` (or wherever ExternalServices is created)
4. All test files that create ExternalServices

## Status

- ✅ Configuration model added (`NunetConf`)
- ✅ Configuration included in `NodeConf`
- ✅ Configuration added to `defaults.conf`
- ❌ **Configuration not being passed to NunetService** ← Current blocker
