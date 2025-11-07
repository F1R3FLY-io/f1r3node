# Firefly System Contract Pattern

**Purpose**: This document describes how to implement system contracts in the Firefly RNode codebase, based on the patterns established in PR #170 (Ollama/OpenAI integration).

**Audience**: Developers implementing new system contracts for external service integration

**Date**: 2025-10-09

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Implementation Pattern](#implementation-pattern)
4. [Registration Flow](#registration-flow)
5. [Enable/Disable Pattern](#enabledisable-pattern)
6. [Error Handling](#error-handling)
7. [Code Organization](#code-organization)
8. [Complete Example](#complete-example)

---

## Overview

System contracts in Firefly allow Rholang code to interact with external services (HTTP APIs, shell commands, databases, etc.) through a standardized interface. They bridge the blockchain runtime with the outside world.

**Key Characteristics**:
- Fixed channels (bytes 0-30) for deterministic access
- URN-based addressing (e.g., `rho:ollama:chat`)
- Replay-safe execution (replay mode vs real mode)
- Configurable enable/disable via environment + config
- Type-safe parameter handling
- Deterministic error handling

**Source Files**:
- System process definitions: `rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala`
- Service implementations: `rholang/src/main/scala/coop/rchain/rholang/externalservices/`
- Registration: `rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntime.scala`

---

## Architecture Components

### 1. Definition[F]

The core abstraction for a system contract:

```scala
Definition[F](
  urn: String,           // How Rholang code accesses it (e.g., "rho:ollama:chat")
  fixedChannel: Par,     // Fixed channel for deterministic routing
  arity: Int,            // Number of parameters
  bodyRef: Long,         // Unique identifier for replay
  handler: Context => Contract[F]  // The actual implementation
)
```

**Example** (from `RhoRuntime.scala:477-483`):
```scala
Definition[F](
  "rho:ollama:chat",
  FixedChannels.OLLAMA_CHAT,
  3,
  BodyRefs.OLLAMA_CHAT,
  { ctx => ctx.systemProcesses.ollamaChat }
)
```

### 2. Fixed Channels

Pre-allocated channels (bytes 0-30) ensure deterministic routing:

**Location**: `SystemProcesses.scala:61-92`

```scala
object FixedChannels {
  def byteName(b: Byte): Par = GPrivate(ByteString.copyFrom(Array[Byte](b)))

  // Bytes 0-25: existing contracts (ed25519, sha256, keccak256, etc.)
  val OLLAMA_CHAT: Par = byteName(28)
  val OLLAMA_GENERATE: Par = byteName(29)
  val OLLAMA_MODELS: Par = byteName(30)
  // ... reserve bytes 31+ for future contracts
}
```

### 3. Body References

Unique identifiers for replay storage:

**Location**: `SystemProcesses.scala:94-124`

```scala
object BodyRefs {
  // 0-25: existing contracts
  val OLLAMA_CHAT: Long = 26L
  val OLLAMA_GENERATE: Long = 27L
  val OLLAMA_MODELS: Long = 28L
  // ... increment for new contracts
}
```

### 4. Contract Implementation

The handler function that processes calls:

**Location**: `SystemProcesses.scala:521-554`

```scala
def ollamaChat: Contract[F] = {
  // Replay mode - return cached output without executing
  case isContractCall(produce, true, previousOutput, Seq(_, _, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  // Real mode - execute external call
  case isContractCall(produce, _, _, Seq(RhoType.String(model), RhoType.String(prompt), ack)) => {
    (for {
      response <- externalServices.ollamaService.chatCompletion(model, prompt)
      output = Seq(RhoType.String(response))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(outputNotProduced = Seq.empty, cause = e).raiseError
    }
  }

  // Invalid arguments
  case _ =>
    Unit.pure[F].map(_ => Seq.empty)
}
```

---

## Implementation Pattern

### Step 1: Define Service Trait

Create a trait for your external service in `rholang/src/main/scala/coop/rchain/rholang/externalservices/`:

```scala
// YourService.scala
package coop.rchain.rholang.externalservices

import cats.effect.Concurrent

trait YourService {
  def yourMethod[F[_]](param1: String, param2: Int)(implicit F: Concurrent[F]): F[String]
}
```

### Step 2: Implement Service

Two implementations are required:

**Real Implementation** (does the actual work):

```scala
class YourServiceImpl extends YourService {
  def yourMethod[F[_]](param1: String, param2: Int)(implicit F: Concurrent[F]): F[String] = {
    // HTTP call, shell execution, etc.
    F.async { cb =>
      // Execute external operation
      // Call cb(Right(result)) or cb(Left(error))
    }
  }
}
```

**Disabled Implementation** (raises error when service is disabled):

```scala
class DisabledYourService extends YourService {
  def yourMethod[F[_]](param1: String, param2: Int)(implicit F: Concurrent[F]): F[String] = {
    F.raiseError(new UnsupportedOperationException(
      "YourService is not enabled. Set YOURSERVICE_ENABLED=true"
    ))
  }
}
```

### Step 3: Add Enable/Disable Logic

In `package.scala`:

```scala
private[rholang] def isYourServiceEnabled: Boolean = {
  val config = ConfigFactory.load()

  // Check environment variable (highest priority)
  val envEnabled = Option(System.getenv("YOURSERVICE_ENABLED")).flatMap { value =>
    value.toLowerCase(Locale.ENGLISH) match {
      case "true" | "1" | "yes" | "on"  => Some(true)
      case "false" | "0" | "no" | "off" => Some(false)
      case _ => None
    }
  }

  // Check configuration file (fallback)
  val configEnabled = if (config.hasPath("yourservice.enabled")) {
    Some(config.getBoolean("yourservice.enabled"))
  } else None

  // Priority: env > config > default false
  envEnabled.getOrElse(configEnabled.getOrElse(false))
}
```

Singleton instance with conditional instantiation:

```scala
object YourServiceImpl {
  lazy val instance: YourService = {
    if (isYourServiceEnabled) new YourServiceImpl
    else new DisabledYourService
  }
}
```

### Step 4: Add to ExternalServices Trait

In `ExternalServices.scala`:

```scala
trait ExternalServices {
  def openAIService: OpenAIService
  def grpcClient: GrpcClientService
  def ollamaService: OllamaService
  def yourService: YourService  // Add your service
}

class RealExternalServices extends ExternalServices {
  val openAIService: OpenAIService = OpenAIServiceImpl.instance
  val grpcClient: GrpcClientService = GrpcClientServiceImpl.instance
  val ollamaService: OllamaService = OllamaServiceImpl.instance
  val yourService: YourService = YourServiceImpl.instance  // Add your service
}

class NoOpExternalServices extends ExternalServices {
  val openAIService: OpenAIService = new DisabledOpenAIService
  val grpcClient: GrpcClientService = new DisabledGrpcClientService
  val ollamaService: OllamaService = new DisabledOllamaService
  val yourService: YourService = new DisabledYourService  // Add your service
}
```

### Step 5: Add Fixed Channel and Body Ref

In `SystemProcesses.scala`:

```scala
object FixedChannels {
  // ... existing channels
  val YOUR_METHOD: Par = byteName(31)  // Use next available byte
}

object BodyRefs {
  // ... existing refs
  val YOUR_METHOD: Long = 29L  // Use next available long
}
```

### Step 6: Implement System Process Handler

In `SystemProcesses.scala`:

```scala
def yourMethod: Contract[F] = {
  // Replay mode
  case isContractCall(produce, true, previousOutput, Seq(_, _, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  // Real mode
  case isContractCall(produce, _, _, Seq(
    RhoType.String(param1),
    RhoType.Number(param2Int),
    ack
  )) => {
    (for {
      result <- externalServices.yourService.yourMethod(param1, param2Int)
      output = Seq(RhoType.String(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  // Invalid arguments
  case _ => Unit.pure[F].map(_ => Seq.empty)
}
```

### Step 7: Register in RhoRuntime

In `RhoRuntime.scala`, create a new process group:

```scala
def stdRhoYourServiceProcesses[F[_]]: Seq[Definition[F]] = Seq(
  Definition[F](
    "rho:yourservice:method",
    FixedChannels.YOUR_METHOD,
    3,  // arity (param1, param2, ack)
    BodyRefs.YOUR_METHOD,
    { ctx => ctx.systemProcesses.yourMethod }
  )
)
```

Add to the registration sequence (line ~542):

```scala
(stdSystemProcesses[F] ++
 stdRhoCryptoProcesses[F] ++
 stdRhoAIProcesses[F] ++
 stdRhoOllamaProcesses[F] ++
 stdRhoYourServiceProcesses[F] ++  // Add your processes
 extraSystemProcesses)
```

---

## Registration Flow

1. **RhoRuntime initialization** (`RhoRuntime.scala:542`)
   - Sequences all `stdRhoXProcesses[F]` together
   - Creates dispatch table with all definitions

2. **Definition registration** (`RhoRuntime.scala:477-483`)
   - URN → Fixed Channel mapping
   - Fixed Channel → Handler function

3. **Contract call routing**
   - Rholang code sends on URN (e.g., `rho:ollama:chat`)
   - URN resolves to fixed channel
   - Fixed channel routes to handler
   - Handler executes and produces result

**Diagram**:
```
Rholang Code
    ↓
@"rho:ollama:chat"!(["model", "prompt", *ack])
    ↓
URN Resolution → FixedChannels.OLLAMA_CHAT
    ↓
Handler Lookup → ctx.systemProcesses.ollamaChat
    ↓
Contract Execution
    ↓ (replay=false)
externalServices.ollamaService.chatCompletion(...)
    ↓
HTTP Request to Ollama API
    ↓
Response → produce(output, ack)
    ↓
Rholang receives result on ack channel
```

---

## Enable/Disable Pattern

**Priority Order** (highest to lowest):
1. Environment variable (`YOURSERVICE_ENABLED`)
2. Configuration file (`yourservice.enabled`)
3. Default (`false`)

**Environment Variable Format** (`package.scala:34-43`):
- Truthy: `"true"`, `"1"`, `"yes"`, `"on"` (case-insensitive)
- Falsy: `"false"`, `"0"`, `"no"`, `"off"` (case-insensitive)
- Invalid: Ignored, falls through to config

**Configuration File** (HOCON format in `application.conf` or `defaults.conf`):
```hocon
yourservice {
  enabled = true
  # ... other settings
}
```

**Why This Pattern?**:
- Environment variables allow runtime control without rebuilding
- Configuration files provide deployment-specific defaults
- Default `false` is safe (prevents accidental external calls)
- Separation of concerns: service logic vs availability

**Example from Ollama** (`package.scala:34-51`):
```scala
private[rholang] def isOllamaEnabled: Boolean = {
  val config = ConfigFactory.load()

  val envEnabled = Option(System.getenv("OLLAMA_ENABLED")).flatMap { value =>
    value.toLowerCase(Locale.ENGLISH) match {
      case "true" | "1" | "yes" | "on"  => Some(true)
      case "false" | "0" | "no" | "off" => Some(false)
      case _ => None
    }
  }

  val configEnabled = if (config.hasPath("ollama.enabled")) {
    Some(config.getBoolean("ollama.enabled"))
  } else None

  envEnabled.getOrElse(configEnabled.getOrElse(false))
}
```

---

## Error Handling

### NonDeterministicProcessFailure

**Purpose**: Indicates an external operation failed during contract execution.

**Usage** (`SystemProcesses.scala:544-550`):
```scala
(for {
  result <- externalServices.yourService.yourMethod(...)
  output = Seq(RhoType.String(result))
  _ <- produce(output, ack)
} yield output).recoverWith {
  case e => NonDeterministicProcessFailure(
    outputNotProduced = Seq.empty,
    cause = e
  ).raiseError
}
```

**What Happens**:
- Error is logged
- Transaction may be retried or marked as failed
- Blockchain state remains consistent (no partial updates)

### Replay Mode

**Replay mode** (`replay = true`) is used when re-executing blocks:
- Skip external calls (non-deterministic)
- Use cached `previousOutput` instead
- Ensures deterministic block validation

**Implementation** (`SystemProcesses.scala:522-524`):
```scala
case isContractCall(produce, true, previousOutput, Seq(_, _, ack)) => {
  produce(previousOutput, ack).map(_ => previousOutput)
}
```

### Pattern Matching

Always provide a catch-all case for invalid arguments:

```scala
case _ => Unit.pure[F].map(_ => Seq.empty)
```

This prevents runtime errors from malformed Rholang calls.

---

## Code Organization

### File Structure

```
rholang/src/main/scala/coop/rchain/rholang/
├── externalservices/
│   ├── ExternalServices.scala       # Trait aggregating all services
│   ├── OllamaService.scala          # Example service (trait + impls)
│   ├── OpenAIService.scala          # Example service
│   ├── YourService.scala            # Your new service
│   └── package.scala                # Enable/disable logic
└── interpreter/
    ├── SystemProcesses.scala        # Contract handlers, channels, refs
    └── RhoRuntime.scala             # Registration and dispatch table
```

### Naming Conventions

| Component | Pattern | Example |
|-----------|---------|---------|
| URN | `rho:<service>:<method>` | `rho:ollama:chat` |
| Fixed Channel | `<SERVICE>_<METHOD>` | `OLLAMA_CHAT` |
| Body Ref | `<SERVICE>_<METHOD>` | `OLLAMA_CHAT` |
| Handler | `<method>` (camelCase) | `ollamaChat` |
| Service Trait | `<Service>Service` | `OllamaService` |
| Implementation | `<Service>ServiceImpl` | `OllamaServiceImpl` |
| Disabled Impl | `Disabled<Service>Service` | `DisabledOllamaService` |
| Env Variable | `<SERVICE>_ENABLED` | `OLLAMA_ENABLED` |
| Config Path | `<service>.enabled` | `ollama.enabled` |

### Best Practices

1. **One method = One contract**: Each service method gets its own Definition
2. **Sequential byte allocation**: Use next available byte for fixed channels
3. **Sequential long allocation**: Use next available long for body refs
4. **Lazy instance creation**: Use `lazy val instance` with enable check
5. **Type-safe parameters**: Use `RhoType.String`, `RhoType.Number`, etc.
6. **Explicit arity**: Match parameter count in Definition and pattern match
7. **Always handle replay**: First case should always be replay mode
8. **Always catch-all**: Last case should be `case _ => ...`

---

## Complete Example

Here's a minimal but complete system contract for a hypothetical "Weather API":

### 1. Service Definition (`WeatherService.scala`)

```scala
package coop.rchain.rholang.externalservices

import cats.effect.Concurrent
import cats.syntax.all._
import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import scala.concurrent.{ExecutionContext, Future}
import spray.json._

trait WeatherService {
  def getCurrentWeather[F[_]](city: String)(implicit F: Concurrent[F]): F[String]
}

class DisabledWeatherService extends WeatherService {
  def getCurrentWeather[F[_]](city: String)(implicit F: Concurrent[F]): F[String] = {
    F.raiseError(new UnsupportedOperationException(
      "Weather service is not enabled. Set WEATHER_ENABLED=true"
    ))
  }
}

class WeatherServiceImpl extends WeatherService {
  private implicit val system: ActorSystem = ActorSystem("weather-service")
  private implicit val ec: ExecutionContext = system.dispatcher

  def getCurrentWeather[F[_]](city: String)(implicit F: Concurrent[F]): F[String] = {
    val apiKey = sys.env.getOrElse("WEATHER_API_KEY", "")
    val url = s"https://api.weather.com/v1/current?city=$city&key=$apiKey"

    F.async { cb =>
      Http().singleRequest(HttpRequest(uri = url)).flatMap { response =>
        response.entity.toStrict(scala.concurrent.duration.FiniteDuration(5, "seconds"))
      }.map { entity =>
        val json = entity.data.utf8String.parseJson.asJsObject
        val temp = json.fields("temperature").toString
        cb(Right(s"Temperature in $city: $temp°C"))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }
}

object WeatherServiceImpl {
  lazy val instance: WeatherService = {
    if (isWeatherEnabled) new WeatherServiceImpl
    else new DisabledWeatherService
  }
}
```

### 2. Enable Logic (`package.scala`)

```scala
package coop.rchain.rholang

import com.typesafe.config.ConfigFactory
import java.util.Locale

package object externalservices {
  // ... existing functions

  private[rholang] def isWeatherEnabled: Boolean = {
    val config = ConfigFactory.load()

    val envEnabled = Option(System.getenv("WEATHER_ENABLED")).flatMap { value =>
      value.toLowerCase(Locale.ENGLISH) match {
        case "true" | "1" | "yes" | "on"  => Some(true)
        case "false" | "0" | "no" | "off" => Some(false)
        case _ => None
      }
    }

    val configEnabled = if (config.hasPath("weather.enabled")) {
      Some(config.getBoolean("weather.enabled"))
    } else None

    envEnabled.getOrElse(configEnabled.getOrElse(false))
  }
}
```

### 3. Add to ExternalServices (`ExternalServices.scala`)

```scala
trait ExternalServices {
  // ... existing services
  def weatherService: WeatherService
}

class RealExternalServices extends ExternalServices {
  // ... existing services
  val weatherService: WeatherService = WeatherServiceImpl.instance
}

class NoOpExternalServices extends ExternalServices {
  // ... existing services
  val weatherService: WeatherService = new DisabledWeatherService
}
```

### 4. System Process (`SystemProcesses.scala`)

```scala
// Add to FixedChannels object
object FixedChannels {
  // ... existing channels
  val WEATHER_CURRENT: Par = byteName(31)
}

// Add to BodyRefs object
object BodyRefs {
  // ... existing refs
  val WEATHER_CURRENT: Long = 29L
}

// Add handler method in SystemProcesses class
def weatherCurrent: Contract[F] = {
  // Replay mode
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  // Real mode
  case isContractCall(produce, _, _, Seq(RhoType.String(city), ack)) => {
    (for {
      weather <- externalServices.weatherService.getCurrentWeather(city)
      output = Seq(RhoType.String(weather))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  // Invalid arguments
  case _ => Unit.pure[F].map(_ => Seq.empty)
}
```

### 5. Registration (`RhoRuntime.scala`)

```scala
// Add new process group
def stdRhoWeatherProcesses[F[_]]: Seq[Definition[F]] = Seq(
  Definition[F](
    "rho:weather:current",
    FixedChannels.WEATHER_CURRENT,
    2,  // city + ack
    BodyRefs.WEATHER_CURRENT,
    { ctx => ctx.systemProcesses.weatherCurrent }
  )
)

// Add to registration sequence
(stdSystemProcesses[F] ++
 stdRhoCryptoProcesses[F] ++
 stdRhoAIProcesses[F] ++
 stdRhoOllamaProcesses[F] ++
 stdRhoWeatherProcesses[F] ++
 extraSystemProcesses)
```

### 6. Rholang Usage

```rholang
new ack in {
  @"rho:weather:current"!(["London", *ack]) |
  for (@result <- ack) {
    // result = "Temperature in London: 15°C"
    stdout!(result)
  }
}
```

### 7. Configuration

**Environment Variable**:
```bash
export WEATHER_ENABLED=true
export WEATHER_API_KEY=your-api-key
```

**Config File** (`defaults.conf`):
```hocon
weather {
  enabled = true
  api-key = "your-api-key"
}
```

---

## Summary

The Firefly system contract pattern provides a robust, type-safe way to integrate external services:

1. **Service layer**: Trait + Real/Disabled implementations
2. **Enable/disable**: Environment → Config → Default false
3. **Fixed routing**: Channels and body refs for determinism
4. **Replay safety**: Cached outputs during block validation
5. **Error handling**: NonDeterministicProcessFailure for failures
6. **Registration**: Definition[F] with URN, arity, handler

This pattern has been successfully used for OpenAI, Ollama, and gRPC integrations. It ensures external calls are controlled, deterministic, and properly isolated from the core consensus logic.

**Next Steps**:
- See `NUNET_CONTRACT_DESIGN.md` for applying this pattern to DMS
- See `IMPLEMENTATION_GUIDE.md` for step-by-step coding instructions
- See `examples/system-contract/` for working code templates
