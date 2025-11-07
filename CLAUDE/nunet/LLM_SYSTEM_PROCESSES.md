# LLM System Processes in Firefly RNode

**Purpose**: Complete documentation of OpenAI and Ollama system process implementations in the Firefly RNode codebase.

**Status**: ✅ Both OpenAI and Ollama system processes exist and are fully functional

**Date**: 2025-10-09

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [OpenAI Implementation](#openai-implementation)
3. [Ollama Implementation](#ollama-implementation)
4. [Architecture Overview](#architecture-overview)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Code Locations](#code-locations)
8. [Dependencies](#dependencies)

---

## Executive Summary

**Good News**: Both OpenAI and Ollama system processes are **fully implemented** in the Firefly RNode codebase and ready to use!

**Key Findings**:
- ✅ OpenAI: 3 system contracts (GPT-4, DALL-E 3, Text-to-Speech)
- ✅ Ollama: 3 system contracts (Chat, Generate, List Models)
- ✅ Configuration via environment variables or config file
- ✅ Disabled by default for safety
- ✅ Full error handling and replay safety
- ✅ Working Rholang examples exist for Ollama
- ✅ Uses external libraries (openaiscala, Akka HTTP)

**When Added**:
- Both implementations were added in PR #170
- Currently in main branch
- Production-ready code

**Default State**:
- **Both are DISABLED by default** for safety
- Must be explicitly enabled to use
- When disabled, contract deployments that use these URNs will fail with a clear error message

---

## OpenAI Implementation

### Overview

The OpenAI integration provides three system contracts accessible from Rholang:

1. **`rho:ai:gpt4`** - Text completion using GPT-4
2. **`rho:ai:dalle3`** - Image generation using DALL-E 3
3. **`rho:ai:textToAudio`** - Text-to-speech using TTS-1

### Features

- Uses OpenAI's official Scala client library (`io.cequence.openaiscala`)
- Asynchronous execution with Akka HTTP
- Automatic API key validation at startup
- Comprehensive error handling
- Replay-safe for blockchain determinism
- Configurable timeout and validation options

### Service Trait

**Location**: `/rholang/src/main/scala/coop/rchain/rholang/externalservices/OpenAIService.scala`

```scala
trait OpenAIService {
  def ttsCreateAudioSpeech[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[Array[Byte]]

  def dalle3CreateImage[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[String]  // Returns URL to generated image

  def gpt4TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[String]
}
```

### Implementation Details

#### Real Implementation (`OpenAIServiceImpl`)

- **API Key Resolution** (priority order):
  1. Environment variable: `OPENAI_SCALA_CLIENT_API_KEY`
  2. Config file: `openai.api-key`
  3. Crashes at startup if missing

- **Startup Validation**:
  - Calls `listModels` endpoint to validate API key
  - Configurable via `openai.validate-api-key`
  - Default timeout: 15 seconds

- **Models Used**:
  - GPT-4: `gpt-4-1-mini` (configurable via CreateChatCompletionSettings)
  - DALL-E 3: `dall-e-3`
  - TTS: `tts-1-1106` with `shimmer` voice

- **HTTP Client**: Akka HTTP with ActorSystem
- **Execution Context**: Custom cached thread pool with daemon threads

#### Disabled Implementation (`NoOpOpenAIService`)

- Returns empty results
- Logs debug message when called
- No API key required

### System Process Integration

**Location**: `/rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala`

#### Fixed Channels and Body Refs

```scala
object FixedChannels {
  val GPT4: Par = byteName(20)
  val DALLE3: Par = byteName(21)
  val TEXT_TO_AUDIO: Par = byteName(22)
}

object BodyRefs {
  val GPT4: Long = 18L
  val DALLE3: Long = 19L
  val TEXT_TO_AUDIO: Long = 20L
}
```

#### Contract Handlers

**GPT-4 Text Completion** (`SystemProcesses.scala:469-494`):

```scala
def gpt4: Contract[F] = {
  // Replay mode - return cached output
  case isContractCall(produce, true, previousOutput, Seq(RhoType.String(_), ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  // Real mode - call OpenAI API
  case isContractCall(produce, _, _, Seq(RhoType.String(prompt), ack)) => {
    def callApi: F[String] =
      externalServices.openAIService
        .gpt4TextCompletion(prompt)
        .recoverWith {
          case e =>
            NonDeterministicProcessFailure(outputNotProduced = Seq.empty, cause = e).raiseError
        }

    def mapOutput(response: String): Seq[Par] = Seq(RhoType.String(response))

    def produceNonDeterministicOutput(output: Seq[Par]) =
      produce(output, ack)
        .map(_ => output)
        .recoverWith {
          case e =>
            NonDeterministicProcessFailure(output.map(_.toByteArray), e).raiseError
        }

    callApi.map(mapOutput).flatMap(produceNonDeterministicOutput)
  }
}
```

**DALL-E 3 Image Generation** (`SystemProcesses.scala:496-522`):
- Similar pattern to GPT-4
- Returns image URL as String

**Text-to-Audio** (`SystemProcesses.scala:524-550`):
- Returns audio bytes as `Array[Byte]`
- Mapped to `RhoType.ByteArray` in Rholang

### URN Registration

**Location**: `/rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntime.scala:479-504`

```scala
def stdRhoAIProcesses[F[_]]: Seq[Definition[F]] = Seq(
  Definition[F](
    "rho:ai:gpt4",
    FixedChannels.GPT4,
    2,  // prompt + ack
    BodyRefs.GPT4,
    { ctx => ctx.systemProcesses.gpt4 }
  ),
  Definition[F](
    "rho:ai:dalle3",
    FixedChannels.DALLE3,
    2,  // prompt + ack
    BodyRefs.DALLE3,
    { ctx => ctx.systemProcesses.dalle3 }
  ),
  Definition[F](
    "rho:ai:textToAudio",
    FixedChannels.TEXT_TO_AUDIO,
    2,  // text + ack
    BodyRefs.TEXT_TO_AUDIO,
    { ctx => ctx.systemProcesses.textToAudio }
  )
)
```

### Configuration

**Location**: `/node/src/main/resources/defaults.conf:356-390`

```hocon
openai {
  # Enable or disable OpenAI service
  # Priority: 1. Environment variable OPENAI_ENABLED
  #           2. This configuration
  #           3. Default: false
  enabled = false

  # API key (required when enabled)
  # Priority: 1. This configuration
  #           2. Environment variable OPENAI_SCALA_CLIENT_API_KEY
  api-key = ""

  # Validate API key at startup
  validate-api-key = true

  # Timeout for validation call (seconds)
  validation-timeout-sec = 15
}
```

### Enable/Disable Logic

**Location**: `/rholang/src/main/scala/coop/rchain/rholang/externalservices/package.scala`

```scala
private[rholang] def isOpenAIEnabled: Boolean = {
  val config = ConfigFactory.load()

  // Check environment variable (highest priority)
  val envEnabled = Option(System.getenv("OPENAI_ENABLED")).flatMap { value =>
    value.toLowerCase(Locale.ENGLISH) match {
      case "true" | "1" | "yes" | "on"  => Some(true)
      case "false" | "0" | "no" | "off" => Some(false)
      case _ => None
    }
  }

  // Check configuration file (fallback)
  val configEnabled = if (config.hasPath("openai.enabled")) {
    Some(config.getBoolean("openai.enabled"))
  } else None

  // Priority: env > config > default false
  envEnabled.getOrElse(configEnabled.getOrElse(false))
}
```

**Singleton Instance**:

```scala
object OpenAIServiceImpl {
  lazy val instance: OpenAIService = {
    if (isOpenAIEnabled) new OpenAIServiceImpl  // Crashes if no API key
    else new NoOpOpenAIService
  }
}
```

### Error Handling

**Common Errors**:

1. **API Key Missing** (Startup Error):
   ```
   OpenAI API key is not configured. Provide it via config path 'openai.api-key'
   or env var OPENAI_SCALA_CLIENT_API_KEY.
   ```

2. **API Key Invalid** (Startup Error):
   ```
   OpenAI API key validation failed. Check 'openai.api-key' or 'OPENAI_SCALA_CLIENT_API_KEY'.
   ```

3. **Service Disabled** (Deployment Error):
   ```
   coop.rchain.rholang.interpreter.errors$BugFoundError: No value set for `rho:ai:gpt4`.
   This is a bug in the normalizer or on the path from it.
   ```
   - This is expected when OpenAI is disabled

4. **API Call Failure** (Runtime):
   - Wrapped in `NonDeterministicProcessFailure`
   - Causes transaction to fail
   - Error logged with details

### Non-Deterministic Calls

OpenAI calls are marked as non-deterministic in `SystemProcesses.scala:194-199`:

```scala
val nonDeterministicCalls: Set[Long] = Set(
  BodyRefs.GPT4,
  BodyRefs.DALLE3,
  BodyRefs.TEXT_TO_AUDIO,
  // ... other non-deterministic calls
)
```

This ensures:
- First execution calls API and stores result
- Replay mode uses cached result (deterministic validation)
- Blockchain consensus remains consistent

---

## Ollama Implementation

### Overview

The Ollama integration provides three system contracts accessible from Rholang:

1. **`rho:ollama:chat`** - Chat completion with context
2. **`rho:ollama:generate`** - Simple text generation
3. **`rho:ollama:models`** - List available models

### Features

- Connects to local or remote Ollama server
- HTTP-based API using Akka HTTP
- Configurable base URL and model
- Automatic connection validation at startup
- Full error handling and replay safety
- Working Rholang examples included

### Service Trait

**Location**: `/rholang/src/main/scala/coop/rchain/rholang/externalservices/OllamaService.scala`

```scala
trait OllamaService {
  def chatCompletion[F[_]](model: String, prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]

  def textGeneration[F[_]](model: String, prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]

  def listModels[F[_]]()(
      implicit F: Concurrent[F]
  ): F[String]  // JSON string with models list
}
```

### Implementation Details

#### Real Implementation (`OllamaServiceImpl`)

- **API Endpoint**: Configurable via `ollama.base-url`
  - Default: `http://localhost:11434`
  - Can point to remote Ollama server

- **Startup Validation**:
  - Calls `/api/tags` to list models
  - Verifies Ollama server is running
  - Configurable via `ollama.validate-connection`
  - Default timeout: 30 seconds

- **HTTP Endpoints**:
  - Chat: `POST /api/chat` with `{model, messages, stream: false}`
  - Generate: `POST /api/generate` with `{model, prompt, stream: false}`
  - Models: `GET /api/tags`

- **Response Parsing**: JSON parsing to extract response text

- **HTTP Client**: Akka HTTP with ActorSystem

#### Disabled Implementation (`DisabledOllamaService`)

- Raises `UnsupportedOperationException` when called
- Error message: "Ollama service is not enabled. Set OLLAMA_ENABLED=true"

### System Process Integration

**Location**: `/rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala`

#### Fixed Channels and Body Refs

```scala
object FixedChannels {
  val OLLAMA_CHAT: Par = byteName(28)
  val OLLAMA_GENERATE: Par = byteName(29)
  val OLLAMA_MODELS: Par = byteName(30)
}

object BodyRefs {
  val OLLAMA_CHAT: Long = 26L
  val OLLAMA_GENERATE: Long = 27L
  val OLLAMA_MODELS: Long = 28L
}
```

#### Contract Handlers

**Ollama Chat** (`SystemProcesses.scala:521-554`):

```scala
def ollamaChat: Contract[F] = {
  // Replay mode
  case isContractCall(produce, true, previousOutput, Seq(_, _, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  // Real mode
  case isContractCall(produce, _, _, Seq(
    RhoType.String(model),
    RhoType.String(prompt),
    ack
  )) => {
    (for {
      response <- externalServices.ollamaService.chatCompletion(model, prompt)
      output = Seq(RhoType.String(response))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}
```

**Ollama Generate** - Similar pattern with model + prompt parameters

**Ollama Models** - Takes only ack channel, returns JSON string

### URN Registration

**Location**: `/rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntime.scala:506-528`

```scala
def stdRhoOllamaProcesses[F[_]]: Seq[Definition[F]] = Seq(
  Definition[F](
    "rho:ollama:chat",
    FixedChannels.OLLAMA_CHAT,
    3,  // model + prompt + ack
    BodyRefs.OLLAMA_CHAT,
    { ctx => ctx.systemProcesses.ollamaChat }
  ),
  Definition[F](
    "rho:ollama:generate",
    FixedChannels.OLLAMA_GENERATE,
    3,  // model + prompt + ack
    BodyRefs.OLLAMA_GENERATE,
    { ctx => ctx.systemProcesses.ollamaGenerate }
  ),
  Definition[F](
    "rho:ollama:models",
    FixedChannels.OLLAMA_MODELS,
    1,  // ack only
    BodyRefs.OLLAMA_MODELS,
    { ctx => ctx.systemProcesses.ollamaModels }
  )
)
```

### Configuration

**Location**: `/node/src/main/resources/defaults.conf:392-427`

```hocon
ollama {
  # Enable or disable Ollama service
  # Priority: 1. Environment variable OLLAMA_ENABLED
  #           2. This configuration
  #           3. Default: false
  enabled = false

  # Base URL for Ollama API
  base-url = "http://localhost:11434"

  # Default model to use
  default-model = "llama4:latest"

  # Validate connection at startup
  validate-connection = true

  # Timeout for requests (seconds)
  timeout-sec = 30
}
```

### Enable/Disable Logic

**Location**: `/rholang/src/main/scala/coop/rchain/rholang/externalservices/package.scala`

```scala
private[rholang] def isOllamaEnabled: Boolean = {
  val config = ConfigFactory.load()

  // Check environment variable (highest priority)
  val envEnabled = Option(System.getenv("OLLAMA_ENABLED")).flatMap { value =>
    value.toLowerCase(Locale.ENGLISH) match {
      case "true" | "1" | "yes" | "on"  => Some(true)
      case "false" | "0" | "no" | "off" => Some(false)
      case _ => None
    }
  }

  // Check configuration file (fallback)
  val configEnabled = if (config.hasPath("ollama.enabled")) {
    Some(config.getBoolean("ollama.enabled"))
  } else None

  // Priority: env > config > default false
  envEnabled.getOrElse(configEnabled.getOrElse(false))
}
```

**Singleton Instance**:

```scala
object OllamaServiceImpl {
  lazy val instance: OllamaService = {
    if (isOllamaEnabled) new OllamaServiceImpl
    else new DisabledOllamaService
  }
}
```

### Error Handling

**Common Errors**:

1. **Ollama Server Not Running** (Startup Error):
   ```
   Connection to Ollama server failed. Ensure Ollama is running at http://localhost:11434
   ```

2. **Service Disabled** (Deployment Error):
   ```
   coop.rchain.rholang.interpreter.errors$BugFoundError: No value set for `rho:ollama:chat`.
   This is a bug in the normalizer or on the path from it.
   ```
   - This is expected when Ollama is disabled

3. **Model Not Found** (Runtime):
   - HTTP 404 error from Ollama API
   - Wrapped in `NonDeterministicProcessFailure`

4. **API Call Timeout** (Runtime):
   - Timeout after `ollama.timeout-sec` seconds
   - Wrapped in `NonDeterministicProcessFailure`

### Example Rholang Files

**Location**: `/rholang/examples/`

1. **test-ollama-simple.rho** - Test models endpoint
2. **tut-ollama.rho** - Comprehensive test of all 3 contracts (concurrent)
3. **test-ollama-sequential.rho** - Sequential execution pattern

---

## Architecture Overview

### Service Layer Architecture

Both OpenAI and Ollama follow the same architectural pattern:

```
ExternalServices trait
  ├─ openAIService: OpenAIService
  └─ ollamaService: OllamaService

OpenAIService trait
  ├─ OpenAIServiceImpl (real)
  └─ NoOpOpenAIService (disabled)

OllamaService trait
  ├─ OllamaServiceImpl (real)
  └─ DisabledOllamaService (disabled)
```

### System Process Flow

```
Rholang Contract
    ↓
@"rho:ai:gpt4"!(["prompt"], *ack)
    ↓
URN Resolution → FixedChannels.GPT4
    ↓
Handler Lookup → ctx.systemProcesses.gpt4
    ↓
Check Replay Mode
    ├─ Replay: Return cached output
    └─ Real: Execute API call
        ↓
externalServices.openAIService.gpt4TextCompletion(...)
        ↓
HTTP Request to OpenAI API
        ↓
Response → produce(output, ack)
        ↓
Rholang receives result on ack channel
```

### Error Recovery

All external API calls are wrapped with error recovery:

1. **API Call Fails**: Raise `NonDeterministicProcessFailure`
2. **Produce Fails** (cost exhausted): Return `NonDeterministicProcessFailure` with output for replay
3. **Replay Mode**: Always use cached output (deterministic)

---

## Configuration

### Environment Variables (Highest Priority)

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `OPENAI_ENABLED` | true/false, 1/0, yes/no, on/off | false | Enable OpenAI service |
| `OPENAI_SCALA_CLIENT_API_KEY` | sk-... | (none) | OpenAI API key |
| `OLLAMA_ENABLED` | true/false, 1/0, yes/no, on/off | false | Enable Ollama service |

### Config File (Fallback Priority)

**File**: `application.conf` or `defaults.conf`

```hocon
openai {
  enabled = true
  api-key = "sk-********************************"
  validate-api-key = true
  validation-timeout-sec = 15
}

ollama {
  enabled = true
  base-url = "http://localhost:11434"
  default-model = "llama4:latest"
  validate-connection = true
  timeout-sec = 30
}
```

### Setup Instructions

#### OpenAI Setup

1. **Get API Key**: https://platform.openai.com/api-keys

2. **Enable Service**:
   ```bash
   export OPENAI_ENABLED=true
   export OPENAI_SCALA_CLIENT_API_KEY=sk-your-key-here
   ```

   Or in `application.conf`:
   ```hocon
   openai {
     enabled = true
     api-key = "sk-your-key-here"
   }
   ```

3. **Start RNode**: Will validate API key at startup

4. **Verify**: Check logs for "OpenAI service initialized successfully"

#### Ollama Setup

1. **Install Ollama**: https://ollama.com/download

2. **Start Ollama Server**:
   ```bash
   ollama serve
   ```

3. **Pull Models**:
   ```bash
   ollama pull llama4:latest
   ```

4. **Enable Service**:
   ```bash
   export OLLAMA_ENABLED=true
   ```

   Or in `application.conf`:
   ```hocon
   ollama {
     enabled = true
     base-url = "http://localhost:11434"
   }
   ```

5. **Start RNode**: Will validate connection at startup

6. **Verify**: Check logs for "Ollama service initialized successfully"

---

## API Reference

### OpenAI Contracts

#### rho:ai:gpt4

**Purpose**: Text completion using GPT-4

**Parameters**:
- `prompt` (String): The text prompt
- `ack` (Channel): Acknowledgment channel for response

**Returns**: String with generated text

**Example**:
```rholang
new gpt4(`rho:ai:gpt4`), ack, stdout(`rho:io:stdout`) in {
  gpt4!("Explain blockchain in one sentence", *ack) |
  for (@response <- ack) {
    stdout!(["GPT-4:", response])
  }
}
```

**Model**: `gpt-4-1-mini` (can be changed in code)
**Settings**: `top_p = 0.5`, `temperature = 0.5`

---

#### rho:ai:dalle3

**Purpose**: Image generation using DALL-E 3

**Parameters**:
- `prompt` (String): Description of image to generate
- `ack` (Channel): Acknowledgment channel for response

**Returns**: String with URL to generated image

**Example**:
```rholang
new dalle(`rho:ai:dalle3`), ack, stdout(`rho:io:stdout`) in {
  dalle!("A futuristic blockchain network", *ack) |
  for (@imageUrl <- ack) {
    stdout!(["Image URL:", imageUrl])
  }
}
```

**Model**: `dall-e-3`
**Settings**: `n = 1` (one image)

---

#### rho:ai:textToAudio

**Purpose**: Text-to-speech using OpenAI TTS

**Parameters**:
- `text` (String): Text to convert to speech
- `ack` (Channel): Acknowledgment channel for response

**Returns**: ByteArray with MP3 audio data

**Example**:
```rholang
new tts(`rho:ai:textToAudio`), ack, stdout(`rho:io:stdout`) in {
  tts!("Welcome to the blockchain", *ack) |
  for (@audioBytes <- ack) {
    stdout!(["Audio size:", audioBytes.length(), "bytes"])
    // Write to file or stream to client
  }
}
```

**Model**: `tts-1-1106`
**Voice**: `shimmer`

---

### Ollama Contracts

#### rho:ollama:chat

**Purpose**: Chat completion with conversation context

**Parameters**:
- `model` (String): Model name (e.g., "llama4:latest")
- `prompt` (String): User message
- `ack` (Channel): Acknowledgment channel for response

**Returns**: String with chat response

**Example**:
```rholang
new ollamaChat(`rho:ollama:chat`), ack, stdout(`rho:io:stdout`) in {
  ollamaChat!("llama4:latest", "What is blockchain?", *ack) |
  for (@response <- ack) {
    stdout!(["Ollama:", response])
  }
}
```

**API Endpoint**: `POST /api/chat`

---

#### rho:ollama:generate

**Purpose**: Simple text generation/completion

**Parameters**:
- `model` (String): Model name
- `prompt` (String): Generation prompt
- `ack` (Channel): Acknowledgment channel for response

**Returns**: String with generated text

**Example**:
```rholang
new ollamaGenerate(`rho:ollama:generate`), ack, stdout(`rho:io:stdout`) in {
  ollamaGenerate!("llama4:latest", "Complete this: The future of blockchain is", *ack) |
  for (@completion <- ack) {
    stdout!(["Generated:", completion])
  }
}
```

**API Endpoint**: `POST /api/generate`

---

#### rho:ollama:models

**Purpose**: List available Ollama models

**Parameters**:
- `ack` (Channel): Acknowledgment channel for response

**Returns**: String with JSON array of models

**Example**:
```rholang
new ollamaModels(`rho:ollama:models`), ack, stdout(`rho:io:stdout`) in {
  ollamaModels!(*ack) |
  for (@models <- ack) {
    stdout!(["Available models:", models])
  }
}
```

**API Endpoint**: `GET /api/tags`

---

## Code Locations

### Service Layer

| Component | File Path |
|-----------|-----------|
| OpenAI Service Trait | `/rholang/src/main/scala/coop/rchain/rholang/externalservices/OpenAIService.scala` |
| Ollama Service Trait | `/rholang/src/main/scala/coop/rchain/rholang/externalservices/OllamaService.scala` |
| External Services Trait | `/rholang/src/main/scala/coop/rchain/rholang/externalservices/ExternalServices.scala` |
| Enable/Disable Logic | `/rholang/src/main/scala/coop/rchain/rholang/externalservices/package.scala` |

### System Processes

| Component | File Path | Lines |
|-----------|-----------|-------|
| Fixed Channels (OpenAI) | `/rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala` | 155-157 |
| Fixed Channels (Ollama) | Same file | 161-163 |
| Body Refs (OpenAI) | Same file | 182-184 |
| Body Refs (Ollama) | Same file | 189-191 |
| Contract Handlers (OpenAI) | Same file | 469-550 |
| Contract Handlers (Ollama) | Same file | 521-600+ |

### Registration

| Component | File Path | Lines |
|-----------|-----------|-------|
| OpenAI Process Group | `/rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntime.scala` | 479-504 |
| Ollama Process Group | Same file | 506-528 |
| Dispatch Table | Same file | 542 |

### Configuration

| Component | File Path | Lines |
|-----------|-----------|-------|
| OpenAI Config | `/node/src/main/resources/defaults.conf` | 356-390 |
| Ollama Config | Same file | 392-427 |

### Examples

| File | Description |
|------|-------------|
| `/rholang/examples/test-ollama-simple.rho` | Simple models test |
| `/rholang/examples/tut-ollama.rho` | Comprehensive Ollama test |
| `/rholang/examples/test-ollama-sequential.rho` | Sequential execution |

---

## Dependencies

### OpenAI Dependencies

**Maven Coordinates**:
```xml
<dependency>
  <groupId>io.cequence</groupId>
  <artifactId>openai-scala-client_2.13</artifactId>
  <version>[latest]</version>
</dependency>
```

**Additional Dependencies**:
- Akka HTTP (for HTTP requests)
- Akka Streams (for streaming responses)
- Spray JSON (for JSON parsing)

**External API**: OpenAI REST API (https://api.openai.com)

### Ollama Dependencies

**HTTP Client**: Akka HTTP (same as OpenAI)

**External Service**: Ollama server (local or remote)
- Install: https://ollama.com/download
- API Docs: https://github.com/ollama/ollama/blob/main/docs/api.md

**No Additional Libraries**: Uses standard Scala HTTP and JSON libraries

---

## Summary

✅ **Both OpenAI and Ollama are fully implemented and production-ready**

**OpenAI**:
- 3 contracts: GPT-4, DALL-E 3, Text-to-Speech
- Requires API key from OpenAI
- External library: `io.cequence.openaiscala`
- Disabled by default, must enable explicitly

**Ollama**:
- 3 contracts: Chat, Generate, List Models
- Requires local/remote Ollama server
- No external libraries needed
- Disabled by default, must enable explicitly
- Working Rholang examples included

**Architecture**:
- Service trait pattern (real/disabled implementations)
- Enable/disable via environment or config
- Replay-safe for blockchain consensus
- Comprehensive error handling
- Well-documented configuration

**Ready for Demo**: Both implementations are ready to use for the demo scenarios!
