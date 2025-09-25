# Ollama Integration for F1R3FLY

This document describes how to use the Ollama REST API integration with F1R3FLY's RChain blockchain platform.

## Overview

The Ollama integration allows Rholang smart contracts to interact with locally-running Large Language Models (LLMs) through Ollama's REST API. This integration follows the same pattern as the existing OpenAI integration but connects to your local Ollama instance instead of external services.

## Prerequisites

### 1. Install and Start Ollama

```bash
# Install Ollama if you haven't already
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, pull a model (like llama3.2)
ollama pull llama3.2
```

### 2. Build F1R3FLY with Ollama Support

```bash
# Build the project
sbt stage
```

## Configuration

### Environment Variable (Recommended)

```bash
# Enable Ollama integration
export OLLAMA_ENABLED=true

# Optional: Disable connection validation if needed
export OLLAMA_VALIDATE_CONNECTION=false
```

### Configuration File

Alternatively, you can configure via `application.conf`:

```hocon
ollama {
  enabled = true
  base-url = "http://localhost:11434"
  default-model = "llama3.2"
  validate-connection = true
  timeout-sec = 30
}
```

## Running F1R3FLY with Ollama

### Interactive Mode (REPL)

```bash
# Set environment variable to enable Ollama
export OLLAMA_ENABLED=true

# Run the staged rnode binary in interactive mode
./node/target/universal/stage/bin/rnode run --standalone --repl
```

### Production Mode

```bash
# Enable Ollama and run normally
export OLLAMA_ENABLED=true
./node/target/universal/stage/bin/rnode run --standalone
```

## Rholang Usage Examples

### Chat Completion

Use the `rho:ollama:chat` system process for conversational AI:

```rholang
new ollama, return in {
  ollama!("llama3.2", "What is 2+2?", *return)
}
```

You can also use the default model by omitting the model parameter:

```rholang
new ollama, return in {
  ollama!("What is the meaning of life?", *return)
}
```

### Text Generation

Use the `rho:ollama:generate` system process for text completion:

```rholang
new generate, return in {
  generate!("llama3.2", "Write a haiku about blockchain", *return)
}
```

With default model:

```rholang
new generate, return in {
  generate!("Complete this sentence: The future of decentralized computing", *return)
}
```

### List Available Models

Use the `rho:ollama:models` system process to see available models:

```rholang
new models, return in {
  models!(*return)
}
```

## System Processes Reference

The Ollama integration provides three system processes:

| Process | Channel | Parameters | Description |
|---------|---------|------------|-------------|
| Chat | `rho:ollama:chat` | `(model, prompt, ack)` or `(prompt, ack)` | Conversational AI completion |
| Generate | `rho:ollama:generate` | `(model, prompt, ack)` or `(prompt, ack)` | Text generation/completion |
| Models | `rho:ollama:models` | `(ack)` | List available models |

## Testing

### Unit Tests

Run the Ollama-specific tests:

```bash
sbt "rholang/testOnly *OllamaServiceSpec"
```

### Integration Testing

1. Start Ollama with a model
2. Enable Ollama in F1R3FLY
3. Run the REPL examples above
4. Verify responses are returned

## Troubleshooting

### Connection Issues

If you see connection errors:

1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check the model is available: `ollama list`
3. Disable connection validation: `export OLLAMA_VALIDATE_CONNECTION=false`

### Configuration Issues

- Ensure `OLLAMA_ENABLED=true` is set
- Check Ollama is running on the configured port (default: 11434)
- Verify the default model exists locally

### Performance Considerations

- Local LLM inference can be CPU/GPU intensive
- Response times depend on model size and hardware
- Consider using smaller models (e.g., `llama3.2:1b`) for faster responses

## Security Notes

- The Ollama integration only connects to localhost by default
- No external API keys or credentials are required
- All processing happens locally on your machine
- Review Ollama's security best practices for production deployments

## Architecture

The integration follows F1R3FLY's system process pattern:

1. **Service Layer**: `OllamaService` trait with implementations
2. **System Processes**: Fixed channels for Rholang interaction
3. **Configuration**: Environment/config driven enablement
4. **Runtime Integration**: Wired through the entire RhoRuntime stack

This design ensures consistency with other F1R3FLY integrations and maintains the platform's security and reliability standards.