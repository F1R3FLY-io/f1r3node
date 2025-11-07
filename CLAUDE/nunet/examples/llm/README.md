# LLM Examples for Firefly RNode

This directory contains working examples of using OpenAI and Ollama system processes from Rholang smart contracts.

## Files

| File | Description |
|------|-------------|
| `openai-example.rho` | Complete examples of all OpenAI contracts (GPT-4, DALL-E 3, TTS) |
| `ollama-example.rho` | Complete examples of all Ollama contracts (Chat, Generate, Models) |
| `demo-contract.rho` | Demo contract for Nunet-Rholang integration showcase |
| `test-llm.sh` | Automated test script for both integrations |
| `README.md` | This file |

## Quick Start

### Option 1: OpenAI

1. **Get API Key**: https://platform.openai.com/api-keys

2. **Enable Service**:
   ```bash
   export OPENAI_ENABLED=true
   export OPENAI_SCALA_CLIENT_API_KEY=sk-your-key-here
   ```

3. **Run Examples**:
   ```bash
   rnode eval openai-example.rho
   ```

### Option 2: Ollama (Recommended for Demo)

1. **Install Ollama**: https://ollama.com/download

2. **Start Server**:
   ```bash
   ollama serve
   ```

3. **Pull Model**:
   ```bash
   ollama pull llama4:latest
   ```

4. **Enable Service**:
   ```bash
   export OLLAMA_ENABLED=true
   ```

5. **Run Examples**:
   ```bash
   rnode eval ollama-example.rho
   ```

## Test Script

The `test-llm.sh` script automates testing and provides clear feedback:

```bash
# Test both OpenAI and Ollama
./test-llm.sh both

# Test only OpenAI
./test-llm.sh openai

# Test only Ollama
./test-llm.sh ollama

# Test demo contract
./test-llm.sh demo
```

The script will:
- Check prerequisites
- Verify services are enabled
- Run example contracts
- Report success/failure

## Available Contracts

### OpenAI

| Contract | Purpose | Input | Output |
|----------|---------|-------|--------|
| `rho:ai:gpt4` | Text completion | String (prompt) | String |
| `rho:ai:dalle3` | Image generation | String (prompt) | String (URL) |
| `rho:ai:textToAudio` | Text-to-speech | String (text) | ByteArray (MP3) |

### Ollama

| Contract | Purpose | Input | Output |
|----------|---------|-------|--------|
| `rho:ollama:chat` | Chat completion | Model + Message | String |
| `rho:ollama:generate` | Text generation | Model + Prompt | String |
| `rho:ollama:models` | List models | None | String (JSON) |

## Example Usage

### Simple GPT-4 Call

```rholang
new gpt4(`rho:ai:gpt4`), ack, stdout(`rho:io:stdout`) in {
  gpt4!("What is blockchain?", *ack) |
  for (@response <- ack) {
    stdout!(response)
  }
}
```

### Simple Ollama Call

```rholang
new ollamaChat(`rho:ollama:chat`), ack, stdout(`rho:io:stdout`) in {
  ollamaChat!("llama4:latest", "What is blockchain?", *ack) |
  for (@response <- ack) {
    stdout!(response)
  }
}
```

### Store Generated Content

```rholang
new gpt4(`rho:ai:gpt4`), ack, storage in {
  gpt4!("Generate a blockchain fact", *ack) |
  for (@fact <- ack) {
    storage!({
      "content": fact,
      "timestamp": "2025-10-09"
    })
  }
}
```

## Demo Contract

The `demo-contract.rho` file showcases:

1. **Generate and Store**: Create content and store on-chain
2. **Batch Processing**: Generate multiple responses sequentially
3. **Fallback Pattern**: Try OpenAI, fallback to Ollama
4. **Comparison**: Compare OpenAI vs Ollama responses
5. **Interactive Query**: Question-answer contract
6. **Conditional Generation**: Different prompts based on input

**Run it**:
```bash
export OLLAMA_ENABLED=true
rnode eval demo-contract.rho
```

## Troubleshooting

### "No value set for rho:ai:gpt4"

**Cause**: Service not enabled

**Fix**:
```bash
export OPENAI_ENABLED=true
```

### "OpenAI API key is not configured"

**Cause**: API key not set

**Fix**:
```bash
export OPENAI_SCALA_CLIENT_API_KEY=sk-your-key-here
```

### "Connection to Ollama server failed"

**Cause**: Ollama not running

**Fix**:
```bash
ollama serve
```

### "Model not found" (Ollama)

**Cause**: Model not pulled

**Fix**:
```bash
ollama pull llama4:latest
```

### Empty Responses

**Possible Causes**:
- API key invalid (OpenAI)
- Rate limit exceeded (OpenAI)
- Server error
- Model not loaded (Ollama)

**Diagnosis**:
- Check RNode logs
- Check OpenAI dashboard
- Check Ollama logs: `ollama logs`

## Integration with Nunet Demo

For the Nunet-Rholang demo, the recommended approach is:

1. **Use Ollama** (no API key needed, runs locally)
2. **Deploy demo-contract.rho** on the bonded shard
3. **Show various scenarios**:
   - Generate blockchain content
   - Store on-chain
   - Query and retrieve

**Demo Flow**:
```bash
# 1. Start Ollama
ollama serve

# 2. Pull model
ollama pull llama4:latest

# 3. Enable in RNode
export OLLAMA_ENABLED=true

# 4. Start RNode with bonded shard
rnode run ...

# 5. Deploy demo contract
rnode deploy demo-contract.rho

# 6. Show generated content
rnode eval show-results.rho
```

## Best Practices

1. **Use Ollama for Development**: Free, local, no API keys
2. **Switch to OpenAI for Production**: If higher quality needed
3. **Keep Prompts Concise**: Faster responses, lower costs
4. **Validate Responses**: Always check for empty/invalid responses
5. **Handle Errors Gracefully**: Implement fallbacks
6. **Test Locally First**: Use test script before deploying

## Cost Considerations

### OpenAI
- Charges per token (~$0.01-$0.03 per 1K tokens)
- Monitor usage on dashboard
- Use for final demo if budget allows

### Ollama
- Free (runs locally)
- Requires good hardware (4GB+ RAM for small models)
- Perfect for development and demos

## Performance

### OpenAI
- Typical response time: 3-10 seconds
- Depends on network and OpenAI load
- Usually consistent

### Ollama
- Depends on model size and hardware
- Small models (7B): 5-15 seconds
- Large models (70B): 30+ seconds
- First call slower (model loading)

## Next Steps

1. **Try Examples**: Run `./test-llm.sh` to see everything working
2. **Read Docs**: See `LLM_SYSTEM_PROCESSES.md` for complete documentation
3. **Usage Guide**: See `LLM_USAGE_GUIDE.md` for patterns and best practices
4. **Customize**: Modify `demo-contract.rho` for your use case
5. **Deploy**: Integrate with Nunet bonded shard deployment

## Support

For issues or questions:
- Check `LLM_SYSTEM_PROCESSES.md` for implementation details
- Check `LLM_USAGE_GUIDE.md` for usage patterns
- Review RNode logs for errors
- Test with `./test-llm.sh` for diagnosis

## License

These examples are part of the Firefly RNode project.
