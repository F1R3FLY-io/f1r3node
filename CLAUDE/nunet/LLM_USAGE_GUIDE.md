# LLM Usage Guide - Rholang Contracts

**Purpose**: Practical guide for using OpenAI and Ollama system processes from Rholang smart contracts.

**Audience**: Developers writing Rholang contracts that interact with LLMs

**Date**: 2025-10-09

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [OpenAI Usage](#openai-usage)
3. [Ollama Usage](#ollama-usage)
4. [Common Patterns](#common-patterns)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

**For OpenAI**:
1. Get API key from https://platform.openai.com/api-keys
2. Enable service:
   ```bash
   export OPENAI_ENABLED=true
   export OPENAI_SCALA_CLIENT_API_KEY=sk-your-key-here
   ```

**For Ollama**:
1. Install Ollama: https://ollama.com/download
2. Start server: `ollama serve`
3. Pull model: `ollama pull llama4:latest`
4. Enable service:
   ```bash
   export OLLAMA_ENABLED=true
   ```

### Your First LLM Contract

```rholang
// Simple GPT-4 example
new gpt4(`rho:ai:gpt4`), ack, stdout(`rho:io:stdout`) in {
  gpt4!("What is blockchain?", *ack) |
  for (@response <- ack) {
    stdout!(["GPT-4 says:", response])
  }
}
```

**Run it**:
```bash
rnode eval your-contract.rho
```

---

## OpenAI Usage

### Available Contracts

| Contract | Purpose | Input | Output |
|----------|---------|-------|--------|
| `rho:ai:gpt4` | Text generation | String | String |
| `rho:ai:dalle3` | Image generation | String | String (URL) |
| `rho:ai:textToAudio` | Text-to-speech | String | ByteArray (MP3) |

---

### GPT-4 Text Completion

**Contract**: `rho:ai:gpt4`

**Basic Usage**:
```rholang
new gpt4(`rho:ai:gpt4`), ack, stdout(`rho:io:stdout`) in {
  gpt4!("Explain smart contracts in one sentence", *ack) |
  for (@response <- ack) {
    stdout!(response)
  }
}
```

**Storing Response On-Chain**:
```rholang
new gpt4(`rho:ai:gpt4`), ack, storage in {
  gpt4!("Generate a blockchain fact", *ack) |
  for (@response <- ack) {
    // Store in contract state
    storage!(response) |

    // Also log it
    @"rho:io:stdout"!(["Stored:", response])
  }
}
```

**Multi-Step Generation**:
```rholang
new gpt4(`rho:ai:gpt4`), step1, step2, stdout(`rho:io:stdout`) in {
  // First generation
  gpt4!("Name a blockchain concept", *step1) |
  for (@concept <- step1) {
    stdout!(["Concept:", concept]) |

    // Use first response in second prompt
    new prompt in {
      prompt!("Explain " ++ concept ++ " in detail") |
      for (@detailPrompt <- prompt) {
        gpt4!(*detailPrompt, *step2) |
        for (@explanation <- step2) {
          stdout!(["Explanation:", explanation])
        }
      }
    }
  }
}
```

**Interactive Contract**:
```rholang
contract generateFact(@topic, return) = {
  new gpt4(`rho:ai:gpt4`), ack in {
    new prompt in {
      prompt!("Generate an interesting fact about " ++ topic) |
      for (@p <- prompt) {
        gpt4!(*p, *ack) |
        for (@fact <- ack) {
          return!(fact)
        }
      }
    }
  }
}

// Use the contract
new result, stdout(`rho:io:stdout`) in {
  generateFact!("blockchain", *result) |
  for (@fact <- result) {
    stdout!(["Fact:", fact])
  }
}
```

---

### DALL-E 3 Image Generation

**Contract**: `rho:ai:dalle3`

**Basic Usage**:
```rholang
new dalle(`rho:ai:dalle3`), ack, stdout(`rho:io:stdout`) in {
  dalle!("A futuristic blockchain network with glowing nodes", *ack) |
  for (@imageUrl <- ack) {
    stdout!(["Image URL:", imageUrl])
    // User can open URL in browser
  }
}
```

**Storing Image URL On-Chain**:
```rholang
new dalle(`rho:ai:dalle3`), ack, imageRegistry in {
  dalle!("A digital art NFT", *ack) |
  for (@url <- ack) {
    // Store in registry with metadata
    @"rho:registry:insert"!({
      "type": "generated-image",
      "url": url,
      "prompt": "A digital art NFT",
      "timestamp": "2025-10-09"
    }, Nil)
  }
}
```

**Batch Generation**:
```rholang
new dalle(`rho:ai:dalle3`), stdout(`rho:io:stdout`) in {
  new generateImages in {
    contract generateImages(@prompts, @index, @results) = {
      match prompts.length() {
        length /\ index < length => {
          new ack in {
            dalle!(prompts.nth(index), *ack) |
            for (@url <- ack) {
              stdout!(["Generated", index + 1, ":", url]) |
              generateImages!(prompts, index + 1, results ++ [url])
            }
          }
        }
        _ => {
          // All done
          stdout!(["All images generated:", results])
        }
      }
    } |

    // Generate 3 images
    generateImages!([
      "A blockchain node",
      "A smart contract",
      "A cryptocurrency wallet"
    ], 0, [])
  }
}
```

---

### Text-to-Audio

**Contract**: `rho:ai:textToAudio`

**Basic Usage**:
```rholang
new tts(`rho:ai:textToAudio`), ack, stdout(`rho:io:stdout`) in {
  tts!("Welcome to the decentralized future", *ack) |
  for (@audioBytes <- ack) {
    stdout!(["Audio generated:", audioBytes.length(), "bytes"])
    // audioBytes is a ByteArray containing MP3 data
  }
}
```

**Important**: The output is raw MP3 bytes. In a real application, you would:
1. Write bytes to a file
2. Stream to a client
3. Upload to storage (IPFS, etc.)
4. Store reference on-chain

**Pseudo-code for file writing** (requires additional system process):
```rholang
new tts(`rho:ai:textToAudio`), ack in {
  tts!("Hello world", *ack) |
  for (@audioBytes <- ack) {
    // Hypothetical file write process
    @"rho:io:writeFile"!("/tmp/audio.mp3", audioBytes, Nil)
  }
}
```

---

## Ollama Usage

### Available Contracts

| Contract | Purpose | Input | Output |
|----------|---------|-------|--------|
| `rho:ollama:chat` | Chat completion | Model + Message | String |
| `rho:ollama:generate` | Text generation | Model + Prompt | String |
| `rho:ollama:models` | List models | None | String (JSON) |

---

### Ollama Chat

**Contract**: `rho:ollama:chat`

**Basic Usage**:
```rholang
new ollamaChat(`rho:ollama:chat`), ack, stdout(`rho:io:stdout`) in {
  ollamaChat!("llama4:latest", "What is Rholang?", *ack) |
  for (@response <- ack) {
    stdout!(["Ollama:", response])
  }
}
```

**Different Models**:
```rholang
new ollamaChat(`rho:ollama:chat`), ack, stdout(`rho:io:stdout`) in {
  // Use llama4
  ollamaChat!("llama4:latest", "Explain consensus", *ack) |
  for (@response1 <- ack) {
    stdout!(["llama4:", response1])
  } |

  // Use mistral
  ollamaChat!("mistral:latest", "Explain consensus", *ack) |
  for (@response2 <- ack) {
    stdout!(["mistral:", response2])
  }
}
```

**Conversational Pattern** (Multi-turn):
```rholang
new ollamaChat(`rho:ollama:chat`), stdout(`rho:io:stdout`) in {
  new conversation in {
    // Turn 1
    ollamaChat!("llama4:latest", "What is blockchain?", *conversation) |
    for (@answer1 <- conversation) {
      stdout!(["User: What is blockchain?"]) |
      stdout!(["AI:", answer1]) |

      // Turn 2 - follow-up question
      ollamaChat!("llama4:latest", "How does it ensure security?", *conversation) |
      for (@answer2 <- conversation) {
        stdout!(["User: How does it ensure security?"]) |
        stdout!(["AI:", answer2]) |

        // Turn 3
        ollamaChat!("llama4:latest", "Give me an example", *conversation) |
        for (@answer3 <- conversation) {
          stdout!(["User: Give me an example"]) |
          stdout!(["AI:", answer3])
        }
      }
    }
  }
}
```

**Note**: Ollama chat doesn't maintain conversation history automatically. For true multi-turn conversations, you'd need to build the message history in the prompt.

---

### Ollama Generate

**Contract**: `rho:ollama:generate`

**Basic Usage**:
```rholang
new ollamaGenerate(`rho:ollama:generate`), ack, stdout(`rho:io:stdout`) in {
  ollamaGenerate!("llama4:latest", "Complete this: The future of blockchain is", *ack) |
  for (@completion <- ack) {
    stdout!(["Generated:", completion])
  }
}
```

**Code Generation**:
```rholang
new ollamaGenerate(`rho:ollama:generate`), ack, stdout(`rho:io:stdout`) in {
  ollamaGenerate!(
    "llama4:latest",
    "Write a Rholang contract that stores a message:\n",
    *ack
  ) |
  for (@code <- ack) {
    stdout!(["Generated Code:"]) |
    stdout!(code)
  }
}
```

**Structured Output**:
```rholang
new ollamaGenerate(`rho:ollama:generate`), ack, stdout(`rho:io:stdout`) in {
  ollamaGenerate!(
    "llama4:latest",
    "Generate a JSON object with blockchain properties (name, type, consensus):\n",
    *ack
  ) |
  for (@json <- ack) {
    // Parse and use JSON
    stdout!(["JSON:", json])
  }
}
```

---

### Ollama List Models

**Contract**: `rho:ollama:models`

**Basic Usage**:
```rholang
new ollamaModels(`rho:ollama:models`), ack, stdout(`rho:io:stdout`) in {
  ollamaModels!(*ack) |
  for (@models <- ack) {
    stdout!(["Available models:", models])
  }
}
```

**Output Format**: JSON string with model information:
```json
{
  "models": [
    {
      "name": "llama4:latest",
      "size": "7.4GB",
      "modified": "2025-10-08T10:30:00Z"
    },
    ...
  ]
}
```

**Check Model Availability**:
```rholang
new ollamaModels(`rho:ollama:models`), ack, ollamaChat(`rho:ollama:chat`),
    stdout(`rho:io:stdout`) in {

  // First check available models
  ollamaModels!(*ack) |
  for (@models <- ack) {
    stdout!(["Available models:", models]) |

    // Then use a specific model
    ollamaChat!("llama4:latest", "Hello", *ack) |
    for (@response <- ack) {
      stdout!(response)
    }
  }
}
```

---

## Common Patterns

### Pattern 1: Store LLM Output On-Chain

```rholang
contract storeGeneratedContent(@topic, return) = {
  new gpt4(`rho:ai:gpt4`), ack, storage in {
    new prompt in {
      prompt!("Generate content about " ++ topic) |
      for (@p <- prompt) {
        gpt4!(*p, *ack) |
        for (@content <- ack) {
          // Store with metadata
          storage!({
            "topic": topic,
            "content": content,
            "model": "gpt-4",
            "timestamp": "2025-10-09"
          }) |
          return!(content)
        }
      }
    }
  }
}
```

---

### Pattern 2: Compare Responses from Different Models

```rholang
new gpt4(`rho:ai:gpt4`), ollamaChat(`rho:ollama:chat`),
    gptAck, ollamaAck, stdout(`rho:io:stdout`) in {

  new prompt in {
    prompt!("Explain blockchain consensus") |
    for (@p <- prompt) {
      // Ask both models
      gpt4!(*p, *gptAck) |
      ollamaChat!("llama4:latest", *p, *ollamaAck) |

      // Wait for both responses
      for (@gptResponse <- gptAck; @ollamaResponse <- ollamaAck) {
        stdout!(["=== GPT-4 ===="]) |
        stdout!(gptResponse) |
        stdout!(["=== Ollama ==="]) |
        stdout!(ollamaResponse)
      }
    }
  }
}
```

---

### Pattern 3: Conditional Generation

```rholang
new ollamaChat(`rho:ollama:chat`), ack, stdout(`rho:io:stdout`) in {
  new checkTopic in {
    contract checkTopic(@topic) = {
      match topic {
        "blockchain" => {
          ollamaChat!("llama4:latest", "Explain blockchain", *ack) |
          for (@response <- ack) {
            stdout!(response)
          }
        }
        "smart-contracts" => {
          ollamaChat!("llama4:latest", "Explain smart contracts", *ack) |
          for (@response <- ack) {
            stdout!(response)
          }
        }
        _ => {
          stdout!(["Unknown topic:", topic])
        }
      }
    } |

    checkTopic!("blockchain")
  }
}
```

---

### Pattern 4: Batch Processing

```rholang
new ollamaChat(`rho:ollama:chat`), stdout(`rho:io:stdout`) in {
  new processBatch in {
    contract processBatch(@prompts, @index) = {
      match prompts.length() {
        length /\ index < length => {
          new ack in {
            ollamaChat!("llama4:latest", prompts.nth(index), *ack) |
            for (@response <- ack) {
              stdout!(["Response", index + 1, ":", response]) |
              processBatch!(prompts, index + 1)
            }
          }
        }
        _ => stdout!("Batch processing complete")
      }
    } |

    processBatch!([
      "What is blockchain?",
      "What are smart contracts?",
      "What is Rholang?"
    ], 0)
  }
}
```

---

### Pattern 5: Retry on Error

```rholang
new gpt4(`rho:ai:gpt4`), stdout(`rho:io:stdout`) in {
  new tryGenerate in {
    contract tryGenerate(@prompt, @maxRetries, @attempt) = {
      if (attempt > maxRetries) {
        stdout!(["Max retries reached"])
      } else {
        new ack in {
          gpt4!(prompt, *ack) |
          for (@response <- ack) {
            // Check if response is valid
            match response {
              "" => {
                // Empty response, retry
                stdout!(["Retry attempt", attempt]) |
                tryGenerate!(prompt, maxRetries, attempt + 1)
              }
              _ => {
                stdout!(["Success:", response])
              }
            }
          }
        }
      }
    } |

    tryGenerate!("Generate a fact", 3, 1)
  }
}
```

---

## Error Handling

### Common Errors

#### 1. Service Not Enabled

**Error Message**:
```
coop.rchain.rholang.interpreter.errors$BugFoundError: No value set for `rho:ai:gpt4`.
```

**Cause**: OpenAI/Ollama is disabled

**Solution**:
```bash
export OPENAI_ENABLED=true
# or
export OLLAMA_ENABLED=true
```

---

#### 2. API Key Missing (OpenAI)

**Error Message** (at startup):
```
OpenAI API key is not configured.
```

**Solution**:
```bash
export OPENAI_SCALA_CLIENT_API_KEY=sk-your-key-here
```

---

#### 3. Ollama Server Not Running

**Error Message** (at startup):
```
Connection to Ollama server failed.
```

**Solution**:
```bash
ollama serve
```

---

#### 4. Model Not Found (Ollama)

**Error**: Contract execution fails

**Cause**: Specified model not pulled

**Solution**:
```bash
ollama pull llama4:latest
```

---

#### 5. Rate Limiting (OpenAI)

**Error**: API call fails after some requests

**Cause**: OpenAI rate limits exceeded

**Solution**:
- Wait and retry
- Upgrade OpenAI plan
- Implement rate limiting in contract

---

### Error Handling in Contracts

**Graceful Degradation**:
```rholang
new gpt4(`rho:ai:gpt4`), ollamaChat(`rho:ollama:chat`),
    gptAck, ollamaAck, stdout(`rho:io:stdout`) in {

  // Try GPT-4 first
  gpt4!("What is blockchain?", *gptAck) |

  // Fallback to Ollama
  for (@response <- gptAck) {
    match response {
      "" => {
        // GPT-4 failed, try Ollama
        stdout!("GPT-4 unavailable, using Ollama") |
        ollamaChat!("llama4:latest", "What is blockchain?", *ollamaAck) |
        for (@ollamaResponse <- ollamaAck) {
          stdout!(ollamaResponse)
        }
      }
      _ => {
        stdout!(response)
      }
    }
  }
}
```

---

## Best Practices

### 1. Choose the Right Service

**Use OpenAI when**:
- Highest quality output required
- Image generation needed (DALL-E)
- Text-to-speech required
- Budget allows API costs

**Use Ollama when**:
- Running locally/in private network
- No internet required
- Free/unlimited usage
- Lower latency needed
- Privacy is critical

---

### 2. Prompt Engineering

**Be Specific**:
```rholang
// Bad
gpt4!("Tell me about blockchain", *ack)

// Good
gpt4!("Explain blockchain consensus in exactly one sentence, suitable for beginners", *ack)
```

**Provide Context**:
```rholang
new prompt in {
  prompt!(
    "You are a blockchain expert. " ++
    "Explain Proof of Stake consensus to a developer. " ++
    "Use technical terms but be concise."
  ) |
  for (@p <- prompt) {
    gpt4!(*p, *ack)
  }
}
```

---

### 3. Manage Costs (OpenAI)

**OpenAI charges per token**. Minimize costs by:

- Keep prompts concise
- Limit response length (can't control from Rholang, but keep prompts short)
- Cache responses on-chain
- Use Ollama for development/testing

**Cost Monitoring**:
```rholang
// Store call count
new callCount in {
  callCount!(0) |

  contract callGPT4(@prompt, return) = {
    for (@count <- callCount) {
      new ack in {
        callCount!(count + 1) |
        @"rho:io:stdout"!(["API calls:", count + 1]) |

        gpt4!(prompt, *ack) |
        for (@response <- ack) {
          return!(response)
        }
      }
    }
  }
}
```

---

### 4. Response Validation

**Always validate responses**:
```rholang
new gpt4(`rho:ai:gpt4`), ack, stdout(`rho:io:stdout`) in {
  gpt4!("Generate a number between 1 and 10", *ack) |
  for (@response <- ack) {
    // Validate response
    match response {
      "" => stdout!("Error: Empty response")
      _ => {
        // Check response is reasonable
        stdout!(["Valid response:", response])
      }
    }
  }
}
```

---

### 5. Determinism Warning

**Important**: LLM responses are **non-deterministic**!

- Same prompt → Different responses
- Handled via replay mode in system processes
- First execution calls API and stores result
- Replays use cached result

**Implication**: Don't rely on exact response text in contract logic

**Good**:
```rholang
// Store any response
gpt4!(prompt, *ack) |
for (@response <- ack) {
  storage!(response)  // OK - just storing
}
```

**Bad**:
```rholang
// Checking exact response
gpt4!(prompt, *ack) |
for (@response <- ack) {
  if (response == "Expected exact text") {  // BAD - won't match reliably
    doSomething!()
  }
}
```

---

### 6. Testing

**Test with Ollama First**:
- Ollama is free and local
- Iterate on prompts
- Once working, switch to OpenAI

**Test in isolation**:
```bash
# Test file: test-llm.rho
rnode eval test-llm.rho
```

**Mock for unit tests**: When testing contract logic, mock LLM responses

---

### 7. Timeout Considerations

**LLM calls can be slow** (3-30 seconds):

- OpenAI: Usually 3-10 seconds
- Ollama: Depends on model size and hardware (5-30 seconds)

**Don't block on LLM calls**:
```rholang
// Good - asynchronous
gpt4!(prompt, *ack) |
for (@response <- ack) {
  // Handle response when ready
} |
// Continue with other work
doOtherStuff!()
```

---

## Troubleshooting

### Issue: "No value set for rho:ai:gpt4"

**Symptom**: Contract deployment fails

**Diagnosis**: OpenAI is disabled

**Fix**:
```bash
export OPENAI_ENABLED=true
export OPENAI_SCALA_CLIENT_API_KEY=sk-...
# Restart RNode
```

---

### Issue: Empty Responses

**Symptom**: LLM returns empty string

**Possible Causes**:
1. API key invalid/expired (OpenAI)
2. Rate limit exceeded (OpenAI)
3. Model not loaded (Ollama)
4. Server error

**Diagnosis**:
- Check RNode logs
- Check OpenAI dashboard (usage/limits)
- Check Ollama logs: `ollama logs`

**Fix**:
- Verify API key
- Wait and retry
- Pull model: `ollama pull llama4:latest`

---

### Issue: Slow Responses

**Symptom**: Long wait times

**Causes**:
- Large model (Ollama)
- Slow hardware (Ollama)
- Network latency (OpenAI)
- Long prompt

**Fix**:
- Use smaller model: `llama4:7b` instead of `llama4:70b`
- Upgrade hardware
- Shorten prompts
- Use faster model: `gpt-4-1-mini` instead of `gpt-4`

---

### Issue: Ollama Connection Failed

**Symptom**: Startup fails with connection error

**Diagnosis**:
```bash
curl http://localhost:11434/api/tags
```

**Fix**:
```bash
# Start Ollama
ollama serve

# Or change base URL in config
ollama.base-url = "http://remote-server:11434"
```

---

### Issue: "Model not found" (Ollama)

**Symptom**: Contract fails when calling Ollama

**Diagnosis**:
```bash
ollama list
```

**Fix**:
```bash
ollama pull llama4:latest
```

---

### Issue: RNode Won't Start

**Symptom**: RNode crashes at startup

**Possible Causes**:
1. OpenAI API key validation failed
2. Ollama server not reachable
3. Network issues

**Fix**:
```hocon
# In application.conf, disable validation
openai {
  enabled = true
  api-key = "sk-..."
  validate-api-key = false  # Skip validation
}

ollama {
  enabled = true
  validate-connection = false  # Skip validation
}
```

**Or**: Fix the underlying issue (API key, Ollama server)

---

## Summary

**Available Contracts**:
- ✅ OpenAI: `rho:ai:gpt4`, `rho:ai:dalle3`, `rho:ai:textToAudio`
- ✅ Ollama: `rho:ollama:chat`, `rho:ollama:generate`, `rho:ollama:models`

**Setup**:
- Enable via environment variables or config
- OpenAI requires API key
- Ollama requires local server

**Common Patterns**:
- Store on-chain
- Multi-step generation
- Batch processing
- Error handling

**Best Practices**:
- Choose right service
- Engineer prompts carefully
- Validate responses
- Test with Ollama first
- Handle async gracefully

**Troubleshooting**:
- Check service enabled
- Verify API keys/server
- Review logs
- Test connection manually

Ready to build LLM-powered Rholang contracts! 🚀
