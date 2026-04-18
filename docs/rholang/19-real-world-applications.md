# Real-World Applications

How Rholang is used in production F1R3FLY applications.

## Embers: Template-Based Contract Generation

Embers is the F1R3FLY blockchain API. It uses a template rendering system to generate Rholang contracts at runtime from structured Rust data, then signs and deploys them.

### Architecture

```
Rust struct (typed data)
  → Askama/Jinja2 template (.rho file with {{ }} placeholders)
    → Rendered Rholang source string
      → Signed with deployer key
        → Deployed to blockchain via gRPC
```

### Template Example

A `.rho` template file with Jinja2 interpolation:

```rho
// templates/agents/create.rho
new rl(`rho:registry:lookup`), agentsCh,
    devNull(`rho:io:devNull`),
    stdout(`rho:io:stdout`) in {

  rl!({{ env_uri }}, *agentsCh) |
  for (@(_, agents) <- agentsCh) {
    @agents!("create",
      {{ id }},
      {{ version }},
      {{ created_at }},
      {{ name }},
      {{ description }},
      {{ code }},
      *devNull
    )
  }
}
```

The Rust side renders this with typed values:

```rust
#[derive(Debug, Clone, Render)]
#[template(path = "agents/create.rho")]
struct Create {
    env_uri: Uri,
    id: Uuid,
    version: Uuid,
    created_at: DateTime<Utc>,
    name: String,
    description: Option<String>,
    code: Option<String>,
}
```

### Value Type Mapping

The rendering system converts Rust types to Rholang literals:

| Rust Type | Rholang Output |
|-----------|---------------|
| `String` | `"hello"` |
| `i64` | `42` |
| `bool` | `true` / `false` |
| `Uuid` | `"550e8400-..."` (string) |
| `DateTime<Utc>` | `1712345678` (unix epoch) |
| `Option<T>` (Some) | rendered value |
| `Option<T>` (None) | `Nil` |
| `Vec<T>` | `[val1, val2, ...]` |
| `BTreeMap<String, T>` | `{"key1": val1, "key2": val2}` |
| `BTreeSet<T>` | `Set(val1, val2)` |
| `Uri` | `` `rho:id:...` `` |

### Template Inheritance

Embers uses a base template for signed registry inserts:

```rho
// templates/common/insert_signed.rho
new rs(`rho:registry:insertSigned:secp256k1`),
    rl(`rho:registry:lookup`),
    uriCh in {

  // Register the contract with signature
  rs!({{ public_key }},
      ({{ version }}, bundle+{*{% block name %}{% endblock %}}),
      {{ sig }},
      *uriCh
  ) |

  for (@uri <- uriCh) {
    {% block initialization %}{% endblock %}
  }
}
```

Service templates extend this:

```rho
{% extends "common/insert_signed.rho" %}
{%- block name -%} agents {%- endblock -%}
{%- block initialization -%}
  // Agent-specific initialization...
{%- endblock -%}
```

### Deploy Flow

Each Embers operation follows a consistent pattern:

1. **Render** -- populate template with typed Rust data
2. **Prepare** -- set phlo limit, valid-after-block, deployer key
3. **Sign** -- secp256k1 signature over deploy data
4. **Deploy** -- submit signed deploy via gRPC
5. **Propose** -- block includes the deploy (auto-propose in single-validator)
6. **Verify** -- check deploy result for errors

```rust
// Simplified from embers domain code
let contract = Create { env_uri, id, name, ... }.render()?;

let signed = prepare_for_signing()
    .code(contract)
    .phlo_limit(500_000)
    .valid_after_block_number(latest_block)
    .call()?;

let deploy_id = client.deploy_signed_contract(signed).await?;
```

### Data Model

Embers organizes on-chain state using multi-level TreeHashMaps (see [Design Patterns](14-design-patterns.md#multi-level-treehashmap)):

```
address → {
  resource_id → {
    version_id → { name, description, metadata, ... }
    "latest"   → { ... }  (latest pointer)
  }
}
```

This structure is used for agents, agent teams, and OSLFs (object storage).

### Service Contracts

Each Embers service (agents, wallets, etc.) deploys as a method-dispatch contract:

```rho
contract agents(@"create", @id, @version, @name, ..., ack) = { ... } |
contract agents(@"get", @id, @version, ret) = { ... } |
contract agents(@"list", @address, ret) = { ... } |
contract agents(@"delete", @id, ack) = { ... }
```

See [Design Patterns - Method Dispatch Object](14-design-patterns.md#method-dispatch-object).

### AI Agent Teams

The agent teams feature compiles a GraphQL node graph into Rholang contracts. Each node type maps to a system channel:

| Node Type | System Channel |
|-----------|---------------|
| TextModel | `rho:ai:gpt4` |
| TTIModel | `rho:ai:dalle3` |
| TTSModel | `rho:ai:textToAudio` |
| Compress | (data transformation) |
| Output | (result collection) |

Nodes are connected via channels based on the graph topology. A prompt enters the first node, flows through the graph, and the final output is stored on-chain keyed by deploy ID.

## Patterns for Building Applications

Based on Embers and the system contracts, production F1R3FLY applications typically:

1. **Use signed registry inserts** for deploying and updating contracts
2. **Organize state** in multi-level TreeHashMaps scoped by address
3. **Expose method-dispatch contracts** as the public API surface
4. **Use the Either monad** (`rho:lang:either`) for error handling in vault/transfer operations
5. **Track versions** with a "latest" pointer alongside version-specific entries
6. **Generate Rholang from application code** using templates rather than writing raw `.rho` files
7. **Sign and deploy** via the gRPC API with proper phlo limits and nonce management
