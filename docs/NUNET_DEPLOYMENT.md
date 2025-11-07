# Nunet DMS Integration - Deployment Guide

**Version**: 1.0
**Last Updated**: October 9, 2025
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [System Contracts Reference](#system-contracts-reference)
6. [Usage Examples](#usage-examples)
7. [Integration with DMS](#integration-with-dms)
8. [Troubleshooting](#troubleshooting)
9. [Performance and Limits](#performance-and-limits)
10. [Security Considerations](#security-considerations)

---

## Overview

The Nunet DMS integration provides 7 system contracts that enable Rholang smart contracts to orchestrate containerized deployments on the Nunet Distributed Management System (DMS).

### Key Features

- ✅ **Deploy Ensembles**: Deploy multi-container applications to Nunet DMS
- ✅ **Monitor Deployments**: Check status, retrieve logs, get manifests
- ✅ **YAML Generation**: Generate ensemble configurations programmatically
- ✅ **Validation**: Validate ensemble YAML before deployment
- ✅ **Full Integration**: Seamless integration with RNode system processes

### Use Cases

1. **Dynamic Shard Deployment**: Deploy RNode validator shards from Rholang contracts
2. **Decentralized Compute**: Orchestrate distributed computation tasks
3. **Service Mesh**: Deploy interconnected microservices with automatic networking
4. **CI/CD Automation**: Deploy applications triggered by on-chain events

---

## Prerequisites

### Required Software

1. **RNode** (Firefly distribution)
   - Version: Latest from `nunet-impl` branch or newer
   - Built with Nunet support

2. **Nunet DMS**
   - Installation: https://docs.nunet.io/installation
   - CLI tool: `nunet` command must be in PATH

3. **Operating System**
   - Linux (recommended)
   - macOS (supported)
   - Windows (via WSL2)

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 20GB available space
- **Network**: Internet connection for DMS communication
- **Permissions**: Ability to execute shell commands (for DMS CLI)

---

## Installation

### Step 1: Install Nunet DMS

Follow the official Nunet installation guide:

```bash
# Example installation (check official docs for latest)
curl -fsSL https://get.nunet.io | sh

# Verify installation
nunet --version
```

### Step 2: Start Nunet DMS

```bash
# Start DMS service
nunet start

# Verify it's running
nunet status
```

### Step 3: Build RNode with Nunet Support

```bash
cd /path/to/f1r3fly
git checkout nunet-impl  # or later branch with Nunet support

# Build the project
sbt "project node" assembly

# Or build Docker image
docker build -t f1r3fly-rnode:nunet .
```

### Step 4: Enable Nunet Contracts

Set the environment variable:

```bash
export NUNET_ENABLED=true
```

Or add to your RNode configuration file (`application.conf`):

```hocon
nunet {
  enabled = true
}
```

### Step 5: Start RNode

```bash
# If using binary
./rnode run --enable-nunet-contracts

# If using Docker
docker run -e NUNET_ENABLED=true f1r3fly-rnode:nunet
```

### Step 6: Verify Integration

Test with a simple Rholang contract:

```rholang
new list(`rho:nunet:deployment:list`), stdout(`rho:io:stdout`) in {
  list!(0) |
  for (@deployments <- @0) {
    stdout!(["Nunet integration working! Deployments:", deployments])
  }
}
```

Save as `test-nunet.rho` and run:

```bash
rnode eval test-nunet.rho
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUNET_ENABLED` | `false` | Enable/disable Nunet system contracts |
| `NUNET_DMS_ENDPOINT` | (none) | Override DMS endpoint URL |
| `NUNET_CLI_PATH` | `nunet` | Path to nunet CLI executable |

### Configuration File (application.conf)

```hocon
nunet {
  # Enable Nunet system contracts
  enabled = true

  # Path to nunet CLI (default: "nunet" from PATH)
  cli-path = "nunet"

  # Operation timeout in seconds
  timeout = 300

  # DMS configuration
  dms {
    # DMS endpoint (optional, CLI uses default)
    endpoint = "https://dms.nunet.io"

    # API key for authentication (if required)
    api-key = ""
  }

  # Resource limits
  limits {
    # Maximum concurrent deployments per node
    max-deployments-per-node = 10

    # Rate limiting: max deployments per minute
    max-deployments-per-minute = 5
  }
}
```

### Recommended Production Settings

```hocon
nunet {
  enabled = true
  timeout = 600  # 10 minutes for large deployments

  limits {
    max-deployments-per-node = 5  # Conservative limit
    max-deployments-per-minute = 2  # Prevent abuse
  }
}
```

---

## System Contracts Reference

### 1. rho:nunet:deployment:new

Deploy a new ensemble to Nunet DMS.

**Signature**:
```rholang
@"rho:nunet:deployment:new"!(ensembleYaml, timeoutMinutes, ackChannel)
```

**Parameters**:
- `ensembleYaml` (String): YAML definition of the ensemble
- `timeoutMinutes` (Int): Deployment timeout in minutes
- `ackChannel` (Channel): Channel to receive ensemble ID

**Returns**: String - Ensemble ID

**Example**:
```rholang
new deploy(`rho:nunet:deployment:new`) in {
  deploy!(
    "version: 1.0\nallocations:\n  - name: web\n    container:\n      image: nginx:latest",
    10,
    0
  ) |
  for (@ensembleId <- @0) {
    @"rho:io:stdout"!(["Deployed:", ensembleId])
  }
}
```

---

### 2. rho:nunet:deployment:status

Check the status of a deployed ensemble.

**Signature**:
```rholang
@"rho:nunet:deployment:status"!(ensembleId, ackChannel)
```

**Parameters**:
- `ensembleId` (String): The ensemble ID to query
- `ackChannel` (Channel): Channel to receive status

**Returns**: String - Status information (format: `"ensemble_id=...,status=...,allocations=..."`)

**Example**:
```rholang
new status(`rho:nunet:deployment:status`) in {
  status!("ensemble-abc123", 0) |
  for (@statusInfo <- @0) {
    @"rho:io:stdout"!(["Status:", statusInfo])
  }
}
```

---

### 3. rho:nunet:deployment:list

List all active deployments.

**Signature**:
```rholang
@"rho:nunet:deployment:list"!(ackChannel)
```

**Parameters**:
- `ackChannel` (Channel): Channel to receive deployment list

**Returns**: List - List of deployment summaries (format: `["ensemble-1:status:count", ...]`)

**Example**:
```rholang
new list(`rho:nunet:deployment:list`) in {
  list!(0) |
  for (@deployments <- @0) {
    @"rho:io:stdout"!(["Active deployments:", deployments])
  }
}
```

---

### 4. rho:nunet:deployment:logs

Retrieve logs from a deployed ensemble.

**Signature**:
```rholang
@"rho:nunet:deployment:logs"!(ensembleId, ackChannel)
```

**Parameters**:
- `ensembleId` (String): The ensemble ID to query
- `ackChannel` (Channel): Channel to receive logs

**Returns**: String - Log output

**Example**:
```rholang
new logs(`rho:nunet:deployment:logs`) in {
  logs!("ensemble-abc123", 0) |
  for (@logOutput <- @0) {
    @"rho:io:stdout"!(["Logs:", logOutput])
  }
}
```

---

### 5. rho:nunet:deployment:manifest

Get deployment manifest (peers, IPs, ports).

**Signature**:
```rholang
@"rho:nunet:deployment:manifest"!(ensembleId, ackChannel)
```

**Parameters**:
- `ensembleId` (String): The ensemble ID to query
- `ackChannel` (Channel): Channel to receive manifest

**Returns**: String - Manifest information (format: `"ensemble_id=...,peers=...,ips=...,ports=..."`)

**Example**:
```rholang
new manifest(`rho:nunet:deployment:manifest`) in {
  manifest!("ensemble-abc123", 0) |
  for (@manifestInfo <- @0) {
    @"rho:io:stdout"!(["Manifest:", manifestInfo])
  }
}
```

---

### 6. rho:nunet:ensemble:generate

Generate ensemble YAML from configuration.

**Signature**:
```rholang
@"rho:nunet:ensemble:generate"!(config, ackChannel)
```

**Parameters**:
- `config` (Map or Nil): Configuration map (currently accepts Nil)
- `ackChannel` (Channel): Channel to receive generated YAML

**Returns**: String - Generated ensemble YAML

**Example**:
```rholang
new generate(`rho:nunet:ensemble:generate`) in {
  generate!(Nil, 0) |
  for (@yaml <- @0) {
    @"rho:io:stdout"!(["Generated YAML:", yaml])
  }
}
```

**Note**: Config parameter parsing is planned for future enhancement.

---

### 7. rho:nunet:ensemble:validate

Validate ensemble YAML syntax and structure.

**Signature**:
```rholang
@"rho:nunet:ensemble:validate"!(ensembleYaml, ackChannel)
```

**Parameters**:
- `ensembleYaml` (String): YAML to validate
- `ackChannel` (Channel): Channel to receive validation result

**Returns**: String - Validation result (format: `"valid=true/false,errors=...,warnings=..."`)

**Example**:
```rholang
new validate(`rho:nunet:ensemble:validate`) in {
  validate!(
    "version: 1.0\nallocations:\n  - name: test",
    0
  ) |
  for (@result <- @0) {
    @"rho:io:stdout"!(["Validation:", result])
  }
}
```

---

## Usage Examples

### Basic Deployment Workflow

```rholang
new deploy(`rho:nunet:deployment:new`),
    status(`rho:nunet:deployment:status`),
    logs(`rho:nunet:deployment:logs`),
    stdout(`rho:io:stdout`) in {

  // Step 1: Deploy ensemble
  deploy!(
    "version: 1.0
allocations:
  - name: web-server
    container:
      image: nginx:alpine
      ports:
        - 80:80
",
    15,  // 15 minute timeout
    0
  ) |

  // Step 2: Wait for ensemble ID
  for (@ensembleId <- @0) {
    stdout!(["Deployed with ID:", ensembleId]) |

    // Step 3: Check status
    status!(ensembleId, 1) |
    for (@statusInfo <- @1) {
      stdout!(["Status:", statusInfo]) |

      // Step 4: Get logs
      logs!(ensembleId, 2) |
      for (@logOutput <- @2) {
        stdout!(["Logs:", logOutput])
      }
    }
  }
}
```

### Validate Before Deploy Pattern

```rholang
new validate(`rho:nunet:ensemble:validate`),
    deploy(`rho:nunet:deployment:new`),
    stdout(`rho:io:stdout`) in {

  // Define YAML
  @"yaml"!("version: 1.0\nallocations:\n  - name: app\n    container:\n      image: alpine") |

  // Validate first
  for (@yaml <- @"yaml") {
    validate!(yaml, 0) |

    for (@validationResult <- @0) {
      // Check if valid
      match validationResult {
        valid if valid.contains("valid=true") => {
          stdout!(["YAML valid, deploying..."]) |

          // Deploy if valid
          for (@yaml2 <- @"yaml") {
            deploy!(yaml2, 10, 1) |
            for (@ensembleId <- @1) {
              stdout!(["Deployed:", ensembleId])
            }
          }
        }
        _ => {
          stdout!(["YAML invalid:", validationResult])
        }
      }
    }
  }
}
```

### Monitor Multiple Deployments

```rholang
new list(`rho:nunet:deployment:list`),
    status(`rho:nunet:deployment:status`),
    stdout(`rho:io:stdout`) in {

  // List all deployments
  list!(0) |

  for (@deployments <- @0) {
    stdout!(["Found deployments:", deployments]) |

    // For each deployment, check status
    // (simplified - actual iteration would be more complex)
    stdout!(["Monitoring deployments..."])
  }
}
```

---

## Integration with DMS

### Ensemble YAML Format

The Nunet DMS uses YAML files to define "ensembles" (similar to docker-compose):

```yaml
version: 1.0

allocations:
  - name: bootstrap-node
    container:
      image: f1r3fly/rnode:latest
      command: ["rnode", "run", "--bootstrap"]
      ports:
        - 40400:40400  # gRPC
        - 40401:40401  # Kademlia
      env:
        RNODE_ROLE: bootstrap
      volumes:
        - source: ./configs/bootstrap.conf
          target: /var/lib/rnode/rnode.conf

  - name: validator-1
    container:
      image: f1r3fly/rnode:latest
      command: ["rnode", "run", "--validator"]
      ports:
        - 40410:40400
      env:
        RNODE_ROLE: validator
        BOOTSTRAP_HOST: bootstrap-node.internal

nodes:
  - peer_id: "QmLocalPeer"  # For local testing, pin to specific peer
```

### Network Communication

Nunet DMS automatically creates a private VPN for ensemble containers:

- **DNS**: Containers can reach each other via `<name>.internal`
- **Service Discovery**: Automatic DNS resolution within the ensemble
- **Isolation**: Each ensemble gets its own network namespace

### DMS CLI Commands (Behind the Scenes)

The system contracts invoke these DMS CLI commands:

```bash
# Deploy
nunet actor cmd --context user /dms/node/deployment/new -f ensemble.yml

# Status
nunet actor cmd --context user /dms/node/deployment/status --id <ensemble-id>

# Logs
nunet actor cmd --context user /dms/node/deployment/logs --id <ensemble-id>

# List
nunet actor cmd --context user /dms/node/deployment/list
```

---

## Troubleshooting

### Issue: "Nunet service not enabled"

**Symptoms**:
```
Error: Nunet service is not enabled. Set NUNET_ENABLED=true
```

**Solution**:
```bash
export NUNET_ENABLED=true
# Or set in application.conf: nunet.enabled = true
```

---

### Issue: "nunet CLI not found"

**Symptoms**:
```
Error: Cannot execute nunet command
```

**Solutions**:
1. Install Nunet DMS: https://docs.nunet.io/installation
2. Add nunet to PATH: `export PATH=$PATH:/path/to/nunet/bin`
3. Specify CLI path in config:
   ```hocon
   nunet.cli-path = "/usr/local/bin/nunet"
   ```

---

### Issue: "DMS connection refused"

**Symptoms**:
```
Error: Failed to connect to DMS endpoint
```

**Solutions**:
1. Verify DMS is running: `nunet status`
2. Start DMS if stopped: `nunet start`
3. Check endpoint configuration in `application.conf`
4. Verify network connectivity

---

### Issue: "Deployment timeout"

**Symptoms**:
```
Error: Deployment exceeded timeout of 10 minutes
```

**Solutions**:
1. Increase timeout parameter in contract: `deploy!(yaml, 30, ack)  // 30 min`
2. Increase global timeout in config:
   ```hocon
   nunet.timeout = 1800  // 30 minutes
   ```
3. Check if DMS has sufficient resources
4. Verify container images are available

---

### Issue: "Invalid ensemble YAML"

**Symptoms**:
```
Error: Failed to parse ensemble YAML
```

**Solutions**:
1. Use `rho:nunet:ensemble:validate` to check YAML before deployment
2. Verify YAML syntax (indentation, colons, dashes)
3. Check required fields (version, allocations)
4. Refer to Nunet ensemble format documentation

---

### Issue: "Rate limit exceeded"

**Symptoms**:
```
Error: Too many deployment requests
```

**Solutions**:
1. Wait before retrying
2. Adjust rate limits in config:
   ```hocon
   nunet.limits.max-deployments-per-minute = 10
   ```
3. Implement exponential backoff in contracts

---

### Issue: "Permission denied"

**Symptoms**:
```
Error: Permission denied executing nunet command
```

**Solutions**:
1. Check RNode has permission to execute shell commands
2. Verify `nunet` CLI has execute permissions: `chmod +x /path/to/nunet`
3. Check user has DMS access rights

---

## Performance and Limits

### Default Limits

| Limit | Default | Configurable |
|-------|---------|--------------|
| Max deployments per node | 10 | Yes |
| Max deployments per minute | 5 | Yes |
| Operation timeout | 300s (5min) | Yes |
| Max ensemble size | Unlimited | No (DMS limited) |

### Performance Characteristics

- **Deploy time**: 10-60 seconds depending on image size
- **Status check**: <1 second (cached by DMS)
- **Log retrieval**: 1-5 seconds
- **List operations**: <1 second

### Recommendations

1. **Production**: Set conservative rate limits
2. **Development**: Increase timeout for large images
3. **CI/CD**: Implement retry logic with exponential backoff
4. **Monitoring**: Poll status instead of logs (more efficient)

---

## Security Considerations

### Authentication

- **DMS API Key**: Store in environment variable, not in contracts
- **Configuration**: Use file-based config, not hardcoded values
- **Access Control**: Limit which contracts can call Nunet system processes

### Network Security

- **Private VPN**: Nunet ensembles use isolated networks
- **Firewall**: Configure appropriate port access
- **TLS**: Use HTTPS for DMS endpoint communication

### Resource Limits

- **DoS Prevention**: Rate limiting prevents abuse
- **Quota Management**: Max deployments per node prevents resource exhaustion
- **Timeout Enforcement**: Prevents hanging operations

### Best Practices

1. ✅ **Enable only when needed**: Keep `nunet.enabled = false` by default
2. ✅ **Use validation**: Always validate YAML before deployment
3. ✅ **Implement retry logic**: Handle transient failures gracefully
4. ✅ **Monitor deployments**: Track resource usage and costs
5. ✅ **Secure credentials**: Never hardcode API keys in contracts

---

## Further Resources

- [Nunet DMS Documentation](https://docs.nunet.io/)
- [Example Contracts](../examples/system-contract/nunet/)
- [Ensemble YAML Reference](https://docs.nunet.io/ensemble-format)
- [RNode System Processes Guide](./SYSTEM_PROCESSES.md)

---

## Support

For issues and questions:

- **GitHub Issues**: https://github.com/F1R3FLY-io/f1r3fly/issues
- **Nunet Discord**: https://discord.gg/nunet
- **Documentation**: https://docs.nunet.io/

---

**Last Updated**: October 9, 2025
**Maintainer**: F1R3FLY Industries
**License**: Apache 2.0
