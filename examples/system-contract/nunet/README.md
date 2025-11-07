# Nunet DMS System Contract Examples

This directory contains example Rholang contracts demonstrating the use of Nunet DMS system contracts.

## Prerequisites

1. **Enable Nunet contracts**:
   ```bash
   export NUNET_ENABLED=true
   ```
   Or in `application.conf`:
   ```hocon
   nunet.enabled = true
   ```

2. **Install Nunet DMS**: See [Nunet documentation](https://docs.nunet.io/)

3. **RNode running**: Ensure your RNode instance is running with Nunet support enabled

## Examples

### 01-simple-deploy.rho
Basic deployment of a simple "Hello World" container to Nunet DMS.

**Usage**:
```bash
rnode eval examples/system-contract/nunet/01-simple-deploy.rho
```

### 02-check-status.rho
Check the status of a deployed ensemble.

**Usage**:
```bash
# Edit the file to replace YOUR_ENSEMBLE_ID with actual ID
rnode eval examples/system-contract/nunet/02-check-status.rho
```

### 03-list-deployments.rho
List all active deployments on the Nunet DMS.

**Usage**:
```bash
rnode eval examples/system-contract/nunet/03-list-deployments.rho
```

### 04-get-logs.rho
Retrieve logs from a deployed ensemble.

**Usage**:
```bash
# Edit the file to replace YOUR_ENSEMBLE_ID with actual ID
rnode eval examples/system-contract/nunet/04-get-logs.rho
```

### 05-get-manifest.rho
Get deployment manifest containing peer IDs, IP addresses, and port mappings.

**Usage**:
```bash
# Edit the file to replace YOUR_ENSEMBLE_ID with actual ID
rnode eval examples/system-contract/nunet/05-get-manifest.rho
```

### 06-generate-ensemble.rho
Generate ensemble YAML from configuration (currently uses default config).

**Usage**:
```bash
rnode eval examples/system-contract/nunet/06-generate-ensemble.rho
```

### 07-validate-ensemble.rho
Validate ensemble YAML syntax and structure.

**Usage**:
```bash
rnode eval examples/system-contract/nunet/07-validate-ensemble.rho
```

## Available System Contracts

| URN | Parameters | Description |
|-----|-----------|-------------|
| `rho:nunet:deployment:new` | ensembleYaml, timeoutMinutes, ack | Deploy new ensemble |
| `rho:nunet:deployment:status` | ensembleId, ack | Check deployment status |
| `rho:nunet:deployment:list` | ack | List all deployments |
| `rho:nunet:deployment:logs` | ensembleId, ack | Get deployment logs |
| `rho:nunet:deployment:manifest` | ensembleId, ack | Get deployment manifest |
| `rho:nunet:ensemble:generate` | config, ack | Generate ensemble YAML |
| `rho:nunet:ensemble:validate` | ensembleYaml, ack | Validate ensemble YAML |

## Testing with Mock DMS

For testing without a real Nunet DMS installation, use the mock service:

```scala
import coop.rchain.rholang.externalservices.NunetServiceMock

// In test code:
val mockService = NunetServiceMock.mockService
```

## Common Patterns

### Deploy and Monitor Pattern
```rholang
new deploy(`rho:nunet:deployment:new`),
    status(`rho:nunet:deployment:status`),
    stdout(`rho:io:stdout`) in {

  // Deploy
  deploy!(yaml, 10, 0) |

  // Get ensemble ID
  for (@ensembleId <- @0) {
    stdout!(["Deployed:", ensembleId]) |

    // Check status
    status!(ensembleId, 1) |
    for (@statusInfo <- @1) {
      stdout!(["Status:", statusInfo])
    }
  }
}
```

### List and Inspect Pattern
```rholang
new list(`rho:nunet:deployment:list`),
    manifest(`rho:nunet:deployment:manifest`),
    stdout(`rho:io:stdout`) in {

  // List all deployments
  list!(0) |

  // For each deployment, get manifest
  for (@deployments <- @0) {
    // Process deployment list
    stdout!(["Found deployments:", deployments])
  }
}
```

## Troubleshooting

### "Nunet service not enabled"
**Solution**: Set `NUNET_ENABLED=true` environment variable or `nunet.enabled = true` in config.

### "nunet CLI not found"
**Solution**: Install Nunet DMS and ensure `nunet` command is in PATH.

### "Connection refused"
**Solution**: Ensure Nunet DMS is running. Check with `nunet status`.

### "Invalid ensemble YAML"
**Solution**: Use `rho:nunet:ensemble:validate` to check YAML syntax before deployment.

## Further Documentation

- [Nunet Deployment Guide](../../../docs/NUNET_DEPLOYMENT.md)
- [Nunet DMS Documentation](https://docs.nunet.io/)
- [Ensemble YAML Format](https://docs.nunet.io/ensemble-format)

## Contributing

To add more examples:

1. Create a new `.rho` file following the naming pattern
2. Include clear comments explaining the example
3. Document prerequisites and usage
4. Update this README with the new example
