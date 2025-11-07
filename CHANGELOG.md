## [Unreleased]

### Added
- **Nunet DMS Integration**: Added 7 system contracts for orchestrating Nunet DMS deployments from Rholang
  - `rho:nunet:deployment:new` - Deploy a new ensemble to Nunet DMS
  - `rho:nunet:deployment:status` - Check deployment status
  - `rho:nunet:deployment:list` - List all deployments
  - `rho:nunet:deployment:logs` - Get deployment logs
  - `rho:nunet:deployment:manifest` - Get deployment manifest (peers, IPs, ports)
  - `rho:nunet:ensemble:generate` - Generate ensemble YAML from config
  - `rho:nunet:ensemble:validate` - Validate ensemble YAML
- New `NunetService` trait and implementation for DMS CLI integration
- Configuration support via HOCON (`nunet.enabled`, `nunet.dms.*`)
- Comprehensive test suite with mock DMS service (8 tests)

### Configuration
To enable Nunet system contracts:
```bash
export NUNET_ENABLED=true
# or in application.conf:
nunet.enabled = true
```

See `docs/NUNET_DEPLOYMENT.md` for complete documentation.

## [v0.1.0-SNAPSHOT] - 2023-05-15
- Added new features
- Fixed bugs
- Improved performance


