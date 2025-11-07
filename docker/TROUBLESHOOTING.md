# F1R3FLY Docker Troubleshooting Guide

## TLS Certificate Verification Failures

### Problem

Nodes fail to connect to each other with errors like:
```
Failed to connect to peer rnode://...: Peer is currently unavailable
```

When debug logging is enabled, you may see:
```
verification of certificate failed
java.security.cert.CertificateException: No name matching <node-id> found
```

### Root Cause

F1R3FLY nodes use the node ID (derived from the TLS certificate's public key) as the hostname for peer verification. The TLS certificate must include this node ID in its Subject Alternative Name (SAN) field, otherwise the hostname verification fails silently.

This issue typically occurs when:
1. Certificates were generated without SANs
2. Certificates were generated before the node ID was known
3. Old certificates are being reused after code changes

### Symptoms

- Genesis ceremony never completes (bootstrap waits forever for validator signatures)
- Validators show "Waiting for first connection" indefinitely
- No ERROR logs by default (only DEBUG-level messages from Netty)
- Kademlia discovery (port 40404) may work, but protocol connections (port 40400) fail

### Diagnosis

Enable debug logging to see certificate verification errors:

```xml
<!-- In docker/conf/logback.xml -->
<logger name="io.netty" level="debug" />
<logger name="io.grpc" level="debug" />
```

Look for `verification of certificate failed` messages in the logs.

With the improved error logging (after code fix), you'll see:
```
ERROR TLS certificate invalid: node ID 'abc123...' not found in certificate SANs [bootstrap, rnode.bootstrap, f1r3fly-bootstrap]. Regenerate certificates with 'cd docker && ./generate-genesis-keys.sh' to include node IDs in SANs.
```

### Solution

Regenerate certificates with node IDs in the SAN field:

```bash
cd docker
./generate-genesis-keys.sh
```

The script now uses a two-pass approach:
1. Generate temporary certificate → derive node ID from it
2. Generate final certificate with node ID included in SAN

Restart the shard:
```bash
docker compose -f shard-with-autopropose.yml down
docker compose -f shard-with-autopropose.yml up -d
```

### Verification

Check that certificates have the correct SANs:

```bash
openssl x509 -in docker/keys/bootstrap/node.certificate.pem -text -noout | grep -A5 "Subject Alternative Name"
```

You should see:
```
X509v3 Subject Alternative Name:
    DNS:bootstrap, DNS:rnode.bootstrap, DNS:f1r3fly-bootstrap, DNS:<40-char-node-id>
```

Verify node ID matches:
```bash
cat docker/keys/bootstrap/node_id.txt
```

The 40-character hex string should appear in the certificate's SAN field.

### Prevention

- Always use `generate-genesis-keys.sh` to create certificates
- Never manually generate certificates without SANs
- When upgrading RNode or changing TLS code, regenerate certificates
- Keep certificates in version control or document the generation process

## Code Fix Applied

The error logging was improved in `comm/src/main/scala/coop/rchain/comm/transport/HostnameTrustManagerFactory.scala` to provide clear, actionable error messages when certificate verification fails.

Before:
- Only DEBUG-level Netty logs showed the error
- Required enabling debug logging and searching through verbose output
- No guidance on how to fix the issue

After:
- ERROR-level log with concise message
- Shows which node ID was not found
- Shows which SANs are present in the certificate
- Provides exact command to fix the issue
