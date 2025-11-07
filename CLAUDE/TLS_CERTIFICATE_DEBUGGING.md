# TLS Certificate Debugging Session - 2025-10-30

## Issue Summary

Validators could not connect to the bootstrap node during genesis ceremony, causing the network to fail to initialize.

## Initial Symptoms

- Bootstrap node: "Failed to connect to peer rnode://...@rnode.validator1: Peer is currently unavailable"
- Validators: "Failed to connect to peer rnode://...@rnode.bootstrap: Peer is currently unavailable"
- Genesis ceremony stuck: "Failed to meet approval conditions. Signatures: 0 of 2 required"
- Validators indefinitely showing "Waiting for first connection"

## Investigation Steps

### 1. Port Configuration Check (Dead End)
- Verified Docker port mappings were correct (40400 for protocol, 40404 for discovery)
- Confirmed all containers on same Docker network
- TCP connectivity tests showed network was functioning
- **Finding**: Ports were configured correctly, not a network issue

### 2. Enable TLS Debug Logging (Breakthrough)
Modified `docker/conf/logback.xml` to enable debug logging:
```xml
<logger name="io.netty" level="debug" />
<logger name="io.grpc" level="debug" />
```

**Key Discovery**: Found the actual error buried in DEBUG logs:
```
java.security.cert.CertificateException: No name matching 361e20acbeb7177e301d97d4886eed9d2ccd5564 found
    at java.base/sun.security.util.HostnameChecker.matchDNS(HostnameChecker.java:234)
    at coop.rchain.comm.transport.HostnameTrustManager.checkIdentity(HostnameTrustManagerFactory.scala:115)
```

### 3. Certificate Inspection
```bash
openssl x509 -in docker/keys/bootstrap/node.certificate.pem -text -noout | grep -A5 "Subject Alternative Name"
```

**Result**: Certificate had NO Subject Alternative Names!

The certificate only had `Subject: CN=f1r3fly-bootstrap`, but F1R3FLY verifies peers using the node ID as the hostname, which must be in the SAN field.

## Root Cause

The `generate-genesis-keys.sh` script generated certificates with:
```bash
openssl req -new -x509 -key "$node_dir/node.key.pem" \
    -out "$node_dir/node.certificate.pem" \
    -days 3650 -subj "/CN=f1r3fly-$node"
```

This creates a certificate with only a Common Name (CN), no Subject Alternative Names. However:
1. F1R3FLY derives the node ID from the certificate's public key
2. Nodes connect to each other using `rnode://<node-id>@<hostname>:40400`
3. TLS verification checks if the certificate's SAN matches `<node-id>`
4. **Certificate had no SAN → verification fails → connection rejected**

## Solution

### Updated Certificate Generation (Two-Pass Approach)

Modified `docker/generate-genesis-keys.sh` to:

1. **First Pass**: Generate temporary certificate to derive node ID
```bash
openssl req -new -x509 -key "$node_dir/node.key.pem" \
    -out "$node_dir/node.certificate.tmp.pem" \
    -days 3650 -subj "/CN=f1r3fly-$node"

node_id=$($NODE_CLI get-node-id -c "$node_dir/node.certificate.tmp.pem")
```

2. **Second Pass**: Generate final certificate with node ID in SAN
```bash
# Create OpenSSL config with SAN extension
cat > "$node_dir/san.cnf" << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
x509_extensions = v3_req

[req_distinguished_name]

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = $node
DNS.2 = rnode.$node
DNS.3 = f1r3fly-$node
DNS.4 = $node_id
EOF

openssl req -new -x509 -key "$node_dir/node.key.pem" \
    -out "$node_dir/node.certificate.pem" \
    -days 3650 -subj "/CN=f1r3fly-$node" \
    -config "$node_dir/san.cnf"
```

3. **Verify**: Certificate now contains all required SANs including the node ID

### Improved Error Logging

Added ERROR-level logging in `comm/src/main/scala/coop/rchain/comm/transport/HostnameTrustManagerFactory.scala`:

```scala
try {
  HostnameChecker.getInstance(HostnameChecker.TYPE_TLS).`match`(host, cert)
} catch {
  case e: CertificateException =>
    val sans = Option(cert.getSubjectAlternativeNames)
      .map(_.asScala.toList.flatMap { san =>
        san.asScala.toList match {
          case List(_: Integer, value: String) => Some(value)
          case _ => None
        }
      })
      .getOrElse(List.empty)

    logger.error(
      s"TLS certificate invalid: node ID '$host' not found in certificate SANs [${sans.mkString(", ")}]. " +
      s"Regenerate certificates with 'cd docker && ./generate-genesis-keys.sh' to include node IDs in SANs."
    )
    throw e
}
```

**Why This Helps**:
- Error is now logged at ERROR level (visible by default)
- Shows exactly which node ID was not found
- Shows which SANs are actually present
- Provides actionable fix command
- **Time savings**: Would have found this issue in seconds instead of hours

## Verification

After regeneration and restart:
```bash
# Regenerate certificates
cd docker
./generate-genesis-keys.sh

# Restart shard
docker compose -f shard-with-autopropose.yml down
docker compose -f shard-with-autopropose.yml up -d

# Wait 30 seconds and check
docker logs rnode.validator1 | grep "approved"
```

**Result**:
```
INFO coop.rchain.casper.engine.Initializing - Received approved block from bootstrap node.
INFO coop.rchain.casper.engine.Initializing - Valid approved block Block #0 (...) received.
```

**SUCCESS!** Genesis ceremony completed, network initialized.

## Lessons Learned

1. **TLS errors are silent by default**: Critical TLS verification failures were only visible in DEBUG logs
2. **F1R3FLY's unique requirement**: Unlike typical TLS, F1R3FLY requires node IDs in certificate SANs
3. **Chicken-and-egg problem**: Need certificate to derive node ID, need node ID to create valid certificate
4. **Solution**: Two-pass generation (temp cert → node ID → final cert with SAN)
5. **Error logging matters**: Hours of debugging could have been seconds with proper ERROR logs

## Time Impact

- **Without ERROR logging**: ~2 hours of investigation
- **With ERROR logging**: Would have been ~2 minutes

The improved error message immediately tells you:
1. What failed (TLS certificate verification)
2. Why it failed (node ID not in SAN)
3. What the certificate has (actual SANs)
4. How to fix it (exact command to run)

## Files Modified

1. `docker/generate-genesis-keys.sh` - Two-pass certificate generation with SANs
2. `comm/src/main/scala/coop/rchain/comm/transport/HostnameTrustManagerFactory.scala` - ERROR-level logging
3. `docker/TROUBLESHOOTING.md` - Documentation for future issues
4. `CLAUDE/TLS_CERTIFICATE_DEBUGGING.md` - This file
