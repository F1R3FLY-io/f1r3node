# Profiling F1R3FLY Nodes with JFR

This guide explains how to profile running F1R3FLY validator nodes using Java Flight Recorder (JFR) and JDK Mission Control.

## Prerequisites

- Oracle JDK installed on your host machine (OpenJDK does not include Mission Control)
- Running F1R3FLY network with validator containers
- The Java process in containers runs as PID 1

## Quick Start

Profile a validator node for 5 minutes and analyze the results:

```bash
# Start JFR recording (5 minutes)
docker exec rnode.validator1 jcmd 1 JFR.start name=profile duration=300s filename=/var/lib/rnode/profile.jfr

# Wait 5 minutes for recording to complete...

# Check if recording is complete
docker exec rnode.validator1 jcmd 1 JFR.check

# Copy the recording file to your host
docker cp rnode.validator1:/var/lib/rnode/profile.jfr ./validator1-profile.jfr

# Open in JDK Mission Control
jmc validator1-profile.jfr
```

## Step-by-Step Guide

### 1. Start a JFR Recording

The Java process in the container always runs as PID 1, so use `jcmd 1`:

```bash
# Basic recording (5 minutes)
docker exec rnode.validator1 jcmd 1 JFR.start name=profile duration=300s filename=/var/lib/rnode/profile.jfr

# Longer recording (30 minutes)
docker exec rnode.validator1 jcmd 1 JFR.start name=profile duration=1800s filename=/var/lib/rnode/profile.jfr

# Recording with settings profile (less overhead)
docker exec rnode.validator1 jcmd 1 JFR.start name=profile duration=300s filename=/var/lib/rnode/profile.jfr settings=profile
```

### 2. Check Recording Status

Monitor the active recording:

```bash
# Check if recording is running or stopped
docker exec rnode.validator1 jcmd 1 JFR.check

# Watch the file size grow
docker exec rnode.validator1 ls -lh /var/lib/rnode/profile.jfr
```

Output will show:
- `(running)` - Recording is still in progress
- `(stopped)` - Recording is complete and ready to copy

### 3. Stop Recording Early (Optional)

If you want to stop before the duration expires:

```bash
# Dump current data to file
docker exec rnode.validator1 jcmd 1 JFR.dump name=profile filename=/var/lib/rnode/profile.jfr

# Stop the recording
docker exec rnode.validator1 jcmd 1 JFR.stop name=profile
```

### 4. Copy Recording to Host

Once the recording is complete (stopped):

```bash
# Copy from container to current directory
docker cp rnode.validator1:/var/lib/rnode/profile.jfr ./validator1-profile.jfr

# Copy with timestamp in filename
docker cp rnode.validator1:/var/lib/rnode/profile.jfr ./validator1-$(date +%Y%m%d-%H%M%S).jfr
```

### 5. Analyze with JDK Mission Control

```bash
# Open JDK Mission Control
jmc

# Or open the file directly
jmc validator1-profile.jfr
```

In Mission Control, you can analyze:
- **CPU Usage**: Method Profiling tab shows hot methods
- **Memory Allocations**: Memory tab shows allocation patterns
- **Lock Contention**: Lock Instances tab shows synchronization bottlenecks
- **I/O Operations**: File and Socket I/O tabs
- **Garbage Collection**: GC tab shows pause times and heap usage

## Profiling Other Validators

The same process works for any validator or bootstrap node:

```bash
# Validator 2
docker exec rnode.validator2 jcmd 1 JFR.start name=profile duration=300s filename=/var/lib/rnode/profile.jfr
docker cp rnode.validator2:/var/lib/rnode/profile.jfr ./validator2-profile.jfr

# Validator 3
docker exec rnode.validator3 jcmd 1 JFR.start name=profile duration=300s filename=/var/lib/rnode/profile.jfr
docker cp rnode.validator3:/var/lib/rnode/profile.jfr ./validator3-profile.jfr

# Bootstrap node
docker exec rnode.bootstrap jcmd 1 JFR.start name=profile duration=300s filename=/var/lib/rnode/profile.jfr
docker cp rnode.bootstrap:/var/lib/rnode/profile.jfr ./bootstrap-profile.jfr
```

## JFR Recording Settings

JFR has two built-in profiles:

- **default**: More events, higher overhead (~2-5% CPU)
- **profile**: Fewer events, lower overhead (~1% CPU)

For production profiling, use the `profile` settings:

```bash
docker exec rnode.validator1 jcmd 1 JFR.start name=profile duration=300s filename=/var/lib/rnode/profile.jfr settings=profile
```

## Troubleshooting

### Recording not found
If you get "No such recording", list all recordings:
```bash
docker exec rnode.validator1 jcmd 1 JFR.check
```

### File not found when copying
Make sure the recording has stopped (not still running):
```bash
docker exec rnode.validator1 jcmd 1 JFR.check
```

### Mission Control won't start
JDK Mission Control requires Oracle JDK. Install from:
https://www.oracle.com/java/technologies/downloads/

### Container not running
Check container status:
```bash
docker ps | grep validator1
```

## Comparing Multiple Nodes

To find performance differences across validators:

1. Record all validators simultaneously:
```bash
for node in rnode.validator1 rnode.validator2 rnode.validator3; do
  docker exec $node jcmd 1 JFR.start name=profile duration=300s filename=/var/lib/rnode/profile.jfr
done
```

2. Wait 5 minutes, then copy all recordings:
```bash
for node in rnode.validator1 rnode.validator2 rnode.validator3; do
  name=$(echo $node | sed 's/rnode.//')
  docker cp $node:/var/lib/rnode/profile.jfr ./$name-profile.jfr
done
```

3. Open each file in Mission Control and compare the Method Profiling results

## Continuous Recording (Advanced)

For ongoing performance monitoring, enable JFR at container startup by modifying the docker-compose.yml command:

```yaml
validator1:
  command:
    [
      "-XX:+FlightRecorder",
      "-XX:StartFlightRecording=name=continuous,disk=true,maxsize=500M,dumponexit=true,filename=/var/lib/rnode/continuous.jfr",
      "-Dlogback.configurationFile=/var/lib/rnode/logback.xml",
      "run",
      # ... rest of args
    ]
```

This creates a continuous circular buffer recording that can be dumped at any time:
```bash
docker exec rnode.validator1 jcmd 1 JFR.dump name=continuous filename=/var/lib/rnode/snapshot.jfr
```

## Resources

- [GraalVM JFR Documentation](https://www.graalvm.org/latest/reference-manual/native-image/debugging-and-diagnostics/JFR/)
- [JDK Mission Control User Guide](https://docs.oracle.com/en/java/java-components/jdk-mission-control/)
- [JFR Command Reference](https://docs.oracle.com/en/java/javase/17/docs/specs/man/jfr.html)
