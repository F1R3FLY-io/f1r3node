import pytest
import docker
import time
import subprocess

# Docker image name from build
IMAGE_NAME = "f1r3flyindustries/f1r3fly-scala-node:latest"
# Ports used by the HEALTHCHECK
PORTS = {"40401/tcp": 40401, "40403/tcp": 40403}
# Timeout for container to become healthy (in seconds)
HEALTHCHECK_TIMEOUT = 60

@pytest.fixture(scope="module")
def docker_client():
    """Initialize Docker client."""
    return docker.from_env()

@pytest.fixture(scope="module")
def rnode_container(docker_client):
    """Start the RNode container and ensure it's healthy."""
    # Pull image if needed (optional, as CI should have it)
    try:
        docker_client.images.get(IMAGE_NAME)
    except docker.errors.ImageNotFound:
        raise Exception(f"Docker image {IMAGE_NAME} not found")

    # Start container
    container = docker_client.containers.run(
        IMAGE_NAME,
        detach=True,
        ports=PORTS,
        name="rnode-healthcheck-test",
        environment=["_JAVA_OPTIONS=-XX:MaxRAMPercentage=35.0"]
    )

    # Wait for container to be healthy
    start_time = time.time()
    while time.time() - start_time < HEALTHCHECK_TIMEOUT:
        container.reload()
        health_status = container.attrs["State"]["Health"]["Status"]
        if health_status == "healthy":
            yield container
            break
        elif health_status == "unhealthy":
            logs = container.logs().decode("utf-8")
            raise Exception(f"Container is unhealthy: {logs}")
        time.sleep(1)
    else:
        logs = container.logs().decode("utf-8")
        raise Exception(f"Container did not become healthy within {HEALTHCHECK_TIMEOUT}s: {logs}")

    # Cleanup
    container.stop()
    container.remove()

def test_healthcheck_command(rnode_container):
    """Test the HEALTHCHECK command defined in the Docker image."""
    # Execute the HEALTHCHECK command
    healthcheck_cmd = (
        "grpcurl -plaintext 127.0.0.1:40401 casper.v1.DeployService.status | jq -e && "
        "curl -s 127.0.0.1:40403/status | jq -e"
    )
    exec_result = rnode_container.exec_run(
        cmd=["/bin/sh", "-c", healthcheck_cmd],
        user="daemon"  # Run as daemon user, matching Docker setup
    )

    # Check exit code and output
    assert exec_result.exit_code == 0, (
        f"HEALTHCHECK command failed with exit code {exec_result.exit_code}: "
        f"{exec_result.output.decode('utf-8')}"
    )

    # Optionally, verify output contains expected JSON (adjust based on actual response)
    output = exec_result.output.decode("utf-8")
    assert output, "HEALTHCHECK command produced no output"
