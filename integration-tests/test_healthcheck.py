import pytest
import docker
import time

@pytest.fixture(scope="module")
def docker_client():
    return docker.from_env()

@pytest.fixture(scope="module")
def rnode_container(docker_client):
    # Start the RNode container
    container = docker_client.containers.run(
        "f1r3flyindustries/f1r3fly-scala-node:latest",
        detach=True,
        ports={"40401/tcp": 40401, "40403/tcp": 40403},
        name="rnode_test_healthcheck"
    )
    
    # Wait for the container to become healthy (up to 60 seconds)
    start_time = time.time()
    while time.time() - start_time < 60:
        container.reload()
        health_status = container.attrs.get("State", {}).get("Health", {}).get("Status")
        if health_status == "healthy":
            break
        elif health_status == "unhealthy":
            container.stop()
            container.remove()
            pytest.fail(f"Container reported unhealthy: {container.attrs['State']['Health']['Log']}")
        time.sleep(2)  # Docker healthcheck runs every 30s by default, poll every 2s
    else:
        container.stop()
        container.remove()
        pytest.fail("Container did not become healthy within 60 seconds")

    yield container

    # Cleanup
    container.stop()
    container.remove()

def test_healthcheck(rnode_container):
    # Verify the container is healthy
    health_status = rnode_container.attrs.get("State", {}).get("Health", {}).get("Status")
    assert health_status == "healthy", (
        f"Container healthcheck failed: {rnode_container.attrs['State']['Health']['Log']}"
    )
