"""
Heartbeat Integration Tests

Tests for the heartbeat proposer functionality that automatically creates blocks
to maintain blockchain liveness when the Last Finalized Block (LFB) becomes stale.

These tests verify:
1. Heartbeat creates blocks automatically when idle (no user deploys)
2. Heartbeat is disabled with proper warning when max-number-of-parents=1
3. Heartbeat logs expected startup messages
"""

import time
from random import Random
from typing import Generator
from contextlib import contextmanager

from f1r3fly.crypto import PrivateKey
from docker.client import DockerClient

from .common import (
    CommandLineOptions,
)
from .conftest import (
    testing_context,
)
from .rnode import (
    Node,
    started_bootstrap_with_network,
)
from .wait import (
    wait_for_approved_block_received_handler_state,
)


USER_KEY = PrivateKey.from_hex("b2527b00340a83e302beae2a8daf6d654e8e57541acfa261cc1b5635eb16aa15")


@contextmanager
def start_node_with_heartbeat(
    command_line_options: CommandLineOptions,
    docker_client: DockerClient,
    random_generator: Random,
    heartbeat_enabled: bool = True,
    heartbeat_check_interval: int = 5,
    heartbeat_max_lfb_age: int = 3,
    max_number_of_parents: int = 10,
) -> Generator[Node, None, None]:
    """Start a node with heartbeat configuration."""
    genesis_vault = {
        USER_KEY: 5000000000
    }

    # cli_flags for boolean flags (no value), cli_options for key-value pairs
    cli_flags = {"--heartbeat-enabled"} if heartbeat_enabled else set()
    cli_options = {
        "--heartbeat-check-interval": f"{heartbeat_check_interval}seconds",
        "--heartbeat-max-lfb-age": f"{heartbeat_max_lfb_age}seconds",
        "--max-number-of-parents": str(max_number_of_parents),
    }

    with testing_context(command_line_options, random_generator, docker_client, wallets_dict=genesis_vault) as context, \
            started_bootstrap_with_network(context=context, cli_flags=cli_flags, cli_options=cli_options) as bootstrap:
        wait_for_approved_block_received_handler_state(context, bootstrap)
        yield bootstrap


def test_heartbeat_creates_blocks_when_idle(
    command_line_options: CommandLineOptions,
    docker_client: DockerClient,
    random_generator: Random
) -> None:
    """
    Test that heartbeat automatically creates blocks when idle.
    
    This test verifies the core heartbeat functionality:
    1. Start node with heartbeat enabled and short intervals
    2. Wait for genesis/running state
    3. Without any deploys, wait for heartbeat cycles
    4. Verify heartbeat created at least one block beyond genesis
    
    Note: In a single-validator setup with InvalidParents validation,
    only one heartbeat block can be created after genesis. Subsequent
    heartbeat attempts will fail InvalidParents until new blocks from
    other validators are received. This is expected consensus behavior.
    """
    with start_node_with_heartbeat(
        command_line_options,
        docker_client,
        random_generator,
        heartbeat_enabled=True,
        heartbeat_check_interval=5,   # Check every 5 seconds
        heartbeat_max_lfb_age=3,      # LFB stale after 3 seconds
        max_number_of_parents=10,
    ) as bootstrap:
        # Wait for heartbeat to create blocks (no deploys needed)
        # With check_interval=5s and max_lfb_age=3s, first block should appear within 10s
        time.sleep(20)

        # Verify heartbeat created at least one block beyond genesis
        # In single-validator mode, exactly one heartbeat block can be created
        # (subsequent attempts fail InvalidParents validation - expected behavior)
        final_count = bootstrap.get_blocks_count(10)
        assert final_count >= 2, \
            f"Heartbeat should have created at least 1 block beyond genesis: count={final_count}"

        # Verify heartbeat log message shows successful block creation
        logs = bootstrap.logs()
        assert "Heartbeat: Successfully created block" in logs, \
            "Should see successful heartbeat block creation in logs"


def test_heartbeat_disabled_when_max_parents_is_one(
    command_line_options: CommandLineOptions,
    docker_client: DockerClient,
    random_generator: Random
) -> None:
    """
    Test that heartbeat is disabled with warning when max-number-of-parents=1.
    
    This is a safety check - heartbeat blocks would fail validation with
    max-number-of-parents=1 because empty blocks can't include all required parents.
    """
    with start_node_with_heartbeat(
        command_line_options,
        docker_client,
        random_generator,
        heartbeat_enabled=True,
        heartbeat_check_interval=5,
        heartbeat_max_lfb_age=3,
        max_number_of_parents=1,  # Triggers safety check
    ) as bootstrap:
        # Verify warning message in logs
        logs = bootstrap.logs()
        assert "Heartbeat incompatible with max-number-of-parents=1" in logs or \
               "CONFIGURATION ERROR" in logs, \
            "Should log warning about max-number-of-parents=1"

        # Get initial block count
        initial_count = bootstrap.get_blocks_count(10)

        # Wait - heartbeat should NOT create blocks
        time.sleep(15)

        # Verify no automatic blocks created
        final_count = bootstrap.get_blocks_count(10)
        assert final_count == initial_count, \
            f"Heartbeat should be disabled: initial={initial_count}, final={final_count}"


def test_heartbeat_log_messages(
    command_line_options: CommandLineOptions,
    docker_client: DockerClient,
    random_generator: Random
) -> None:
    """
    Test that heartbeat logs expected startup messages.
    
    Verifies proper initialization logging for debugging and monitoring.
    """
    with start_node_with_heartbeat(
        command_line_options,
        docker_client,
        random_generator,
        heartbeat_enabled=True,
        heartbeat_check_interval=10,
        heartbeat_max_lfb_age=30,
        max_number_of_parents=10,
    ) as bootstrap:
        logs = bootstrap.logs()

        # Verify heartbeat started with expected log message
        assert "Heartbeat: Starting with random initial delay" in logs, \
            "Should log heartbeat startup message"


def test_heartbeat_disabled_by_config(
    command_line_options: CommandLineOptions,
    docker_client: DockerClient,
    random_generator: Random
) -> None:
    """
    Test that heartbeat is disabled when --heartbeat-enabled is not set.
    
    Verifies the configuration option works correctly.
    """
    with start_node_with_heartbeat(
        command_line_options,
        docker_client,
        random_generator,
        heartbeat_enabled=False,  # Explicitly disabled
        heartbeat_check_interval=5,
        heartbeat_max_lfb_age=3,
        max_number_of_parents=10,
    ) as bootstrap:
        # Get initial block count
        initial_count = bootstrap.get_blocks_count(10)

        # Wait - heartbeat should NOT create blocks
        time.sleep(15)

        # Verify no automatic blocks created
        final_count = bootstrap.get_blocks_count(10)
        assert final_count == initial_count, \
            f"Heartbeat should be disabled: initial={initial_count}, final={final_count}"

        # Verify no heartbeat startup message
        logs = bootstrap.logs()
        assert "Heartbeat: Starting with random initial delay" not in logs, \
            "Should NOT log heartbeat startup when disabled"
