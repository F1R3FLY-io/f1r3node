import logging
import time
from random import Random
from typing import Any

import pytest

from docker.client import DockerClient
from f1r3fly.crypto import PrivateKey

from . import conftest
from .common import (
    CommandLineOptions,
)
from .rnode import (
    ready_bootstrap_with_network
)

logger = logging.getLogger(__name__)

BOOTSTRAP_NODE_KEYS = PrivateKey.from_hex("80366db5fbb8dad7946f27037422715e4176dda41d582224db87b6c3b783d709")
BONDED_VALIDATOR_KEY_1 = PrivateKey.from_hex("120d42175739387af0264921bb117e4c4c05fbe2ce5410031e8b158c6e414bb5")


# Increase gRPC max message size to accommodate deploy data + overhead
node_cli_options = {
    "--api-grpc-max-recv-message-size": 2_000_000_000,
}

# Set large max message size to support large deploys
client_grpc_options = (
    ('grpc.max_send_message_length', 2_000_000_000),
    ('grpc.max_receive_message_length', 2_000_000_000),
)

@pytest.mark.skip(reason="Long-running test - skipped to avoid blocking CI/CD")
def test_parametrized_deploy_data(command_line_options: CommandLineOptions, random_generator: Random,
                                     docker_client: DockerClient, record_property: Any) -> None:
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, cli_options=node_cli_options, mem_limit="13G") as bootstrap_node:
        rholang_term = """
            new stdout(`rho:io:stdout`),
                myData(`rho:deploy:param:myData`),
                myInt(`rho:deploy:param:myInt`),
                myString(`rho:deploy:param:myString`),
                myBool(`rho:deploy:param:myBool`)
            in {
                stdout!([ "accessing bytes", *myData.length()]) |
                stdout!([ "accessing int", *myInt]) |
                stdout!([ "accessing string", *myString]) |
                stdout!([ "accessing bool", *myBool])
            }
        """

        binary_data_length = 5 * 1024 * 1024  # 5MB
        binary_data = bytes(i % 256 for i in range(binary_data_length))

        start_time = time.time()
        bootstrap_node.deploy_string(
            rholang_term,
            BOOTSTRAP_NODE_KEYS,
            phlo_limit=100000000000,
            phlo_price=1,
            parameters={"myData": binary_data, "myInt": 123, "myString": "test", "myBytes": binary_data, "myBool": True},
            grpc_options=client_grpc_options,
        )
        block_hash = bootstrap_node.propose()
        duration = time.time() - start_time
        logger.info("Deploy and propose took %.3f seconds in total.", duration)
        record_property("deploy_propose_duration", f"{duration:.3f}")
        block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
        deploy = block_info.deploys[0]
        assert not deploy.errored

def test_parametrized_deploy_data_6gb_total_1mb_per_chunk_8mb_propose(command_line_options: CommandLineOptions, random_generator: Random,
                                     docker_client: DockerClient, record_property: Any) -> None:
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, cli_options=node_cli_options, mem_limit="17G") as bootstrap_node:
        # Deploy data using 64MB chunks, propose every 512MB
        chunk_size = 1 * 1024 * 1024  # 1MB per chunk
        total_size = 6 * 1024 * 1024 * 1024  # 6GB total
        num_chunks = total_size // chunk_size  # 6144 chunks
        chunks_per_block = 8 * 1024 * 1024 // chunk_size  # 8MB per block

        block_hashes = []
        start_time = time.time()

        for i in range(num_chunks):
            rholang_term = f"""
                new stdout(`rho:io:stdout`),
                    myData(`rho:deploy:param:myBytes`)
                in {{
                    stdout!([ "accessing bytes chunk {i}", *myData.length()]) | 
                    @{i}!(*myData)
                }}
            """

            # Generate 64MB chunk data
            binary_data = bytes(j % 256 for j in range(chunk_size))

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                parameters={"myBytes": binary_data},
                grpc_options=client_grpc_options,
            )
            logger.info("Deployed chunk %d/%d (%f %% done)", i + 1, num_chunks, (i / num_chunks) * 100)

            # Propose every 128MB
            if (i + 1) % chunks_per_block == 0:
                block_hash = bootstrap_node.propose()
                block_hashes.append(block_hash)
                logger.info("Proposed block %d with hash %s (128MB)", len(block_hashes), block_hash)

        # Propose any remaining deploys
        if num_chunks % chunks_per_block != 0:
            block_hash = bootstrap_node.propose()
            block_hashes.append(block_hash)
            logger.info("Proposed final block %d with hash %s", len(block_hashes), block_hash)

        duration = time.time() - start_time
        logger.info("Deploy and propose took %.3f seconds in total for %d blocks.", duration, len(block_hashes))
        record_property("deploy_propose_duration", f"{duration:.3f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash)
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified successfully with %d deploys", idx + 1, len(block_info.deploys))


def test_parametrized_deploy_data_6gb_total_2mb_per_chunk_32mb_propose(command_line_options: CommandLineOptions, random_generator: Random,
                                     docker_client: DockerClient, record_property: Any) -> None:
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, cli_options=node_cli_options, mem_limit="17G") as bootstrap_node:
        # Deploy data using 64MB chunks, propose every 512MB
        chunk_size = 2 * 1024 * 1024  # 2MB per chunk
        total_size = 6 * 1024 * 1024 * 1024  # 6GB total
        num_chunks = total_size // chunk_size  # 3072 chunks
        chunks_per_block = 32 * 1024 * 1024 // chunk_size  # 32MB per block

        block_hashes = []
        start_time = time.time()

        for i in range(num_chunks):
            rholang_term = f"""
                new stdout(`rho:io:stdout`),
                    myData(`rho:deploy:param:myBytes`)
                in {{
                    stdout!([ "accessing bytes chunk {i}", *myData.length()]) | 
                    @{i}!(*myData)
                }}
            """

            # Generate 64MB chunk data
            binary_data = bytes(j % 256 for j in range(chunk_size))

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                parameters={"myBytes": binary_data},
                grpc_options=client_grpc_options,
            )
            logger.info("Deployed chunk %d/%d (%f %% done)", i + 1, num_chunks, (i / num_chunks) * 100)

            # Propose every 128MB
            if (i + 1) % chunks_per_block == 0:
                block_hash = bootstrap_node.propose()
                block_hashes.append(block_hash)
                logger.info("Proposed block %d with hash %s (128MB)", len(block_hashes), block_hash)

        # Propose any remaining deploys
        if num_chunks % chunks_per_block != 0:
            block_hash = bootstrap_node.propose()
            block_hashes.append(block_hash)
            logger.info("Proposed final block %d with hash %s", len(block_hashes), block_hash)

        duration = time.time() - start_time
        logger.info("Deploy and propose took %.3f seconds in total for %d blocks.", duration, len(block_hashes))
        record_property("deploy_propose_duration", f"{duration:.3f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash)
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified successfully with %d deploys", idx + 1, len(block_info.deploys))


#@pytest.mark.skip(reason="Long-running test (6GB data processing) - skipped to avoid blocking CI/CD")
def test_parametrized_deploy_data_6gb_total_2mb_per_chunk_128mb_propose(command_line_options: CommandLineOptions, random_generator: Random,
                                     docker_client: DockerClient, record_property: Any) -> None:
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, cli_options=node_cli_options, mem_limit="17G") as bootstrap_node:
        # Deploy data using 64MB chunks, propose every 512MB
        chunk_size = 2 * 1024 * 1024  # 32MB per chunk
        total_size = 6 * 1024 * 1024 * 1024  # 6GB total
        num_chunks = total_size // chunk_size  # 192 chunks
        chunks_per_block = 128 * 1024 * 1024 // chunk_size  # 128MB per block

        block_hashes = []
        start_time = time.time()

        for i in range(num_chunks):
            rholang_term = f"""
                new stdout(`rho:io:stdout`),
                    myData(`rho:deploy:param:myBytes`)
                in {{
                    stdout!([ "accessing bytes chunk {i}", *myData.length()]) | 
                    @{i}!(*myData)
                }}
            """

            # Generate 64MB chunk data
            binary_data = bytes(j % 256 for j in range(chunk_size))

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                parameters={"myBytes": binary_data},
                grpc_options=client_grpc_options,
            )
            logger.info("Deployed chunk %d/%d (%f %% done)", i + 1, num_chunks, (i / num_chunks) * 100)

            # Propose every 128MB
            if (i + 1) % chunks_per_block == 0:
                block_hash = bootstrap_node.propose()
                block_hashes.append(block_hash)
                logger.info("Proposed block %d with hash %s (128MB)", len(block_hashes), block_hash)

        # Propose any remaining deploys
        if num_chunks % chunks_per_block != 0:
            block_hash = bootstrap_node.propose()
            block_hashes.append(block_hash)
            logger.info("Proposed final block %d with hash %s", len(block_hashes), block_hash)

        duration = time.time() - start_time
        logger.info("Deploy and propose took %.3f seconds in total for %d blocks.", duration, len(block_hashes))
        record_property("deploy_propose_duration", f"{duration:.3f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash)
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified successfully with %d deploys", idx + 1, len(block_info.deploys))

#@pytest.mark.skip(reason="Long-running test (6GB data processing) - skipped to avoid blocking CI/CD")
def test_parametrized_deploy_data_6gb_total_32mb_per_chunk_128mb_propose(command_line_options: CommandLineOptions, random_generator: Random,
                                     docker_client: DockerClient, record_property: Any) -> None:
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, cli_options=node_cli_options, mem_limit="17G") as bootstrap_node:
        # Deploy data using 64MB chunks, propose every 512MB
        chunk_size = 32 * 1024 * 1024  # 32MB per chunk
        total_size = 6 * 1024 * 1024 * 1024  # 6GB total
        num_chunks = total_size // chunk_size  # 192 chunks
        chunks_per_block = 128 * 1024 * 1024 // chunk_size  # 128MB per block

        block_hashes = []
        start_time = time.time()

        for i in range(num_chunks):
            rholang_term = f"""
                new stdout(`rho:io:stdout`),
                    myData(`rho:deploy:param:myBytes`)
                in {{
                    stdout!([ "accessing bytes chunk {i}", *myData.length()]) | 
                    @{i}!(*myData)
                }}
            """

            # Generate 64MB chunk data
            binary_data = bytes(j % 256 for j in range(chunk_size))

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                parameters={"myBytes": binary_data},
                grpc_options=client_grpc_options,
            )
            logger.info("Deployed chunk %d/%d (%f %% done)", i + 1, num_chunks, (i / num_chunks) * 100)

            # Propose every 128MB
            if (i + 1) % chunks_per_block == 0:
                block_hash = bootstrap_node.propose()
                block_hashes.append(block_hash)
                logger.info("Proposed block %d with hash %s (128MB)", len(block_hashes), block_hash)

        # Propose any remaining deploys
        if num_chunks % chunks_per_block != 0:
            block_hash = bootstrap_node.propose()
            block_hashes.append(block_hash)
            logger.info("Proposed final block %d with hash %s", len(block_hashes), block_hash)

        duration = time.time() - start_time
        logger.info("Deploy and propose took %.3f seconds in total for %d blocks.", duration, len(block_hashes))
        record_property("deploy_propose_duration", f"{duration:.3f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash)
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified successfully with %d deploys", idx + 1, len(block_info.deploys))

def test_parametrized_deploy_data_6gb_total_32mb_per_chunk_256mb_propose(command_line_options: CommandLineOptions, random_generator: Random,
                                     docker_client: DockerClient, record_property: Any) -> None:
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, cli_options=node_cli_options, mem_limit="17G") as bootstrap_node:
        # Deploy data using 64MB chunks, propose every 512MB
        chunk_size = 32 * 1024 * 1024  # 32MB per chunk
        total_size = 6 * 1024 * 1024 * 1024  # 6GB total
        num_chunks = total_size // chunk_size  # 192 chunks
        chunks_per_block = 256 * 1024 * 1024 // chunk_size  # 128MB per block

        block_hashes = []
        start_time = time.time()

        for i in range(num_chunks):
            rholang_term = f"""
                new stdout(`rho:io:stdout`),
                    myData(`rho:deploy:param:myBytes`)
                in {{
                    stdout!([ "accessing bytes chunk {i}", *myData.length()]) | 
                    @{i}!(*myData)
                }}
            """

            # Generate 64MB chunk data
            binary_data = bytes(j % 256 for j in range(chunk_size))

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                parameters={"myBytes": binary_data},
                grpc_options=client_grpc_options,
            )
            logger.info("Deployed chunk %d/%d (%f %% done)", i + 1, num_chunks, (i / num_chunks) * 100)

            # Propose every 128MB
            if (i + 1) % chunks_per_block == 0:
                block_hash = bootstrap_node.propose()
                block_hashes.append(block_hash)
                logger.info("Proposed block %d with hash %s (128MB)", len(block_hashes), block_hash)

        # Propose any remaining deploys
        if num_chunks % chunks_per_block != 0:
            block_hash = bootstrap_node.propose()
            block_hashes.append(block_hash)
            logger.info("Proposed final block %d with hash %s", len(block_hashes), block_hash)

        duration = time.time() - start_time
        logger.info("Deploy and propose took %.3f seconds in total for %d blocks.", duration, len(block_hashes))
        record_property("deploy_propose_duration", f"{duration:.3f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash)
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified successfully with %d deploys", idx + 1, len(block_info.deploys))

#@pytest.mark.skip(reason="Long-running test (6GB data processing) - skipped to avoid blocking CI/CD")
def test_parametrized_deploy_data_6gb_total_128mb_per_chunk_256mb_propose(command_line_options: CommandLineOptions, random_generator: Random,
                                     docker_client: DockerClient, record_property: Any) -> None:
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, cli_options=node_cli_options, mem_limit="17G") as bootstrap_node:
        # Deploy data using 64MB chunks, propose every 512MB
        chunk_size = 128 * 1024 * 1024  # 32MB per chunk
        total_size = 6 * 1024 * 1024 * 1024  # 6GB total
        num_chunks = total_size // chunk_size  # 192 chunks
        chunks_per_block = 256 * 1024 * 1024 // chunk_size  # 128MB per block

        block_hashes = []
        start_time = time.time()

        for i in range(num_chunks):
            rholang_term = f"""
                new stdout(`rho:io:stdout`),
                    myData(`rho:deploy:param:myBytes`)
                in {{
                    stdout!([ "accessing bytes chunk {i}", *myData.length()]) | 
                    @{i}!(*myData)
                }}
            """

            # Generate 64MB chunk data
            binary_data = bytes(j % 256 for j in range(chunk_size))

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                parameters={"myBytes": binary_data},
                grpc_options=client_grpc_options,
            )
            logger.info("Deployed chunk %d/%d (%f %% done)", i + 1, num_chunks, (i / num_chunks) * 100)

            # Propose every 128MB
            if (i + 1) % chunks_per_block == 0:
                block_hash = bootstrap_node.propose()
                block_hashes.append(block_hash)
                logger.info("Proposed block %d with hash %s (128MB)", len(block_hashes), block_hash)

        # Propose any remaining deploys
        if num_chunks % chunks_per_block != 0:
            block_hash = bootstrap_node.propose()
            block_hashes.append(block_hash)
            logger.info("Proposed final block %d with hash %s", len(block_hashes), block_hash)

        duration = time.time() - start_time
        logger.info("Deploy and propose took %.3f seconds in total for %d blocks.", duration, len(block_hashes))
        record_property("deploy_propose_duration", f"{duration:.3f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash)
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified successfully with %d deploys", idx + 1, len(block_info.deploys))

def test_parametrized_deploy_data_6gb_total_256mb_per_chunk_256mb_propose(command_line_options: CommandLineOptions, random_generator: Random,
                                     docker_client: DockerClient, record_property: Any) -> None:
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, cli_options=node_cli_options, mem_limit="17G") as bootstrap_node:
        # Deploy data using 64MB chunks, propose every 512MB
        chunk_size = 256 * 1024 * 1024  # 256MB per chunk
        total_size = 6 * 1024 * 1024 * 1024  # 6GB total
        num_chunks = total_size // chunk_size  # 24 chunks
        chunks_per_block = 256 * 1024 * 1024 // chunk_size  # 1 chunk per block

        block_hashes = []
        start_time = time.time()

        for i in range(num_chunks):
            rholang_term = f"""
                new stdout(`rho:io:stdout`),
                    myData(`rho:deploy:param:myBytes`)
                in {{
                    stdout!([ "accessing bytes chunk {i}", *myData.length()]) | 
                    @{i}!(*myData)
                }}
            """

            # Generate 64MB chunk data
            binary_data = bytes(j % 256 for j in range(chunk_size))

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                parameters={"myBytes": binary_data},
                grpc_options=client_grpc_options,
            )
            logger.info("Deployed chunk %d/%d (%f %% done)", i + 1, num_chunks, (i / num_chunks) * 100)

            # Propose every 128MB
            if (i + 1) % chunks_per_block == 0:
                block_hash = bootstrap_node.propose()
                block_hashes.append(block_hash)
                logger.info("Proposed block %d with hash %s (128MB)", len(block_hashes), block_hash)

        # Propose any remaining deploys
        if num_chunks % chunks_per_block != 0:
            block_hash = bootstrap_node.propose()
            block_hashes.append(block_hash)
            logger.info("Proposed final block %d with hash %s", len(block_hashes), block_hash)

        duration = time.time() - start_time
        logger.info("Deploy and propose took %.3f seconds in total for %d blocks.", duration, len(block_hashes))
        record_property("deploy_propose_duration", f"{duration:.3f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash)
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified successfully with %d deploys", idx + 1, len(block_info.deploys))
