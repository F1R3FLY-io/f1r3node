"""
Parametrized Deploy Data Test Matrix

This module provides a systematic exploration of chunk size, propose size, 
total data, and RAM configurations to identify optimal settings.

Based on previous findings:
- 1MB chunk × 256MB propose @ 17GB RAM: PASSED (204.80 MB/min)
- Larger chunks tend to OOM faster
- Smaller propose batches had finalization issues (now fixed)

Test naming convention: test_matrix_{chunk}mb_chunk_{propose}mb_propose_{total}mb_total_{ram}g_ram
"""
import logging
import time
from random import Random
from typing import Any


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


# pylint: disable=too-many-positional-arguments, too-many-locals
def run_deploy_test(
    command_line_options: CommandLineOptions,
    random_generator: Random,
    docker_client: DockerClient,
    record_property: Any,
    chunk_size_mb: int,
    propose_size_mb: int,
    total_size_mb: int,
    ram_limit_gb: int,
    sleep_after_minutes: int = 0,
) -> None:
    """
    Generic test runner for deploy matrix tests.
    
    Args:
        chunk_size_mb: Size of each chunk in MB
        propose_size_mb: Total MB to deploy before proposing
        total_size_mb: Total data to deploy in MB
        ram_limit_gb: RAM limit for the node in GB
        sleep_after_minutes: Optional sleep after test for monitoring
    """
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    chunk_size = chunk_size_mb * 1024 * 1024
    total_size = total_size_mb * 1024 * 1024
    num_chunks = total_size // chunk_size
    chunks_per_block = propose_size_mb // chunk_size_mb

    # Log test configuration
    logger.info("=" * 60)
    logger.info("TEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info("  Chunk Size: %d MB", chunk_size_mb)
    logger.info("  Propose Size: %d MB", propose_size_mb)
    logger.info("  Total Size: %d MB", total_size_mb)
    logger.info("  RAM Limit: %d GB", ram_limit_gb)
    logger.info("  Total Chunks: %d", num_chunks)
    logger.info("  Chunks per Block: %d", chunks_per_block)
    logger.info("  Expected Blocks: %d", (num_chunks + chunks_per_block - 1) // chunks_per_block)
    logger.info("=" * 60)

    # Record test params
    record_property("chunk_size_mb", chunk_size_mb)
    record_property("propose_size_mb", propose_size_mb)
    record_property("total_size_mb", total_size_mb)
    record_property("ram_limit_gb", ram_limit_gb)
    record_property("num_chunks", num_chunks)
    record_property("chunks_per_block", chunks_per_block)

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, 
                                        cli_options=node_cli_options, 
                                        mem_limit=f"{ram_limit_gb}G") as bootstrap_node:
        
        block_hashes = []
        valid_after_block_no = 0
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

            binary_data = bytes(j % 256 for j in range(chunk_size))

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                valid_after_block_no=valid_after_block_no,
                parameters={"myBytes": binary_data},
                grpc_options=client_grpc_options,
            )
            
            elapsed = time.time() - start_time
            mb_deployed = (i + 1) * chunk_size_mb
            speed = mb_deployed / (elapsed / 60) if elapsed > 0 else 0
            logger.info("Deployed chunk %d/%d (%.1f%%) - %d MB @ %.1f MB/min", 
                       i + 1, num_chunks, ((i + 1) / num_chunks) * 100, mb_deployed, speed)

            # Propose when we've accumulated enough data
            if (i + 1) % chunks_per_block == 0:
                propose_start = time.time()
                block_hash = bootstrap_node.propose()
                propose_time = time.time() - propose_start
                block_hashes.append(block_hash)
                valid_after_block_no += 1
                logger.info("Proposed block %d with hash %s in %.1fs (%d MB)", 
                           len(block_hashes), block_hash[:16], propose_time, propose_size_mb)

        # Propose any remaining deploys
        if num_chunks % chunks_per_block != 0:
            remaining_mb = (num_chunks % chunks_per_block) * chunk_size_mb
            propose_start = time.time()
            block_hash = bootstrap_node.propose()
            propose_time = time.time() - propose_start
            block_hashes.append(block_hash)
            logger.info("Proposed final block %d with hash %s in %.1fs (%d MB)", 
                       len(block_hashes), block_hash[:16], propose_time, remaining_mb)

        duration = time.time() - start_time
        speed = total_size_mb / (duration / 60)
        
        logger.info("=" * 60)
        logger.info("TEST COMPLETE")
        logger.info("=" * 60)
        logger.info("  Duration: %.1f minutes", duration / 60)
        logger.info("  Speed: %.1f MB/min", speed)
        logger.info("  Blocks: %d", len(block_hashes))
        logger.info("=" * 60)
        
        record_property("duration_seconds", f"{duration:.1f}")
        record_property("duration_minutes", f"{duration/60:.1f}")
        record_property("speed_mb_per_min", f"{speed:.1f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash[:16])
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified: %d deploys OK", idx + 1, len(block_info.deploys))

        if sleep_after_minutes > 0:
            logger.info("Sleeping for %d minutes for monitoring...", sleep_after_minutes)
            time.sleep(sleep_after_minutes * 60)


# pylint: disable=too-many-positional-arguments, too-many-locals
def run_combined_deploy_test(
    command_line_options: CommandLineOptions,
    random_generator: Random,
    docker_client: DockerClient,
    record_property: Any,
    chunk_size_mb: int,
    propose_size_mb: int,
    total_size_mb: int,
    ram_limit_gb: int,
    sleep_after_minutes: int = 0,
) -> None:
    """
    Test runner that combines multiple chunks into a single deploy per propose.
    
    Each deploy contains:
    - Combined Rholang terms joined with '|' operator
    - Data split across 2 deploy parameters (myBytes1, myBytes2)
    
    This tests different I/O patterns: fewer deploys, larger per-deploy data.
    
    Args:
        chunk_size_mb: Size of each logical chunk in MB
        propose_size_mb: Total MB per block (combined into 1 deploy)
        total_size_mb: Total data to deploy in MB
        ram_limit_gb: RAM limit for the node in GB
        sleep_after_minutes: Optional sleep after test for monitoring
    """
    genesis_vault = {
        BOOTSTRAP_NODE_KEYS: 500000000000,
        BONDED_VALIDATOR_KEY_1: 500000000000,
    }

    chunks_per_block = propose_size_mb // chunk_size_mb
    num_blocks = total_size_mb // propose_size_mb

    # Data per param should match the chunk size since we have one param per chunk
    data_per_param = chunk_size_mb * 1024 * 1024

    # Log test configuration
    logger.info("=" * 60)
    logger.info("COMBINED DEPLOY TEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info("  Chunk Size: %d MB", chunk_size_mb)
    logger.info("  Propose Size: %d MB", propose_size_mb)
    logger.info("  Total Size: %d MB", total_size_mb)
    logger.info("  RAM Limit: %d GB", ram_limit_gb)
    logger.info("  Chunks per Block: %d", chunks_per_block)
    logger.info("  Total Blocks (1 deploy each): %d", num_blocks)
    logger.info("  Data per deploy param: %d MB", data_per_param // (1024 * 1024))
    logger.info("=" * 60)

    # Record test params
    record_property("test_type", "combined_deploy")
    record_property("chunk_size_mb", chunk_size_mb)
    record_property("propose_size_mb", propose_size_mb)
    record_property("total_size_mb", total_size_mb)
    record_property("ram_limit_gb", ram_limit_gb)
    record_property("chunks_per_block", chunks_per_block)
    record_property("num_blocks", num_blocks)

    with conftest.testing_context(command_line_options, random_generator, docker_client,
                                  wallets_dict=genesis_vault) as context, \
            ready_bootstrap_with_network(context=context, synchrony_constraint_threshold=0, 
                                        cli_options=node_cli_options, 
                                        mem_limit=f"{ram_limit_gb}G") as bootstrap_node:
        
        block_hashes = []
        valid_after_block_no = 0
        start_time = time.time()
        global_chunk_idx = 0

        for block_idx in range(num_blocks):
            # Build combined Rholang term with all chunks for this block
            # Each chunk writes to a unique channel @N
            import_header = []
            chunk_terms = []
            for local_chunk_idx in range(chunks_per_block):
                chunk_id = global_chunk_idx + local_chunk_idx
                # Alternate between myBytes1 and myBytes2 for each chunk
                import_header.append("myBytes" + str(local_chunk_idx) + "(`rho:deploy:param:myBytes" + str(local_chunk_idx) + "`)")
                chunk_terms.append("@" + str(chunk_id) + "!(*myBytes" + str(local_chunk_idx) + ")")
            
            global_chunk_idx += chunks_per_block
            
            # Combine all terms with Rholang parallel operator
            rholang_term = f"""
                new stdout(`rho:io:stdout`),
                    { ' , '.join(import_header) }
                in {{ 
                   stdout!("deploying {global_chunk_idx} chunks") | {" | ".join(chunk_terms)}
                }}
            """

            print("\n\n\nrholang_term = ", rholang_term)
            print("\n\n\n")

            bootstrap_node.deploy_string(
                rholang_term,
                BOOTSTRAP_NODE_KEYS,
                phlo_limit=100000000000,
                phlo_price=1,
                valid_after_block_no=valid_after_block_no,
                parameters={"myBytes" + str(i): bytes(j % 256 for j in range(data_per_param)) for i in range(chunks_per_block)},
                grpc_options=client_grpc_options,
            )
            
            elapsed = time.time() - start_time
            mb_deployed = (block_idx + 1) * propose_size_mb
            speed = mb_deployed / (elapsed / 60) if elapsed > 0 else 0
            logger.info("Deployed combined block %d/%d (%.1f%%) - %d MB @ %.1f MB/min", 
                       block_idx + 1, num_blocks, ((block_idx + 1) / num_blocks) * 100, 
                       mb_deployed, speed)

            # Propose immediately (1 deploy per propose)
            propose_start = time.time()
            block_hash = bootstrap_node.propose()
            propose_time = time.time() - propose_start
            block_hashes.append(block_hash)
            valid_after_block_no += 1
            logger.info("Proposed block %d with hash %s in %.1fs (%d MB, 1 deploy)", 
                       len(block_hashes), block_hash[:16], propose_time, propose_size_mb)

        duration = time.time() - start_time
        speed = total_size_mb / (duration / 60)
        
        logger.info("=" * 60)
        logger.info("COMBINED DEPLOY TEST COMPLETE")
        logger.info("=" * 60)
        logger.info("  Duration: %.1f minutes", duration / 60)
        logger.info("  Speed: %.1f MB/min", speed)
        logger.info("  Blocks: %d", len(block_hashes))
        logger.info("  Deploys per block: 1")
        logger.info("=" * 60)
        
        record_property("duration_seconds", f"{duration:.1f}")
        record_property("duration_minutes", f"{duration/60:.1f}")
        record_property("speed_mb_per_min", f"{speed:.1f}")
        record_property("num_blocks", len(block_hashes))

        # Verify each block
        for idx, block_hash in enumerate(block_hashes):
            logger.info("Verifying block %d/%d: %s", idx + 1, len(block_hashes), block_hash[:16])
            block_info = bootstrap_node.get_block(block_hash, grpc_options=client_grpc_options)
            assert len(block_info.deploys) == 1, f"Expected 1 deploy per block, got {len(block_info.deploys)}"
            for deploy in block_info.deploys:
                assert not deploy.errored, f"Deploy in block {block_hash} errored"
            logger.info("Block %d verified: 1 combined deploy OK", idx + 1)

        if sleep_after_minutes > 0:
            logger.info("Sleeping for %d minutes for monitoring...", sleep_after_minutes)
            time.sleep(sleep_after_minutes * 60)




# =============================================================================
# 6GB TEST MATRIX - Comprehensive parameter exploration
# Total: 6144MB (6GB), RAM: 17GB
# =============================================================================

# --- 1MB Chunks ---

# def test_matrix_1mb_chunk_128mb_propose_6144mb_total_17g_ram_v2(
#     command_line_options: CommandLineOptions, random_generator: Random,
#     docker_client: DockerClient, record_property: Any
# ) -> None:
#     """6GB test: 1MB chunks, 128MB propose batches (48 blocks)."""
#     run_deploy_test(
#         command_line_options, random_generator, docker_client, record_property,
#         chunk_size_mb=1, propose_size_mb=128, total_size_mb=6144, ram_limit_gb=17
#     )


# =============================================================================
# 6GB COMBINED DEPLOY TEST MATRIX - 1 deploy per propose
# Total: 6144MB (6GB), RAM: 17GB
# Each deploy combines all chunks for that block with | operator
# Data split across 2 deploy parameters (myBytes1, myBytes2)
# =============================================================================

# --- 1MB Chunks (Combined) ---

def test_combined_1mb_chunk_128mb_propose_6144mb_total_17g_ram(
    command_line_options: CommandLineOptions, random_generator: Random,
    docker_client: DockerClient, record_property: Any
) -> None:
    """6GB combined: 1MB chunks, 128MB per deploy (256 terms combined), 24 blocks."""
    run_combined_deploy_test(
        command_line_options, random_generator, docker_client, record_property,
        chunk_size_mb=1, propose_size_mb=128, total_size_mb=6144, ram_limit_gb=17
    )
