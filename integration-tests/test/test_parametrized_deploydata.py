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
                myBytes(`rho:deploy:param:myBytes`),
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
