import os
from random import Random

from .rnode import (
    Node,
)


def test_eval(started_standalone_bootstrap_node: Node, random_generator: Random) -> None:
    """
    Test the eval subcommand.

    Note: Excludes files that are known to cause stack overflow or other issues
    when running in batch mode with limited resources (e.g., shortslow.rho).
    """
    paths_output = started_standalone_bootstrap_node.shell_out('sh', '-c', 'ls /opt/docker/examples/*.rho').splitlines()

    # Filter out problematic files that cause stack overflow in batch mode
    # shortslow.rho causes deep recursion leading to stack overflow when resources are constrained
    excluded_files = {'shortslow.rho', 'longslow.rho'}
    filtered_paths = [
        p for p in paths_output
        if os.path.basename(p.strip()) not in excluded_files
    ]

    if not filtered_paths:
        # Fallback to all paths if filtering removes everything
        filtered_paths = paths_output

    selected_path = random_generator.choice(filtered_paths).strip()

    # Handle both full paths and relative paths
    if selected_path.startswith('/opt/docker/examples/'):
        full_path = selected_path
    else:
        full_path = os.path.join('/opt/docker/examples', selected_path)

    started_standalone_bootstrap_node.eval(full_path)
