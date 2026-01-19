from .rnode import (
    Node,
)


def test_repl(started_standalone_bootstrap_node: Node) -> None:
    repl_commands = [
        '5',
        'new s(`rho:io:stdout`) in { s!("foo") }',
        '@"listCh"!([1, 2, 3]) | for(@list <- @"listCh"){ match list { [a, b, c] => { new s(`rho:io:stdout`) in { s!(a) } } } }',
    ]
    for repl_cmd in repl_commands:
        started_standalone_bootstrap_node.repl(repl_cmd)


def test_repl_detects_invalid_rholang(started_standalone_bootstrap_node: Node) -> None:
    input = 'foo'
    output = started_standalone_bootstrap_node.repl(input, stderr=False)
    
    # Scala node format (from ReplGrpcService): "Error: coop.rchain.rholang.interpreter.errors$TopLevelFreeVariablesNotAllowedError"
    # Rust node format: "Error: Top level free variables are not allowed: foo at 0:0"
    # Both are valid, but we need to match the exact format
    scala_format = 'Error: coop.rchain.rholang.interpreter.errors$TopLevelFreeVariablesNotAllowedError'
    rust_format_prefix = 'Error: Top level free variables are not allowed'
    
    # Check for exact match of either format
    assert (
        scala_format in output or
        rust_format_prefix in output
    ), f"Expected error message in Scala format '{scala_format}' or Rust format '{rust_format_prefix}...', got: {repr(output)})"
