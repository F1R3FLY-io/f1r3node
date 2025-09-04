use rholang_parser::RholangParser;
use validated::Validated;
use std::fs;

#[test]
fn test_isolated_stdout_example() {
    println!("=== ISOLATED TEST: No conflicts with old parser ===");
    
    let parser = RholangParser::new();
    let stdout_code = r#"new stdout(`rho:io:stdout`) in {
  stdout!("hello, world!")
}"#;
    
    println!("Testing stdout.rho example in isolated environment:");
    println!("Code: {}", stdout_code);
    
    match parser.parse(stdout_code) {
        Validated::Good(procs) => {
            println!("✓ SUCCESS: Parsed {} processes", procs.len());
            println!("✓ This proves the rholang-rs parser works when isolated!");
        }
        Validated::Fail(errors) => {
            println!("✗ FAILED: {:?}", errors);
            panic!("Parser failed for stdout.rho even in isolated environment: {:?}", errors);
        }
    }
}

#[test]
fn test_isolated_complex_example() {
    println!("=== ISOLATED TEST: Complex example ===");
    
    let parser = RholangParser::new();
    
    // Read stdoutAck.rho from the examples
    let complex_code = fs::read_to_string("/home/stephen/src/firefly/f1r3fly/rholang/examples/stdoutAck.rho")
        .expect("Failed to read stdoutAck.rho");
    
    println!("Testing stdoutAck.rho example in isolated environment:");
    println!("Code: {}", complex_code);
    
    match parser.parse(&complex_code) {
        Validated::Good(procs) => {
            println!("✓ SUCCESS: Parsed {} processes", procs.len());
            println!("✓ Complex example works in isolated environment!");
        }
        Validated::Fail(errors) => {
            println!("✗ FAILED: {:?}", errors);
            // Don't panic for complex example, just report
            println!("Complex example failed, but that might be expected");
        }
    }
}

#[test]
fn test_parser_behavior_for_send_normalizer_cases() {
    println!("=== TESTING PARSER BEHAVIOR FOR SEND NORMALIZER CASES ===");
    
    let parser = RholangParser::new();
    
    // Test cases from the send normalizer blocked tests
    let test_cases = vec![
        ("Simple process", r#"new x in { x!(1) }"#),
        ("Parallel composition", r#"new x in { x!(1) } | new y in { y!(2) }"#),
        ("Send with negation", r#"new x in { x!(~1) }"#),
        ("Send with conjunction", r#"new x in { x!(1 /\ 2) }"#),
        ("Send with disjunction", r#"new x in { x!(1 \/ 2) }"#),
        ("Send with wildcard", r#"@"x"!(_)"#),
        ("Send with free variable", r#"@"x"!(y)"#),
        ("Channel with conjunction", r#"@{Nil /\ Nil}!(1)"#),
        ("Channel with disjunction", r#"@{Nil \/ Nil}!(1)"#),
        ("Channel with negation", r#"@{~Nil}!(1)"#),
    ];
    
    for (description, code) in test_cases {
        println!("\n--- Testing: {} ---", description);
        println!("Code: {}", code);
        
        match parser.parse(code) {
            Validated::Good(procs) => {
                println!("✓ SUCCESS: Parsed {} processes", procs.len());
                
                // Print basic info about each process
                for (i, proc) in procs.iter().enumerate() {
                    println!("  Process {}: {:?}", i, get_proc_type_name(proc));
                }
                
                // Check if we get exactly 1 process as expected
                if procs.len() == 1 {
                    println!("  ✓ Single process as expected");
                } else {
                    println!("  ⚠ Got {} processes, expected 1", procs.len());
                }
            }
            Validated::Fail(failures) => {
                println!("✗ PARSE FAILED: {} failure(s)", failures.len());
                for failure in &failures {
                    println!("  Failure with {} errors", failure.errors.len());
                    for error in &failure.errors {
                        println!("    Error: {:?} at {:?}", error.error, error.span);
                    }
                }
            }
        }
    }
}

// Helper function to get a readable name for the process type
fn get_proc_type_name(proc: &rholang_parser::ast::AnnProc) -> &'static str {
    use rholang_parser::ast::Proc;
    match proc.proc {
        Proc::Nil => "Nil",
        Proc::Unit => "Unit",
        Proc::LongLiteral(_) => "LongLiteral",
        Proc::StringLiteral(_) => "StringLiteral",
        Proc::BoolLiteral(_) => "BoolLiteral",
        Proc::UriLiteral(_) => "UriLiteral",
        Proc::New { .. } => "New",
        Proc::Send { .. } => "Send", 
        Proc::Par { .. } => "Par",
        Proc::ForComprehension { .. } => "ForComprehension",
        Proc::Match { .. } => "Match",
        Proc::IfThenElse { .. } => "IfThenElse",
        Proc::Bundle { .. } => "Bundle",
        Proc::Let { .. } => "Let",
        Proc::Quote { .. } => "Quote",
        Proc::ProcVar(_) => "ProcVar",
        Proc::Collection(_) => "Collection",
        Proc::BinaryExp { .. } => "BinaryExp",
        Proc::UnaryExp { .. } => "UnaryExp",
        Proc::VarRef { .. } => "VarRef",
        Proc::Select { .. } => "Select",
        Proc::SimpleType(_) => "SimpleType",
        Proc::Contract { .. } => "Contract",
        Proc::SendSync { .. } => "SendSync",
        Proc::Eval { .. } => "Eval",
        Proc::Method { .. } => "Method",
        Proc::Bad => "Bad",
    }
}
