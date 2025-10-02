// See rholang/src/main/scala/coop/rchain/rholang/interpreter/RholangCLI.scala

use clap::Parser;
use models::rhoapi::{BindPattern, ListParWithRandom, Par};
use rholang::rust::interpreter::compiler::compiler::Compiler;
use rholang::rust::interpreter::errors::InterpreterError;
use rholang::rust::interpreter::matcher::r#match::Matcher;
use rholang::rust::interpreter::pretty_printer::PrettyPrinter;
use rholang::rust::interpreter::rho_runtime::{create_runtime_from_kv_store, RhoRuntime, RhoRuntimeImpl};
use rholang::rust::interpreter::storage::storage_printer;
use rholang::rust::interpreter::system_processes::Definition;
use rspace_plus_plus::rspace::shared::rspace_store_manager::get_or_create_rspace_store;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;

#[derive(Parser, Debug)]
#[command(name = "rholang")]
#[command(version = "0.2")]
#[command(about = "Rholang Mercury 0.2\nOptions:", long_about = None)]
struct Conf {
    /// outputs binary protobuf serialization
    #[arg(long)]
    binary: bool,

    /// outputs textual protobuf serialization
    #[arg(long)]
    text: bool,

    /// don't print tuplespace after evaluation
    #[arg(long)]
    quiet: bool,

    /// only print unmatched sends after evaluation
    #[arg(long = "unmatched-sends-only")]
    unmatched_sends_only: bool,

    /// Path to data directory
    #[arg(long = "data-dir")]
    data_dir: Option<PathBuf>,

    /// Map size (in bytes)
    #[arg(long = "map-size", default_value = "1073741824")]
    map_size: usize,

    /// Rholang source file(s)
    files: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut conf = Conf::parse();

    let data_dir = conf
        .data_dir
        .take()
        .unwrap_or_else(|| tempfile::tempdir().unwrap().path().to_path_buf());

    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    
    runtime.block_on(async move {
        let stores = get_or_create_rspace_store(&data_dir.to_string_lossy(), conf.map_size)?;
        let matcher_impl = Matcher::default();
        let matcher: Arc<Box<dyn rspace_plus_plus::rspace::r#match::Match<BindPattern, ListParWithRandom>>> = 
            Arc::new(Box::new(matcher_impl));
        let mut additional_system_processes: Vec<Definition> = vec![];

        let mut rho_runtime = create_runtime_from_kv_store(
            stores,
            Par::default(),
            true,
            &mut additional_system_processes,
            matcher,
        )
        .await;

        let result = if !conf.files.is_empty() {
            let mut problems = Vec::new();
            for file in &conf.files {
                match process_file(
                    &conf,
                    &mut rho_runtime,
                    file,
                    conf.quiet,
                    conf.unmatched_sends_only,
                )
                .await
                {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("error in: {}", file);
                        match error_or_bug(&e) {
                            ErrorKind::UserError => eprintln!("{}", e),
                            ErrorKind::Bug => eprintln!("{:?}", e),
                        }
                        problems.push((file.clone(), e));
                    }
                }
            }
            if !problems.is_empty() {
                std::process::exit(1);
            }
            Ok(())
        } else {
            repl(&mut rho_runtime).await
        };

        result
    })
}

enum ErrorKind {
    UserError,
    Bug,
}

fn error_or_bug(err: &InterpreterError) -> ErrorKind {
    match err {
        InterpreterError::LexerError(_) => ErrorKind::UserError,
        InterpreterError::SyntaxError(_) => ErrorKind::UserError,
        InterpreterError::NormalizerError(_) => ErrorKind::UserError,
        InterpreterError::TopLevelFreeVariablesNotAllowedError(_) => ErrorKind::UserError,
        InterpreterError::TopLevelLogicalConnectivesNotAllowedError(_) => ErrorKind::UserError,
        InterpreterError::TopLevelWildcardsNotAllowedError(_) => ErrorKind::UserError,
        InterpreterError::UnexpectedReuseOfNameContextFree { .. } => ErrorKind::UserError,
        InterpreterError::UnexpectedBundleContent(_) => ErrorKind::UserError,
        InterpreterError::UnexpectedNameContext { .. } => ErrorKind::UserError,
        InterpreterError::UnexpectedProcContext { .. } => ErrorKind::UserError,
        InterpreterError::UnrecognizedNormalizerError(_) => ErrorKind::UserError,
        _ => ErrorKind::Bug,
    }
}

fn print_prompt() {
    print!("\nrholang> ");
    io::stdout().flush().unwrap();
}

fn print_normalized_term(normalized_term: &Par) {
    println!("\nEvaluating rhocli:");
    let mut printer = PrettyPrinter::new();
    println!("{}", printer.build_string_from_message(normalized_term));
}

fn print_storage_contents(
    runtime: &RhoRuntimeImpl,
    unmatched_sends_only: bool,
) {
    println!("\nStorage Contents:");
    let output = if unmatched_sends_only {
        storage_printer::pretty_print_unmatched_sends(runtime)
    } else {
        storage_printer::pretty_print(runtime)
    };
    println!("{}", output);
}

fn print_cost(cost: &rholang::rust::interpreter::accounting::costs::Cost) {
    println!("Estimated deploy cost: {:?}", cost);
}

fn print_errors(errors: &[InterpreterError]) {
    if !errors.is_empty() {
        println!("Errors received during evaluation:");
        for error in errors {
            println!("{}", error);
        }
    }
}

async fn repl(
    runtime: &mut RhoRuntimeImpl,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        print_prompt();
        let mut line = String::new();
        match io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => {
                println!("\nExiting...");
                return Ok(());
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Err(e) = evaluate(runtime, trimmed).await {
                    eprintln!("Error: {}", e);
                }
            }
        }
    }
}

async fn evaluate(
    runtime: &mut impl RhoRuntime,
    source: &str,
) -> Result<(), InterpreterError> {
    let result = runtime.evaluate_with_term(source).await?;
    
    if !result.errors.is_empty() {
        for error in &result.errors {
            match error {
                InterpreterError::LexerError(_)
                | InterpreterError::SyntaxError(_)
                | InterpreterError::NormalizerError(_)
                | InterpreterError::TopLevelFreeVariablesNotAllowedError(_)
                | InterpreterError::TopLevelLogicalConnectivesNotAllowedError(_)
                | InterpreterError::TopLevelWildcardsNotAllowedError(_)
                | InterpreterError::UnexpectedReuseOfNameContextFree { .. }
                | InterpreterError::UnexpectedBundleContent(_)
                | InterpreterError::UnexpectedNameContext { .. }
                | InterpreterError::UnexpectedProcContext { .. }
                | InterpreterError::UnrecognizedNormalizerError(_) => {
                    eprint!("{}", error);
                }
                _ => {
                    eprintln!("{:?}", error);
                }
            }
        }
    }
    
    Ok(())
}

async fn process_file(
    conf: &Conf,
    runtime: &mut RhoRuntimeImpl,
    file_name: &str,
    quiet: bool,
    unmatched_sends_only: bool,
) -> Result<(), InterpreterError> {
    let source = fs::read_to_string(file_name).await
        .map_err(|e| InterpreterError::BugFoundError(format!("Failed to read file: {}", e)))?;

    if conf.binary {
        write_binary(file_name, &source).await?;
    } else if conf.text {
        write_human_readable(file_name, &source).await?;
    } else {
        evaluate_par(runtime, &source, quiet, unmatched_sends_only).await?;
    }

    Ok(())
}

async fn write_human_readable(file_name: &str, source: &str) -> Result<(), InterpreterError> {
    let sorted_term = Compiler::source_to_adt(source)?;
    let compiled_file_name = file_name.replace(".rho", "") + ".rhoc";
    
    fs::write(&compiled_file_name, format!("{:?}", sorted_term)).await
        .map_err(|e| InterpreterError::BugFoundError(format!("Failed to write file: {}", e)))?;
    
    println!("Compiled {} to {}", file_name, compiled_file_name);
    Ok(())
}

async fn write_binary(file_name: &str, source: &str) -> Result<(), InterpreterError> {
    let sorted_term = Compiler::source_to_adt(source)?;
    let binary_file_name = file_name.replace(".rho", "") + ".bin";
    
    let bytes = bincode::serialize(&sorted_term)
        .map_err(|e| InterpreterError::BugFoundError(format!("Failed to serialize: {}", e)))?;
    
    fs::write(&binary_file_name, bytes).await
        .map_err(|e| InterpreterError::BugFoundError(format!("Failed to write file: {}", e)))?;
    
    println!("Compiled {} to {}", file_name, binary_file_name);
    Ok(())
}

async fn evaluate_par(
    runtime: &mut RhoRuntimeImpl,
    source: &str,
    quiet: bool,
    unmatched_sends_only: bool,
) -> Result<(), InterpreterError> {
    let par = Compiler::source_to_adt(source)?;
    
    if !quiet {
        print_normalized_term(&par);
    }
    
    let result = runtime.evaluate_with_term(source).await?;
    
    print_cost(&result.cost);
    print_errors(&result.errors);
    
    if !quiet {
        print_storage_contents(runtime, unmatched_sends_only);
    }
    
    Ok(())
}