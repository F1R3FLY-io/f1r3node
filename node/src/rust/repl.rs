use std::io::IsTerminal;

use colored::Colorize;
use eyre::Result;
use tokio::runtime::Runtime;

use crate::rust::effects::{console_io::ConsoleIO, repl_client::ReplClientService};

/// --- ReplRuntime translation ---
pub struct ReplRuntime;

impl ReplRuntime {
    const LOGO: &'static str = r#"
  ╦═╗┌─┐┬ ┬┌─┐┬┌┐┌  ╔╗╔┌─┐┌┬┐┌─┐  ╦═╗╔═╗╔═╗╦  
  ╠╦╝│  ├─┤├─┤││││  ║║║│ │ ││├┤   ╠╦╝║╣ ╠═╝║  
  ╩╚═└─┘┴ ┴┴ ┴┴┘└┘  ╝╚╝└─┘─┴┘└─┘  ╩╚═╚═╝╩  ╩═╝
"#;

    pub fn new() -> Self {
        Self {}
    }

    /// Scala: def replProgram[F[_]: Monad: ConsoleIO: ReplClient]: F[Boolean]
    /// Rust: returns Ok(true) if loop continued, Ok(false) if terminated (on ":q" or failed run)
    pub fn repl_program<C, R>(&self, rt_handle: &Runtime, console: &mut C, repl: &R) -> Result<bool>
    where
        C: ConsoleIO,
        R: ReplClientService,
    {
        // Show logo (red) only in read mode
        if if_read_mode() {
            console.println_colored(&Self::LOGO.red())?;
        }

        // One iteration body (mirrors Scala's `rep`)
        fn rep<C: ConsoleIO, R: ReplClientService>(
            rt_handle: &Runtime,
            console: &mut C,
            repl: &R,
        ) -> Result<bool> {
            let line = console.read_line()?;
            let line = line.trim().to_string();

            let res = if line.is_empty() {
                console.println_str("")?;
                true // continue
            } else if line == ":q" || line == "exit" {
                false // stop
            } else {
                // run(program): print result; continue only if Right
                match rt_handle.block_on(repl.run(line)) {
                    Ok(output) => {
                        console.println_colored(&output.blue())?;
                        true
                    }
                    Err(err) => {
                        console.println_colored(&format!("Error: {}", err).red())?;
                        false
                    }
                }
            };
            Ok(res)
        }

        // Tail-recursive loop (mirrors Scala's `repl`)
        loop {
            let keep_going = rep(rt_handle, console, repl)?;
            if keep_going {
                continue;
            } else {
                return Ok(false);
            }
        }
    }

    /// Scala: def evalProgram[F[_]: Monad: ReplClient: ConsoleIO](fileNames, printUnmatchedSendsOnly, language): F[Unit]
    pub fn eval_program<C, R>(
        &self,
        rt_handle: &Runtime,
        console: &mut C,
        repl: &R,
        file_names: Vec<String>,
        print_unmatched_sends_only: bool,
        language: String,
    ) -> Result<()>
    where
        C: ConsoleIO,
        R: ReplClientService,
    {
        fn print_result<C: ConsoleIO>(console: &mut C, res: &Result<String>) -> Result<()> {
            match res {
                Ok(s) => console.println_str(s)?,
                Err(e) => {
                    console.println_colored(&format!("Error: {}", e).red())?;
                }
            }
            Ok(())
        }

        fn print_results<C: ConsoleIO>(
            console: &mut C,
            labeled: &[(String, Result<String>)],
        ) -> Result<()> {
            for (file, res) in labeled {
                console.println_str("")?;
                console.println_colored(&format!("Result for {}:", file).blue())?;
                print_result(console, res)?;
            }
            Ok(())
        }

        console.println_str(&format!("Evaluating from {}", file_names.join(", ")))?;

        let results =
            rt_handle.block_on(repl.eval_files(&file_names, print_unmatched_sends_only, language));

        let labeled: Vec<(String, Result<String>)> =
            file_names.into_iter().zip(results.into_iter()).collect();

        print_results(console, &labeled)?;
        Ok(())
    }

    pub fn keywords() -> Vec<&'static str> {
        vec!["stdout", "stdoutack", "stderr", "stderrack", "for", "!!"]
    }
}

fn if_read_mode() -> bool {
    std::io::stdin().is_terminal()
}
