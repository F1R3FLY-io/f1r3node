// console_io.rs
use colored::{ColoredString, Colorize};
use eyre::Result;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::MemHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Context, Editor, Helper};
use std::collections::HashSet;

pub fn keywords() -> Vec<&'static str> {
    vec!["stdout", "stdoutack", "stderr", "stderrack", "for", "!!"]
}

/// ===== 1) Trait: mirrors Scala ConsoleIO[F] exactly in semantics =====
pub trait ConsoleIO {
    fn read_line(&mut self) -> Result<String>;
    fn read_password(&mut self, prompt: &str) -> Result<String>;

    fn println_str(&mut self, s: &str) -> Result<()>;
    fn println_colored(&mut self, s: &ColoredString) -> Result<()>;

    /// Replace completion candidates with the given history set
    fn update_completion(&mut self, history: &HashSet<String>) -> Result<()>;

    fn close(&mut self) -> Result<()>;
}

/// Convenience overloads mirroring Scala's println overloading
pub trait ConsolePrintExt {
    fn println(&mut self, s: &str) -> Result<()>;
    fn printlnc(&mut self, s: &ColoredString) -> Result<()>;
}
impl<T: ConsoleIO + ?Sized> ConsolePrintExt for T {
    fn println(&mut self, s: &str) -> Result<()> {
        self.println_str(s)
    }
    fn printlnc(&mut self, s: &ColoredString) -> Result<()> {
        self.println_colored(s)
    }
}

/// ===== 2) NOP implementation (Scala: NOPConsoleIO[F]) =====
pub struct NopConsoleIO;
impl ConsoleIO for NopConsoleIO {
    fn read_line(&mut self) -> Result<String> {
        Ok(String::new())
    }
    fn read_password(&mut self, _prompt: &str) -> Result<String> {
        Ok(String::new())
    }
    fn println_str(&mut self, _s: &str) -> Result<()> {
        Ok(())
    }
    fn println_colored(&mut self, _s: &ColoredString) -> Result<()> {
        Ok(())
    }
    fn update_completion(&mut self, _history: &HashSet<String>) -> Result<()> {
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// ===== 3) JLine-equivalent using rustyline =====xs
#[derive(Clone, Default)]
struct StringsHelper {
    // Fixed keywords (like ReplRuntime.keywords)
    keywords: Vec<String>,
}
impl StringsHelper {
    fn new(keywords: Vec<String>) -> Self {
        Self { keywords }
    }
    fn set_keywords(&mut self, words: Vec<String>) {
        self.keywords = words;
    }
}
impl Helper for StringsHelper {}
impl Highlighter for StringsHelper {}
impl Hinter for StringsHelper {
    type Hint = String;
}
impl Validator for StringsHelper {
    fn validate(&self, _: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        Ok(ValidationResult::Valid(None))
    }
}
impl Completer for StringsHelper {
    type Candidate = Pair;
    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (start, stem) = current_token(line, pos);
        let mut out = Vec::new();
        for w in self.keywords.iter() {
            if w.starts_with(stem) {
                out.push(Pair {
                    display: w.clone(),
                    replacement: w.clone(),
                });
            }
        }
        Ok((start, out))
    }
}

fn current_token(line: &str, pos: usize) -> (usize, &str) {
    let bytes = line.as_bytes();
    let mut i = pos;
    while i > 0 && !bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    (i, &line[i..pos])
}

/// RustConsoleIO mimics:
/// - history enabled
/// - prompt colored only when `read_mode == true`
/// - StringsCompleter(history) replacement on update_completion
pub struct RustConsoleIO {
    rl: Editor<StringsHelper, MemHistory>,
    prompt: String, // e.g. "rholang $ "
}

impl RustConsoleIO {
    pub fn new(keywords: Vec<String>, prompt: impl Into<String>) -> Result<Self> {
        let mut rl: Editor<StringsHelper, MemHistory> =
            Editor::with_history(rustyline::Config::default(), MemHistory::new())?;
        rl.set_helper(Some(StringsHelper::new(keywords)));

        Ok(Self {
            rl,
            prompt: prompt.into(),
        })
    }

    #[inline]
    fn helper_mut(&mut self) -> &mut StringsHelper {
        self.rl.helper_mut().expect("helper is set")
    }
}

impl ConsoleIO for RustConsoleIO {
    fn read_line(&mut self) -> Result<String> {
        let p = self.prompt.green().to_string();

        let line = self.rl.readline(&p)?;
        if !line.trim().is_empty() {
            self.rl.add_history_entry(line.as_str())?;
        }
        Ok(line)
    }

    fn read_password(&mut self, prompt: &str) -> Result<String> {
        // Equivalent to console.readLine(prompt, '*')
        // Prints the prompt and reads without echoing
        let pwd = rpassword::prompt_password(prompt)?;
        Ok(pwd)
    }

    fn println_str(&mut self, s: &str) -> Result<()> {
        println!("{s}");
        Ok(())
    }

    fn println_colored(&mut self, s: &ColoredString) -> Result<()> {
        println!("{s}");

        Ok(())
    }

    fn update_completion(&mut self, history: &HashSet<String>) -> Result<()> {
        // Replace completer contents like:
        //   console.getCompleters.foreach(remove); addCompleter(StringsCompleter(history))
        let mut words: Vec<String> = history.iter().cloned().collect();
        words.sort();
        self.helper_mut().set_keywords(words);
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        // Rustyline restores terminal modes automatically on drop.
        Ok(())
    }
}

/// Factory (like your `consoleIO[F: Sync]`)
pub fn console_io() -> Result<RustConsoleIO> {
    let ks: Vec<String> = keywords().into_iter().map(Into::into).collect();
    let inst = RustConsoleIO::new(ks, "rholang $ ")?;
    Ok(inst)
}
