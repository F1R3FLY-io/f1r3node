// See rholang/src/main/scala/coop/rchain/rholang/build/CompileRholangSource.scala
//
// Source-only constructor for compiled Rholang. The previous filesystem
// fallback (`apply`, `apply_with_env`, `load_source`, file-based template
// loading) was removed: production callers pass the source code directly
// — the genesis ceremony embeds its `.rho` / `.rhox` files at compile
// time via `include_str!`. This makes the binary self-contained and
// eliminates the CWD-relative resource-path search.

use models::rhoapi::Par;
use std::collections::HashMap;

use crate::rust::interpreter::compiler::compiler::Compiler;
use crate::rust::interpreter::errors::InterpreterError;

pub struct CompiledRholangSource {
    pub code: String,
    pub normalizer_env: HashMap<String, Par>,
    pub path: String,
    pub term: Par,
}

impl CompiledRholangSource {
    pub fn new(
        code: String,
        normalizer_env: HashMap<String, Par>,
        path: String,
    ) -> Result<Self, InterpreterError> {
        let term = Compiler::source_to_adt_with_normalizer_env(&code, normalizer_env.clone())?;

        Ok(CompiledRholangSource {
            code,
            normalizer_env,
            path,
            term,
        })
    }
}

/// Compiles a Rholang template, performing `$$macro$$` substitution against
/// the provided pairs before parsing. Callers pass the raw template body
/// (typically from an `include_str!` constant) — there is no filesystem
/// resolution.
pub struct CompiledRholangTemplate;

impl CompiledRholangTemplate {
    pub fn new(
        name: &str,
        template: &str,
        normalizer_env: HashMap<String, Par>,
        macros: &[(&str, &str)],
    ) -> CompiledRholangSource {
        let final_content = macros
            .iter()
            .fold(template.to_string(), |content, (key, value)| {
                content.replace(&format!("$${}$$", key), value)
            });

        CompiledRholangSource::new(final_content, normalizer_env, name.to_string())
            .expect("Failed to compile template")
    }
}
