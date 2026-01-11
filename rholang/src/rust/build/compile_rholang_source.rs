// See rholang/src/main/scala/coop/rchain/rholang/build/CompileRholangSource.scala

use models::rhoapi::Par;
use std::collections::HashMap;
use std::fs;

use crate::rust::interpreter::compiler::compiler::Compiler;
use crate::rust::interpreter::errors::InterpreterError;

/** TODO: Currently all calls to this class use empty environment. See [[NormalizerEnv]]. - OLD */
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
        // TODO: Remove clone
        let term = Compiler::source_to_adt_with_normalizer_env(&code, normalizer_env.clone())?;

        Ok(CompiledRholangSource {
            code,
            normalizer_env,
            path,
            term,
        })
    }

    pub fn load_source(filepath: &str) -> Result<String, InterpreterError> {
        // Try multiple possible resource locations
        let possible_paths = [
            format!("casper/src/main/resources/{}", filepath),
            format!("casper/src/test/resources/{}", filepath), //TODO should we moved it from Scala to Rust [tests] folder?
            format!("src/main/resources/{}", filepath),
            format!("src/test/resources/{}", filepath),
            format!("../casper/src/main/resources/{}", filepath),
            format!("../casper/src/test/resources/{}", filepath),
        ];

        for path in &possible_paths {
            if let Ok(content) = fs::read_to_string(path) {
                tracing::debug!(
                    "Loaded from resource file <<{}>> at path: {}",
                    filepath,
                    path
                );
                return Ok(content);
            }
        }

        // If all paths fail, return the original error with the first path
        Err(InterpreterError::from(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Could not find resource file '{}' in any of the expected locations: {:?}",
                filepath, possible_paths
            ),
        )))
    }

    pub fn apply(classpath: &str) -> Result<CompiledRholangSource, InterpreterError> {
        Self::apply_with_env(classpath, HashMap::new())
    }

    pub fn apply_with_env(
        classpath: &str,
        env: HashMap<String, Par>,
    ) -> Result<CompiledRholangSource, InterpreterError> {
        let code = Self::load_source(classpath)?;
        Ok(CompiledRholangSource::new(
            code,
            env,
            classpath.to_string(),
        )?)
    }
}

/**
 * Loads code from a resource file while doing macro substitution with the provided values.
 * The macros have the format $$macro$$
 * @param classpath
 * @param env a sequence of pairs macro -> value
 * @return
 */
pub struct CompiledRholangTemplate {
    pub classpath: String,
    pub normalizer_env: HashMap<String, Par>,
    pub path: String,
}

impl CompiledRholangTemplate {
    pub fn new(
        classpath: &str,
        normalizer_env: HashMap<String, Par>,
        macros: &[(&str, &str)],
    ) -> CompiledRholangSource {
        let code = Self::load_template(classpath, macros).expect("Failed to load template");
        CompiledRholangSource::new(code, normalizer_env, classpath.to_string())
            .expect("Failed to compile template")
    }

    pub fn load_template(
        classpath: &str,
        macros: &[(&str, &str)],
    ) -> Result<String, InterpreterError> {
        // Try multiple possible resource locations
        let possible_paths = [
            format!("casper/src/main/resources/{}", classpath),
            format!("src/main/resources/{}", classpath),
            format!("../casper/src/main/resources/{}", classpath),
        ];

        let mut original_content = None;
        for path in &possible_paths {
            if let Ok(content) = fs::read_to_string(path) {
                original_content = Some(content);
                break;
            }
        }

        let original_content = original_content.ok_or_else(|| {
            InterpreterError::from(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Could not find template file '{}' in any of the expected locations: {:?}",
                    classpath, possible_paths
                ),
            ))
        })?;

        let final_content = macros
            .iter()
            .fold(original_content, |content, (name, value)| {
                content.replace(&format!("$${}$$", name), value)
            });

        tracing::debug!(
            "Loaded from resource file <<{}>>{}",
            classpath,
            final_content
        );
        Ok(final_content)
    }
}
