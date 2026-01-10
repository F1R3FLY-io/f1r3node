// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/TestUtil.scala
// This is the ORIGINAL SCALA MAIN CODE implementation (not the modified version currently in main)

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::Par;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use rholang::rust::interpreter::accounting::costs::Cost;
use rholang::rust::interpreter::compiler::compiler::Compiler;
use rholang::rust::interpreter::env::Env;
use rholang::rust::interpreter::errors::InterpreterError;
use rholang::rust::interpreter::rho_runtime::RhoRuntime;
use std::collections::HashMap;

pub struct TestUtil;

impl TestUtil {
    pub async fn eval_source<R: RhoRuntime>(
        source: &CompiledRholangSource,
        runtime: &R,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        Self::eval(&source.code, runtime, source.normalizer_env.clone(), rand).await
    }

    pub async fn eval<R: RhoRuntime>(
        code: &str,
        runtime: &R,
        normalizer_env: HashMap<String, Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        let term = Compiler::source_to_adt_with_normalizer_env(code, normalizer_env)?;
        Self::eval_term(term, runtime, rand).await
    }

    async fn eval_term<R: RhoRuntime>(
        term: Par,
        runtime: &R,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        runtime.cost().set(Cost::unsafe_max());
        runtime.inj(term, Env::new(), rand).await
    }
}
