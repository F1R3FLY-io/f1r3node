//! REPL gRPC Service implementation
//!
//! This module provides a gRPC service for the REPL (Read-Eval-Print Loop) functionality,
//! allowing clients to execute Rholang code and receive formatted output.

use std::collections::HashMap;

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use eyre::Result;

/// Protobuf message types for REPL service
pub mod repl {
    tonic::include_proto!("repl");
}

use itertools::Itertools;
use models::rhoapi::Par;
use repl::{CmdRequest, EvalRequest, ReplResponse};
use rholang::rust::interpreter::{
    accounting::costs::Cost,
    compiler::compiler::Compiler,
    interpreter::EvaluateResult,
    pretty_printer::PrettyPrinter,
    rho_runtime::{RhoRuntime, RhoRuntimeImpl},
};
use tracing::error;

/// REPL gRPC Service trait defining the interface for REPL operations
#[allow(async_fn_in_trait)] // implemented as a trait with async functions because of the ISpace dependency inability to be sent between threads
pub trait ReplGrpcService {
    async fn exec(
        &mut self,
        source: String,
        print_unmatched_sends_only: bool,
    ) -> Result<ReplResponse>;

    async fn run(&mut self, request: CmdRequest) -> Result<ReplResponse>;

    async fn eval(&mut self, request: EvalRequest) -> Result<ReplResponse>;
}

pub struct ReplGrpcServiceImpl {
    runtime: RhoRuntimeImpl,
}

impl ReplGrpcServiceImpl {
    pub fn new(runtime: RhoRuntimeImpl) -> Self {
        Self { runtime }
    }

    async fn execute_code(
        &mut self,
        source: &str,
        print_unmatched_sends_only: bool,
    ) -> Result<ReplResponse> {
        // TODO: maybe we should move this call to tokio::task::spawn_blocking if the execution will block the task for a long time
        use rholang::rust::interpreter::storage::storage_printer;
        let par =
            Compiler::source_to_adt_with_normalizer_env(source, HashMap::new()).map_err(|e| {
                error!("Error: {}", e.to_string());
                e
            })?;

        tokio::task::spawn_blocking(move || print_normalized_term(&par)).await?;

        let rand = Blake2b512Random::create_from_length(10);
        let EvaluateResult { cost, errors, .. } = self
            .runtime
            .evaluate(source, Cost::unsafe_max(), HashMap::new(), rand)
            .await?;

        let pretty_storage = if print_unmatched_sends_only {
            storage_printer::pretty_print_unmatched_sends(&self.runtime)
        } else {
            storage_printer::pretty_print(&self.runtime)
        };

        let error_str = if errors.is_empty() {
            String::new()
        } else {
            format!(
                "Errors received during evaluation:\n{}\n",
                errors.into_iter().map(|err| err.to_string()).join("\n")
            )
        };

        let output = format!(
            "Deployment cost: {cost:?}\n
        {error_str}Storage Contents:\n{pretty_storage}",
        );

        Ok(ReplResponse { output })
    }
}

fn print_normalized_term(normalized_term: &Par) {
    println!(
        "\nEvaluating:{}",
        PrettyPrinter::new().build_channel_string(normalized_term)
    );
}

impl ReplGrpcService for ReplGrpcServiceImpl {
    async fn exec(
        &mut self,
        source: String,
        print_unmatched_sends_only: bool,
    ) -> Result<ReplResponse> {
        let output = self
            .execute_code(&source, print_unmatched_sends_only)
            .await?;

        Ok(output)
    }

    async fn run(&mut self, request: CmdRequest) -> Result<ReplResponse> {
        self.exec(request.line, false).await
    }

    async fn eval(&mut self, request: EvalRequest) -> Result<ReplResponse> {
        self.exec(request.program, request.print_unmatched_sends_only)
            .await
    }
}

pub fn create_repl_grpc_service(runtime: RhoRuntimeImpl) -> impl ReplGrpcService {
    ReplGrpcServiceImpl::new(runtime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::rhoapi::Par;
    use rholang::rust::interpreter::{
        matcher::r#match::Matcher, rho_runtime::create_runtime_from_kv_store,
        system_processes::test_framework_contracts,
    };
    use rspace_plus_plus::rspace::shared::{
        in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
    };
    use std::sync::Arc;

    async fn create_test_runtime_with_stdout() -> RhoRuntimeImpl {
        let mut kvm = InMemoryStoreManager::new();
        let store = kvm.r_space_stores().await.unwrap();
        let runtime = create_runtime_from_kv_store(
            store,
            Par::default(),
            true,
            &mut test_framework_contracts(),
            Arc::new(Box::new(Matcher)),
        )
        .await;

        runtime
    }

    #[tokio::test]
    async fn test_repl_service_exec_success() {
        let runtime = create_test_runtime_with_stdout().await;
        let mut service = ReplGrpcServiceImpl::new(runtime);
        let result = service.exec("1 + 1".to_string(), false).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.output.contains("Storage Contents"));
    }

    #[tokio::test]
    async fn test_repl_service_run() {
        let runtime = create_test_runtime_with_stdout().await;
        let mut service = ReplGrpcServiceImpl::new(runtime);
        let request = CmdRequest {
            line: "1 + 1".to_string(),
        };

        let result = service.run(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_repl_service_eval() {
        let runtime = create_test_runtime_with_stdout().await;
        let mut service = ReplGrpcServiceImpl::new(runtime);
        let request = EvalRequest {
            program: "1 + 1".to_string(),
            print_unmatched_sends_only: true,
            language: "rho".to_string(),
        };

        let result = service.eval(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.output.contains("Storage Contents"));
    }
}
