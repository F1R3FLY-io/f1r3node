//! REPL gRPC Service implementation
//!
//! This module provides a gRPC service for the REPL (Read-Eval-Print Loop) functionality,
//! allowing clients to execute Rholang code and receive formatted output.

use std::collections::HashMap;
use std::sync::Arc;

use crypto::rust::hash::blake2b512_random::Blake2b512Random;

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

use crate::rust::api::repl_grpc_service::repl::repl_server::Repl;

#[derive(Clone)]
pub struct ReplGrpcServiceImpl {
    runtime: Arc<RhoRuntimeImpl>,
}

impl ReplGrpcServiceImpl {
    pub fn new(runtime: RhoRuntimeImpl) -> Self {
        Self {
            runtime: Arc::new(runtime),
        }
    }

    async fn execute_code(
        &self,
        source: &str,
        print_unmatched_sends_only: bool,
    ) -> eyre::Result<ReplResponse> {
        // TODO: maybe we should move this call to tokio::task::spawn_blocking if the execution will block the task for a long time
        use rholang::rust::interpreter::storage::storage_printer;

        // Match Scala behavior: catch compilation errors and return them as successful responses
        // with "Error: {error}" format, rather than propagating as gRPC errors
        let par = match Compiler::source_to_adt_with_normalizer_env(source, HashMap::new()) {
            Ok(p) => p,
            Err(e) => {
                // Return error as successful response, matching Scala's ReplGrpcService behavior
                // Scala: case _: InterpreterError => Sync[F].delay(s"Error: ${er.toString}")
                let error_msg = format!("Error: {}", e.to_string());
                return Ok(ReplResponse { output: error_msg });
            }
        };

        tokio::task::spawn_blocking(move || print_normalized_term(&par)).await?;

        let rand = Blake2b512Random::create_from_length(10);
        let EvaluateResult { cost, errors, .. } = self
            .runtime
            .evaluate(source, Cost::unsafe_max(), HashMap::new(), rand)
            .await?;

        let pretty_storage = if print_unmatched_sends_only {
            storage_printer::pretty_print_unmatched_sends(&*self.runtime)
        } else {
            storage_printer::pretty_print(&*self.runtime)
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

pub fn create_repl_grpc_service(runtime: RhoRuntimeImpl) -> impl Repl {
    ReplGrpcServiceImpl::new(runtime)
}

#[async_trait::async_trait]
impl Repl for ReplGrpcServiceImpl {
    async fn run(
        &self,
        request: tonic::Request<CmdRequest>,
    ) -> Result<tonic::Response<ReplResponse>, tonic::Status> {
        let cmd_request = request.into_inner();
        let response = self
            .execute_code(&cmd_request.line, false)
            .await
            .map_err(|e| tonic::Status::internal(e.to_string()))?;

        Ok(tonic::Response::new(response))
    }

    async fn eval(
        &self,
        request: tonic::Request<EvalRequest>,
    ) -> Result<tonic::Response<ReplResponse>, tonic::Status> {
        let eval_request = request.into_inner();
        let response = self
            .execute_code(
                &eval_request.program,
                eval_request.print_unmatched_sends_only,
            )
            .await
            .map_err(|e| tonic::Status::internal(e.to_string()))?;
        Ok(tonic::Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::rhoapi::Par;
    use rholang::rust::interpreter::{
        external_services::ExternalServices,
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
            ExternalServices::noop(),
        )
        .await;

        runtime
    }

    #[tokio::test]
    async fn test_repl_service_run() {
        let runtime = create_test_runtime_with_stdout().await;
        let service = ReplGrpcServiceImpl::new(runtime);
        let request = tonic::Request::new(CmdRequest {
            line: "1 + 1".to_string(),
        });

        let result = service.run(request).await;
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        assert!(response.output.contains("Storage Contents"));
    }

    #[tokio::test]
    async fn test_repl_service_eval() {
        let runtime = create_test_runtime_with_stdout().await;
        let service = ReplGrpcServiceImpl::new(runtime);
        let request = tonic::Request::new(EvalRequest {
            program: "1 + 1".to_string(),
            print_unmatched_sends_only: true,
            language: "rho".to_string(),
        });

        let result = service.eval(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.output.contains("Storage Contents"));
    }
}
