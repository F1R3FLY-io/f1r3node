//! REPL client module
//!

//! REPL client for F1r3fly node
//!
//! This module provides a gRPC client for interacting with the REPL service.

pub mod repl {
    tonic::include_proto!("repl");
}

use futures::future::try_join_all;
use repl::repl_client::ReplClient;
use repl::CmdRequest;
use repl::ReplResponse;

use std::path::Path;
use std::time::Duration;
use tokio::fs;
use tonic::transport::{Channel, Endpoint};
use tonic::Status;

use crate::rust::effects::repl_client::repl::EvalRequest;

/// Trait for REPL client operations
#[async_trait::async_trait]
pub trait ReplClientService {
    /// Run a single line of code
    async fn run(&self, line: String) -> eyre::Result<String>;

    /// Evaluate multiple files
    async fn eval_files(
        &self,
        file_names: Vec<String>,
        print_unmatched_sends_only: bool,
        language: String,
    ) -> eyre::Result<Vec<String>>;

    /// Evaluate a single file
    async fn eval_file(
        &self,
        file_name: String,
        print_unmatched_sends_only: bool,
        language: String,
    ) -> eyre::Result<String>;
}

/// gRPC REPL client implementation
pub struct GrpcReplClient {
    client: ReplClient<Channel>,
}

impl GrpcReplClient {
    /// Create a new gRPC REPL client
    pub async fn new(
        host: String,
        port: u16,
        max_message_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let endpoint = Endpoint::from_shared(format!("http://{host}:{port}"))?
            .connect_timeout(Duration::from_secs(5)) // TODO adjust the connect_timeout if necessary
            .timeout(Duration::from_secs(30)); // TODO adjust the timeout if necessary

        let channel = endpoint.connect().await?;

        Ok(Self {
            client: ReplClient::new(channel).max_decoding_message_size(max_message_size),
        })
    }

    /// Read content from a file
    async fn read_content(file_path: &Path) -> eyre::Result<String> {
        let content = fs::read_to_string(file_path).await?;
        Ok(content)
    }

    /// Process gRPC errors
    fn process_error(error: Status) -> eyre::Report {
        // Extract the root cause if available
        let message = error.message().to_string();
        eyre::Report::new(std::io::Error::new(std::io::ErrorKind::Other, message))
    }
}

#[async_trait::async_trait]
impl ReplClientService for GrpcReplClient {
    async fn run(&self, line: String) -> eyre::Result<String> {
        let req = CmdRequest { line: line.into() };

        // Call the RPC
        match self.client.clone().run(req).await {
            Ok(resp) => {
                let ReplResponse { output } = resp.into_inner();
                Ok(output)
            }
            Err(status) => Err(Self::process_error(status)),
        }
    }

    async fn eval_files(
        &self,
        file_names: Vec<String>,
        print_unmatched_sends_only: bool,
        language: String,
    ) -> eyre::Result<Vec<String>> {
        let res: Vec<String> = try_join_all(file_names.into_iter().map(|file_name| async {
            self.eval_file(file_name, print_unmatched_sends_only, language.clone())
                .await
        }))
        .await?;

        Ok(res)
    }

    async fn eval_file(
        &self,
        file_name: String,
        print_unmatched_sends_only: bool,
        language: String,
    ) -> eyre::Result<String> {
        let file_path = Path::new(&file_name);

        if !file_path.exists() {
            return Err(eyre::Report::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found",
            )));
        }

        let content = Self::read_content(file_path).await?;

        let req = EvalRequest {
            program: content,
            print_unmatched_sends_only,
            language,
        };

        // Call the RPC
        match self.client.clone().eval(req).await {
            Ok(resp) => {
                let ReplResponse { output } = resp.into_inner();
                Ok(output)
            }
            Err(status) => Err(Self::process_error(status)),
        }
    }
}
