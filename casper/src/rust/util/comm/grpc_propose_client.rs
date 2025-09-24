use models::casper::v1::propose_service_client::ProposeServiceClient;
use models::casper::v1::{propose_response, propose_result_response};
use models::casper::{ProposeQuery, ProposeResultQuery};
use tonic::transport::{Channel, Endpoint};

use super::{error_to_vec, ServiceResult};

#[async_trait::async_trait]
pub trait ProposeServiceTrait {
    async fn propose(&mut self, is_async: bool) -> ServiceResult<String>;
    async fn propose_result(&mut self) -> ServiceResult<String>;
}

pub struct ProposeService {
    client: ProposeServiceClient<Channel>,
}

#[async_trait::async_trait]
impl ProposeServiceTrait for ProposeService {
    async fn propose(&mut self, is_async: bool) -> ServiceResult<String> {
        self.propose_impl(is_async).await
    }

    async fn propose_result(&mut self) -> ServiceResult<String> {
        self.propose_result_impl().await
    }
}

impl ProposeService {
    pub async fn new(host: &str, port: u16, max_inbound_bytes: usize) -> eyre::Result<Self> {
        let endpoint = Endpoint::from_shared(format!("http://{host}:{port}"))?;

        let channel = endpoint.connect().await?;
        Ok(Self {
            client: ProposeServiceClient::new(channel).max_decoding_message_size(max_inbound_bytes),
        })
    }

    async fn propose_impl(&mut self, is_async: bool) -> ServiceResult<String> {
        let query = ProposeQuery { is_async };
        let resp = self.client.propose(query).await.map_err(error_to_vec)?;

        match resp.into_inner().message {
            Some(propose_response::Message::Error(err)) => Err(error_to_vec(err)),
            Some(propose_response::Message::Result(s)) => Ok(s),
            None => Err(vec!["empty ProposeResponse.message".to_string()]),
        }
    }

    async fn propose_result_impl(&mut self) -> ServiceResult<String> {
        let query = ProposeResultQuery {};
        let resp = self
            .client
            .propose_result(query)
            .await
            .map_err(error_to_vec)?;

        match resp.into_inner().message {
            Some(propose_result_response::Message::Error(err)) => Err(error_to_vec(err)),
            Some(propose_result_response::Message::Result(s)) => Ok(s),
            None => Err(vec!["empty ProposeResultResponse.message".to_string()]),
        }
    }
}
