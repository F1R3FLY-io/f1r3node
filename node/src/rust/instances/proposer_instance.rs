// See node/src/main/scala/coop/rchain/node/instances/ProposerInstance.scala

use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::BlockMessage;

use casper::rust::blocks::proposer::{
    propose_result::ProposeResult,
    proposer::{ProductionProposer, ProposerResult},
};
use casper::rust::casper::MultiParentCasper;
use casper::rust::errors::CasperError;
use comm::rust::transport::transport_layer::TransportLayer;

pub struct ProposerInstance<T: TransportLayer + Send + Sync + 'static> {
    pub casper: Arc<dyn MultiParentCasper + Send + Sync + 'static>,
    pub propose_requests_queue_rx: mpsc::UnboundedReceiver<(bool, oneshot::Sender<ProposerResult>)>,
    pub proposer: Arc<tokio::sync::Mutex<ProductionProposer<T>>>,
}

impl<T: TransportLayer + Send + Sync + 'static> ProposerInstance<T> {
    pub fn new(
        casper: Arc<dyn MultiParentCasper + Send + Sync + 'static>,
        propose_requests_queue_rx: mpsc::UnboundedReceiver<(bool, oneshot::Sender<ProposerResult>)>,
        proposer: Arc<tokio::sync::Mutex<ProductionProposer<T>>>,
    ) -> Self {
        Self {
            casper,
            propose_requests_queue_rx,
            proposer,
        }
    }

    /// Create and start the proposer stream
    /// Returns a receiver that receives propose results
    pub fn create(
        self,
    ) -> Result<mpsc::UnboundedReceiver<(ProposeResult, Option<BlockMessage>)>, CasperError> {
        // TODO: current implementation differs a little bit from the Scala original one.
        // In the original code while the lock is held on the proposer, all other requests are full-filled with the ProposerEmpty result.
        // And instead of many requests execution which arrived during the lock time, only one request for the proposer is added into the queue.
        // In current implementation all requests which arrived during the lock is waiting for the lock and then executed one by one.

        let (result_tx, result_rx) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            let Self {
                casper,
                mut propose_requests_queue_rx,
                proposer,
            } = self;

            // Process propose requests
            while let Some((is_async, propose_id_sender)) = propose_requests_queue_rx.recv().await {
                let proposer = proposer.clone();
                let result_tx = result_tx.clone();
                let casper = casper.clone();

                let mut proposer = proposer.lock().await;
                let res = proposer.propose(casper, is_async, propose_id_sender).await;

                tracing::info!("Propose started");

                match res {
                    Ok((propose_result, block_opt)) => match block_opt {
                        Some(block) => {
                            let block_str = PrettyPrinter::build_string_block_message(&block, true);

                            tracing::info!(
                                "Propose finished: {:?} Block {} created and added.",
                                propose_result.propose_status,
                                block_str
                            );

                            match result_tx.send((propose_result, Some(block))) {
                                Ok(_) => {}
                                Err(e) => {
                                    tracing::error!("Failed to send propose result: {}", e);
                                }
                            }
                        }
                        None => {
                            tracing::error!("Propose failed: {}", propose_result.propose_status)
                        }
                    },
                    Err(e) => {
                        tracing::error!("Error proposing: {}", e);
                    }
                }
            }

            tracing::info!("Propose requests queue closed, stopping proposer");

            Result::<(), CasperError>::Ok(())
        });

        Ok(result_rx)
    }
}
