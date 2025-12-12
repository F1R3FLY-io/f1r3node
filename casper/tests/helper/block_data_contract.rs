// See casper/src/test/scala/coop/rchain/casper/helper/BlockDataContract.scala

use crypto::rust::public_key::PublicKey;
use models::rhoapi::{ListParWithRandom, Par};
use rholang::rust::interpreter::{
    errors::illegal_argument_error,
    errors::InterpreterError,
    rho_type::{RhoByteArray, RhoNumber, RhoString},
    system_processes::ProcessContext,
};

use crate::helper::process_context_ext::ProcessContextExt;

pub async fn set(
    ctx: ProcessContext,
    message: (Vec<ListParWithRandom>, bool, Vec<Par>),
) -> Result<Vec<Par>, InterpreterError> {
    let is_contract_call = ctx.contract_call();

    if let Some((produce, _, _, args)) = is_contract_call.unapply(message) {
        match args.as_slice() {
            [key_par, value_par, ack_channel] => {
                if let Some(key) = RhoString::unapply(key_par) {
                    match key.as_str() {
                        "sender" => {
                            if let Some(public_key_bytes) = RhoByteArray::unapply(value_par) {
                                let mut block_data = ctx.block_data.write().await;
                                block_data.sender = PublicKey {
                                    bytes: public_key_bytes.clone().into(),
                                };
                                drop(block_data);

                                let result_par = vec![Par::default()];
                                produce(&result_par, &ack_channel).await?;
                                Ok(result_par)
                            } else {
                                Err(illegal_argument_error("block_data_set"))
                            }
                        }
                        "blockNumber" => {
                            if let Some(block_number) = RhoNumber::unapply(value_par) {
                                let mut block_data = ctx.block_data.write().await;
                                block_data.block_number = block_number;
                                drop(block_data);

                                let result_par = vec![Par::default()];
                                produce(&result_par, &ack_channel).await?;
                                Ok(result_par)
                            } else {
                                Err(illegal_argument_error("block_data_set"))
                            }
                        }
                        _ => Err(illegal_argument_error("block_data_set")),
                    }
                } else {
                    Err(illegal_argument_error("block_data_set"))
                }
            }
            _ => Err(illegal_argument_error("block_data_set")),
        }
    } else {
        Err(illegal_argument_error("block_data_set"))
    }
}
