// See casper/src/test/scala/coop/rchain/casper/helper/CasperInvalidBlocksContract.scala

use models::rhoapi::{ListParWithRandom, Par};
use rholang::rust::interpreter::{
    errors::illegal_argument_error, errors::InterpreterError, system_processes::ProcessContext,
};

use crate::helper::process_context_ext::ProcessContextExt;

pub async fn set(
    ctx: ProcessContext,
    message: (Vec<ListParWithRandom>, bool, Vec<Par>),
) -> Result<Vec<Par>, InterpreterError> {
    let is_contract_call = ctx.contract_call();

    if let Some((produce, _, _, args)) = is_contract_call.unapply(message) {
        match args.as_slice() {
            [new_invalid_blocks_par, ack_channel] => {
                let mut invalid_blocks_lock = ctx.invalid_blocks.invalid_blocks.write().await;
                *invalid_blocks_lock = new_invalid_blocks_par.clone();

                let result_par = vec![Par::default()];
                produce(&result_par, &ack_channel).await?;
                Ok(result_par)
            }
            _ => Err(illegal_argument_error("casper_invalid_blocks_set")),
        }
    } else {
        Err(illegal_argument_error("casper_invalid_blocks_set"))
    }
}
