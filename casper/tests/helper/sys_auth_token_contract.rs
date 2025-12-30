// See casper/src/test/scala/coop/rchain/casper/helper/SysAuthTokenContract.scala

use models::rhoapi::{ListParWithRandom, Par};
use models::rust::utils::new_gsys_auth_token_par;
use rholang::rust::interpreter::{
    errors::illegal_argument_error, errors::InterpreterError, system_processes::ProcessContext,
};

use crate::helper::process_context_ext::ProcessContextExt;

pub async fn get(
    ctx: ProcessContext,
    message: (Vec<ListParWithRandom>, bool, Vec<Par>),
) -> Result<Vec<Par>, InterpreterError> {
    let is_contract_call = ctx.contract_call();

    if let Some((produce, _, _, args)) = is_contract_call.unapply(message) {
        match args.as_slice() {
            [ack_channel] => {
                let auth_token = new_gsys_auth_token_par(Vec::new(), false);

                let output = vec![auth_token];
                produce(&output, &ack_channel).await?;
                Ok(output)
            }
            _ => Err(illegal_argument_error("sys_auth_token_make")),
        }
    } else {
        Err(illegal_argument_error("sys_auth_token_make"))
    }
}
