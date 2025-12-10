// See casper/src/test/scala/coop/rchain/casper/helper/DeployerIdContract.scala

use models::rhoapi::{ListParWithRandom, Par};
use rholang::rust::interpreter::{
    errors::illegal_argument_error,
    errors::InterpreterError,
    rho_type::{RhoByteArray, RhoDeployerId, RhoString},
    system_processes::ProcessContext,
};

use crate::helper::process_context_ext::ProcessContextExt;

pub async fn get(
    ctx: ProcessContext,
    message: (Vec<ListParWithRandom>, bool, Vec<Par>),
) -> Result<Vec<Par>, InterpreterError> {
    let is_contract_call = ctx.contract_call();

    if let Some((produce, _, _, args)) = is_contract_call.unapply(message) {
        match args.as_slice() {
            [deployer_id_par, key_par, ack_channel] => {
                if let (Some(deployer_id_str), Some(public_key)) = (
                    RhoString::unapply(deployer_id_par),
                    RhoByteArray::unapply(key_par),
                ) {
                    if deployer_id_str == "deployerId" {
                        let output = vec![RhoDeployerId::create_par(public_key)];
                        produce(&output, &ack_channel).await?;
                        Ok(output)
                    } else {
                        Err(illegal_argument_error("deployer_id_make"))
                    }
                } else {
                    Err(illegal_argument_error("deployer_id_make"))
                }
            }
            _ => Err(illegal_argument_error("deployer_id_make")),
        }
    } else {
        Err(illegal_argument_error("deployer_id_make"))
    }
}
