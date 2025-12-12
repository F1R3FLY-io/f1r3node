// See casper/src/test/scala/coop/rchain/casper/helper/Secp256k1SignContract.scala

use k256::{
    ecdsa::{signature::hazmat::PrehashSigner, Signature, SigningKey},
    elliptic_curve::generic_array::GenericArray,
};
use models::{
    rhoapi::{ListParWithRandom, Par},
    rust::utils::new_gbytearray_par,
};
use rholang::rust::interpreter::{
    errors::illegal_argument_error, errors::InterpreterError, rho_type::RhoByteArray,
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
            [hash_par, sk_par, ack_channel] => {
                if let (Some(hash), Some(secret_key)) = (
                    RhoByteArray::unapply(hash_par),
                    RhoByteArray::unapply(sk_par),
                ) {
                    if secret_key.len() != 32 {
                        return Err(InterpreterError::BugFoundError(format!(
                            "Invalid private key length: must be 32 bytes, got {}",
                            secret_key.len()
                        )));
                    }

                    let key_bytes = GenericArray::clone_from_slice(&secret_key);
                    let signing_key =
                        SigningKey::from_bytes(&key_bytes).expect("Invalid private key");

                    let signature: Signature = signing_key
                        .sign_prehash(&hash)
                        .expect("Failed to sign prehash");
                    let der_bytes = signature.to_der().as_bytes().to_vec();

                    let result_par = new_gbytearray_par(der_bytes, Vec::new(), false);

                    let output = vec![result_par];
                    produce(&output, &ack_channel).await?;
                    Ok(output)
                } else {
                    Err(illegal_argument_error("secp256k1_sign"))
                }
            }
            _ => Err(illegal_argument_error("secp256k1_sign")),
        }
    } else {
        Err(illegal_argument_error("secp256k1_sign"))
    }
}
