// See casper/src/test/scala/coop/rchain/casper/helper/RhoLoggerContract.scala

use models::rhoapi::{ListParWithRandom, Par};
use rholang::rust::interpreter::{
    errors::illegal_argument_error, errors::InterpreterError, pretty_printer::PrettyPrinter,
    rho_type::RhoString, system_processes::ProcessContext,
};

use crate::helper::process_context_ext::ProcessContextExt;

pub async fn handle_message(
    ctx: ProcessContext,
    message: (Vec<ListParWithRandom>, bool, Vec<Par>),
) -> Result<Vec<Par>, InterpreterError> {
    let is_contract_call = ctx.contract_call();

    if let Some((_, _, _, args)) = is_contract_call.unapply(message) {
        match args.as_slice() {
            [log_level_par, par] => {
                if let Some(log_level) = RhoString::unapply(log_level_par) {
                    let mut pretty_printer = PrettyPrinter::new();
                    let msg = pretty_printer.build_string_from_message(par);

                    match log_level.as_str() {
                        "trace" => {
                            println!("trace: {}", msg);
                            Ok(vec![])
                        }
                        "debug" => {
                            println!("debug: {}", msg);
                            Ok(vec![])
                        }
                        "info" => {
                            println!("info: {}", msg);
                            Ok(vec![])
                        }
                        "warn" => {
                            println!("warn: {}", msg);
                            Ok(vec![])
                        }
                        "error" => {
                            println!("error: {}", msg);
                            Ok(vec![])
                        }
                        _ => Err(illegal_argument_error("rho_logger")),
                    }
                } else {
                    Err(illegal_argument_error("rho_logger"))
                }
            }
            _ => Err(illegal_argument_error("rho_logger")),
        }
    } else {
        Err(illegal_argument_error("rho_logger"))
    }
}
