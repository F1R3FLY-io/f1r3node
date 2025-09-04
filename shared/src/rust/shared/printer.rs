// See shared/src/main/scala/coop/rchain/shared/Printer.scala

use std::env;

pub struct Printer;

impl Printer {
    pub fn output_capped() -> Option<i32> {
        match env::var("PRETTY_PRINTER_OUTPUT_TRIM_AFTER") {
            Ok(value) => match value.parse::<i32>() {
                Ok(n) if n >= 0 => Some(n),

                _ => None,
            },

            Err(_) => None,
        }
    }
}
