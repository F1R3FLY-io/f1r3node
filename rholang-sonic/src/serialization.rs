//! RGB state serialization module
//!
//! This module handles conversion between RGB state values and Rholang-compatible representations,
//! including channel naming and state value serialization with proper Rholang Par types.

use amplify_num::hex::ToHex;
use hypersonic::{CellAddr, StateValue};
use models::rhoapi::{Par, Expr, EList};
use models::rhoapi::expr::ExprInstance;

use crate::RholangCodex;

impl<S: hypersonic::Stock> RholangCodex<S> {
    /// Convert RGB cell address to Rholang channel name
    pub(crate) fn cell_to_channel_name(&self, cell_addr: CellAddr) -> String {
        format!("rgb_cell_{}_{}", cell_addr.opid.to_hex(), cell_addr.pos)
    }

    /// Convert RGB StateValue to proper Rholang Par object
    /// This enables native Rholang processing instead of string-based serialization
    pub(crate) fn state_value_to_par(&self, value: StateValue) -> Par {
        match value {
            StateValue::None => {
                // Represent None as Nil - empty Par with no expressions
                Par::default()
            }
            StateValue::Single { first } => {
                // Single field element as byte array Par
                let bytes = first.to_u256().to_le_bytes().to_vec();
                self.create_byte_array_par(bytes)
            }
            StateValue::Double { first, second } => {
                // Double as list of two byte arrays
                let first_bytes = first.to_u256().to_le_bytes().to_vec();
                let second_bytes = second.to_u256().to_le_bytes().to_vec();
                self.create_tuple_par(vec![
                    self.create_byte_array_par(first_bytes),
                    self.create_byte_array_par(second_bytes),
                ])
            }
            StateValue::Triple {
                first,
                second,
                third,
            } => {
                // Triple as list of three byte arrays
                let first_bytes = first.to_u256().to_le_bytes().to_vec();
                let second_bytes = second.to_u256().to_le_bytes().to_vec();
                let third_bytes = third.to_u256().to_le_bytes().to_vec();
                self.create_tuple_par(vec![
                    self.create_byte_array_par(first_bytes),
                    self.create_byte_array_par(second_bytes),
                    self.create_byte_array_par(third_bytes),
                ])
            }
            StateValue::Quadruple {
                first,
                second,
                third,
                fourth,
            } => {
                // Quadruple as list of four byte arrays
                let first_bytes = first.to_u256().to_le_bytes().to_vec();
                let second_bytes = second.to_u256().to_le_bytes().to_vec();
                let third_bytes = third.to_u256().to_le_bytes().to_vec();
                let fourth_bytes = fourth.to_u256().to_le_bytes().to_vec();
                self.create_tuple_par(vec![
                    self.create_byte_array_par(first_bytes),
                    self.create_byte_array_par(second_bytes),
                    self.create_byte_array_par(third_bytes),
                    self.create_byte_array_par(fourth_bytes),
                ])
            }
        }
    }

    /// Legacy method that serializes StateValue to string for backward compatibility
    /// Uses the new Par-based conversion internally for consistency
    #[deprecated(note = "Use state_value_to_par() for proper Rholang integration")]
    pub(crate) fn serialize_state_value(&self, value: StateValue) -> String {
        // Convert to Par first, then serialize to string representation
        let par = self.state_value_to_par(value);
        self.par_to_string(&par)
    }

    /// Create a Rholang Par object containing a byte array
    fn create_byte_array_par(&self, bytes: Vec<u8>) -> Par {
        // Create a byte array expression
        let expr = Expr {
            expr_instance: Some(ExprInstance::GByteArray(bytes)),
        };
        Par::default().with_exprs(vec![expr])
    }

    /// Create a Rholang Par object representing a tuple/list of Par objects
    fn create_tuple_par(&self, elements: Vec<Par>) -> Par {
        // Create a list expression containing all elements
        let expr = Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: elements,
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        };
        Par::default().with_exprs(vec![expr])
    }

    /// Convert a Par object to string representation (for debugging/legacy support)
    fn par_to_string(&self, par: &Par) -> String {
        // For now, provide a simple string representation
        // In production, this could use proper Rholang pretty-printing
        if par.exprs.is_empty() {
            "Nil".to_string()
        } else if par.exprs.len() == 1 {
            match &par.exprs[0].expr_instance {
                Some(ExprInstance::GByteArray(bytes)) => {
                    format!("ByteArray({})", bytes.to_hex())
                }
                Some(ExprInstance::EListBody(list)) => {
                    let elements: Vec<String> = list.ps
                        .iter()
                        .map(|p| self.par_to_string(p))
                        .collect();
                    format!("[{}]", elements.join(", "))
                }
                _ => format!("Par({:?})", par),
            }
        } else {
            format!("MultiPar({})", par.exprs.len())
        }
    }
}
