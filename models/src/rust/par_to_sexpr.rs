// Converts Rholang Par AST to S-expression string representation
// This is used to generate deterministic path keys for PathMap

use crate::rhoapi::expr::ExprInstance;
use crate::rhoapi::var::VarInstance;
use crate::rhoapi::{Bundle, Expr, New, Par, Receive, Send};

// TODO -> it feels like this entire file should be replaced with just having From<Par> for SExpr

pub struct ParToSExpr;

impl ParToSExpr {
    /// Convert a Par to an S-expression string
    pub fn par_to_sexpr(par: &Par) -> String {
        let mut parts = Vec::new();

        // Process sends
        for send in &par.sends {
            parts.push(Self::send_to_sexpr(send));
        }

        // Process receives
        for receive in &par.receives {
            parts.push(Self::receive_to_sexpr(receive));
        }

        // Process news
        for new in &par.news {
            parts.push(Self::new_to_sexpr(new));
        }

        // Process expressions
        for expr in &par.exprs {
            parts.push(Self::expr_to_sexpr(expr));
        }

        // Process bundles
        for bundle in &par.bundles {
            parts.push(Self::bundle_to_sexpr(bundle));
        }

        // If multiple parts, wrap in a sequence
        if parts.is_empty() {
            String::from("Nil")
        } else if parts.len() == 1 {
            parts[0].clone()
        } else {
            format!("(par {})", parts.join(" "))
        }
    }

    fn send_to_sexpr(send: &Send) -> String {
        let chan = send
            .chan
            .as_ref()
            .map(|c| Self::par_to_sexpr(c))
            .unwrap_or_else(|| "Nil".to_string());
        let data: Vec<String> = send.data.iter().map(|d| Self::par_to_sexpr(d)).collect();
        format!("(! {} {})", chan, data.join(" "))
    }

    fn receive_to_sexpr(receive: &Receive) -> String {
        let binds: Vec<String> = receive
            .binds
            .iter()
            .map(|b| {
                let source = b
                    .source
                    .as_ref()
                    .map(|s| Self::par_to_sexpr(s))
                    .unwrap_or_else(|| "Nil".to_string());
                format!("(bind <- {})", source)
            })
            .collect();
        let body = receive
            .body
            .as_ref()
            .map(|b| Self::par_to_sexpr(b))
            .unwrap_or_else(|| "Nil".to_string());
        format!("(for ({}) {})", binds.join(" "), body)
    }

    fn new_to_sexpr(new: &New) -> String {
        let vars: Vec<String> = (0..new.bind_count).map(|i| format!("x{}", i)).collect();
        let body = new
            .p
            .as_ref()
            .map(|p| Self::par_to_sexpr(p))
            .unwrap_or_else(|| "Nil".to_string());
        format!("(new {} {})", vars.join(" "), body)
    }

    fn bundle_to_sexpr(bundle: &Bundle) -> String {
        let body = bundle
            .body
            .as_ref()
            .map(|b| Self::par_to_sexpr(b))
            .unwrap_or_else(|| "Nil".to_string());
        format!("(bundle {})", body)
    }

    fn expr_to_sexpr(expr: &Expr) -> String {
        match &expr.expr_instance {
            Some(expr_instance) => match expr_instance {
                ExprInstance::GBool(b) => format!("{}", b),
                ExprInstance::GInt(i) => format!("{}", i),
                ExprInstance::GString(s) => format!("\"{}\"", s),
                ExprInstance::GUri(u) => format!("`{}`", u),
                ExprInstance::GByteArray(ba) => format!("0x{}", hex::encode(ba)),

                ExprInstance::EListBody(list) => {
                    let elements: Vec<String> =
                        list.ps.iter().map(|p| Self::par_to_sexpr(p)).collect();
                    format!("[{}]", elements.join(" "))
                }

                ExprInstance::ETupleBody(tuple) => {
                    let elements: Vec<String> =
                        tuple.ps.iter().map(|p| Self::par_to_sexpr(p)).collect();
                    format!("(tuple {})", elements.join(" "))
                }

                ExprInstance::ESetBody(set) => {
                    let elements: Vec<String> =
                        set.ps.iter().map(|p| Self::par_to_sexpr(p)).collect();
                    format!("(set {})", elements.join(" "))
                }

                ExprInstance::EMapBody(map) => {
                    let pairs: Vec<String> = map
                        .kvs
                        .iter()
                        .map(|kv| {
                            format!(
                                "({} : {})",
                                Self::par_to_sexpr(kv.key.as_ref().unwrap_or(&Par::default())),
                                Self::par_to_sexpr(kv.value.as_ref().unwrap_or(&Par::default()))
                            )
                        })
                        .collect();
                    format!("(map {})", pairs.join(" "))
                }

                ExprInstance::EVarBody(evar) => {
                    if let Some(var) = &evar.v {
                        match &var.var_instance {
                            Some(VarInstance::BoundVar(bv)) => format!("_{}", bv),
                            Some(VarInstance::FreeVar(fv)) => format!("${}", fv),
                            Some(VarInstance::Wildcard(_)) => String::from("_"),
                            None => String::from("var"),
                        }
                    } else {
                        String::from("var")
                    }
                }

                ExprInstance::ENegBody(eneg) => {
                    format!(
                        "(- {})",
                        Self::par_to_sexpr(eneg.p.as_ref().unwrap_or(&Par::default()))
                    )
                }

                ExprInstance::ENotBody(enot) => {
                    format!(
                        "(not {})",
                        Self::par_to_sexpr(enot.p.as_ref().unwrap_or(&Par::default()))
                    )
                }

                ExprInstance::EMultBody(emult) => {
                    format!(
                        "(* {} {})",
                        Self::par_to_sexpr(emult.p1.as_ref().unwrap_or(&Par::default())),
                        Self::par_to_sexpr(emult.p2.as_ref().unwrap_or(&Par::default()))
                    )
                }

                ExprInstance::EDivBody(ediv) => {
                    format!(
                        "(/ {} {})",
                        Self::par_to_sexpr(ediv.p1.as_ref().unwrap_or(&Par::default())),
                        Self::par_to_sexpr(ediv.p2.as_ref().unwrap_or(&Par::default()))
                    )
                }

                ExprInstance::EPlusBody(eplus) => {
                    format!(
                        "(+ {} {})",
                        Self::par_to_sexpr(eplus.p1.as_ref().unwrap_or(&Par::default())),
                        Self::par_to_sexpr(eplus.p2.as_ref().unwrap_or(&Par::default()))
                    )
                }

                ExprInstance::EMinusBody(eminus) => {
                    format!(
                        "(- {} {})",
                        Self::par_to_sexpr(eminus.p1.as_ref().unwrap_or(&Par::default())),
                        Self::par_to_sexpr(eminus.p2.as_ref().unwrap_or(&Par::default()))
                    )
                }

                ExprInstance::EMethodBody(method) => {
                    let target =
                        Self::par_to_sexpr(method.target.as_ref().unwrap_or(&Par::default()));
                    let args: Vec<String> = method
                        .arguments
                        .iter()
                        .map(|a| Self::par_to_sexpr(a))
                        .collect();
                    format!("({}.{} {})", target, method.method_name, args.join(" "))
                }

                _ => String::from("(expr)"),
            },
            None => String::from("Nil"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_int() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GInt(42)),
            }],
            ..Default::default()
        };
        assert_eq!(ParToSExpr::par_to_sexpr(&par), "42");
    }

    #[test]
    fn test_simple_string() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString("hello".to_string())),
            }],
            ..Default::default()
        };
        assert_eq!(ParToSExpr::par_to_sexpr(&par), "\"hello\"");
    }

    #[test]
    fn test_list() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EListBody(crate::rhoapi::EList {
                    ps: vec![
                        Par {
                            exprs: vec![Expr {
                                expr_instance: Some(ExprInstance::GString("a".to_string())),
                            }],
                            ..Default::default()
                        },
                        Par {
                            exprs: vec![Expr {
                                expr_instance: Some(ExprInstance::GString("b".to_string())),
                            }],
                            ..Default::default()
                        },
                    ],
                    locally_free: vec![],
                    connective_used: false,
                    remainder: None,
                })),
            }],
            ..Default::default()
        };
        assert_eq!(ParToSExpr::par_to_sexpr(&par), "[\"a\" \"b\"]");
    }
}
