//! Full round-trip decoder: SExpr → Par.
//!
//! This is the inverse of the encoding path: `par_to_sexpr` → `parse_sexpr` → `SExpr::encode`.
//! The reverse is: `SExpr::decode(bytes)` → `sexpr_to_par(&SExpr)`.
//!
//! Symbol nodes contain the raw string representations from `par_to_sexpr`
//! (including quotes, brackets, etc.). This decoder pattern-matches on those
//! to reconstruct the original Par values.

use crate::rhoapi::expr::ExprInstance;
use crate::rhoapi::var::VarInstance;
use crate::rhoapi::{
    Bundle, EDiv, EList, EMethod, EMinus, EMult, ENeg, ENot, EPlus, ETuple, EVar, Expr,
    New, Par, Receive, ReceiveBind, Send, Var,
};
use crate::rust::path_map_encoder::SExpr;
use crate::rust::pathmap_integration::{parse_sexpr, split_sexpr};
use crate::rust::par_set::ParSet;
use crate::rust::par_set_type_mapper::ParSetTypeMapper;
use crate::rust::par_map::ParMap;
use crate::rust::par_map_type_mapper::ParMapTypeMapper;

pub struct SExprToPar;

impl SExprToPar {
    /// Decode a single encoded byte segment back to a Par.
    pub fn decode_segment(bytes: &[u8]) -> Result<Par, String> {
        let sexpr = SExpr::decode(bytes)?;
        Self::sexpr_to_par(&sexpr)
    }

    /// Convert an SExpr tree back to a Par.
    pub fn sexpr_to_par(sexpr: &SExpr) -> Result<Par, String> {
        match sexpr {
            SExpr::Symbol(s) => Self::symbol_to_par(s),
            SExpr::List(children) => Self::tagged_list_to_par(children),
        }
    }

    /// Decode a symbol string back to the Par it represents.
    /// Dispatch order is optimized for common cases.
    fn symbol_to_par(s: &str) -> Result<Par, String> {
        // Quoted strings: "..."
        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
            let inner = &s[1..s.len() - 1];
            return Ok(Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::GString(inner.to_string())),
                }],
                ..Default::default()
            });
        }

        // Nil → empty Par
        if s == "Nil" {
            return Ok(Par::default());
        }

        // Boolean literals
        if s == "true" {
            return Ok(Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::GBool(true)),
                }],
                ..Default::default()
            });
        }
        if s == "false" {
            return Ok(Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::GBool(false)),
                }],
                ..Default::default()
            });
        }

        // Wildcard: "_"
        if s == "_" {
            return Ok(Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::EVarBody(EVar {
                        v: Some(Var {
                            var_instance: Some(VarInstance::Wildcard(
                                crate::rhoapi::var::WildcardMsg {},
                            )),
                        }),
                    })),
                }],
                ..Default::default()
            });
        }

        // Bound variable: _N (where N is an integer)
        if s.starts_with('_') && s.len() > 1 {
            if let Ok(n) = s[1..].parse::<i32>() {
                return Ok(Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::EVarBody(EVar {
                            v: Some(Var {
                                var_instance: Some(VarInstance::BoundVar(n)),
                            }),
                        })),
                    }],
                    ..Default::default()
                });
            }
        }

        // Free variable: $N (where N is an integer)
        if s.starts_with('$') && s.len() > 1 {
            if let Ok(n) = s[1..].parse::<i32>() {
                return Ok(Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::EVarBody(EVar {
                            v: Some(Var {
                                var_instance: Some(VarInstance::FreeVar(n)),
                            }),
                        })),
                    }],
                    connective_used: true,
                    ..Default::default()
                });
            }
        }

        // NewVar marker: bare "$"
        if s == "$" {
            return Ok(Par::default());
        }

        // URI: `...`
        if s.starts_with('`') && s.ends_with('`') && s.len() >= 2 {
            let inner = &s[1..s.len() - 1];
            return Ok(Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::GUri(inner.to_string())),
                }],
                ..Default::default()
            });
        }

        // Hex byte array: 0x...
        if s.starts_with("0x") {
            if let Ok(bytes) = hex::decode(&s[2..]) {
                return Ok(Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GByteArray(bytes)),
                    }],
                    ..Default::default()
                });
            }
        }

        // Integer literal
        if let Ok(i) = s.parse::<i64>() {
            return Ok(Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::GInt(i)),
                }],
                ..Default::default()
            });
        }

        // List literal: [e1 e2 ...]
        if s.starts_with('[') && s.ends_with(']') {
            return Self::bracket_list_to_par(&s[1..s.len() - 1]);
        }

        // Fallback: treat unknown symbols as GString
        Ok(Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.to_string())),
            }],
            ..Default::default()
        })
    }

    /// Decode a tagged S-expression list: (tag child1 child2 ...)
    fn tagged_list_to_par(children: &[SExpr]) -> Result<Par, String> {
        match children.first() {
            Some(SExpr::Symbol(tag)) => match tag.as_str() {
                "par" => {
                    // Parallel composition: merge children[1..] into a combined Par
                    let mut combined = Par::default();
                    for child in &children[1..] {
                        let child_par = Self::sexpr_to_par(child)?;
                        combined = combined.append(child_par);
                    }
                    Ok(combined)
                }

                "!" => {
                    // Send: (! chan data1 data2 ...)
                    if children.len() < 2 {
                        return Err("Send requires at least a channel".to_string());
                    }
                    let chan = Self::sexpr_to_par(&children[1])?;
                    let data: Vec<Par> = children[2..]
                        .iter()
                        .map(|c| Self::sexpr_to_par(c))
                        .collect::<Result<_, _>>()?;
                    Ok(Par {
                        sends: vec![Send {
                            chan: Some(chan),
                            data,
                            persistent: false,
                            locally_free: Vec::new(),
                            connective_used: false,
                        }],
                        ..Default::default()
                    })
                }

                "for" => {
                    // Receive: (for (binds...) body)
                    if children.len() < 3 {
                        return Err("Receive requires binds and body".to_string());
                    }
                    let binds = match &children[1] {
                        SExpr::List(bind_list) => bind_list
                            .iter()
                            .map(|b| match b {
                                SExpr::List(parts) if parts.len() >= 3 => {
                                    let source = Self::sexpr_to_par(parts.last().expect(
                                        "Receive bind should have source as last element",
                                    ))?;
                                    Ok(ReceiveBind {
                                        patterns: Vec::new(),
                                        source: Some(source),
                                        remainder: None,
                                        free_count: 0,
                                    })
                                }
                                _ => Err("Invalid bind format".to_string()),
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                        _ => return Err("Binds must be a list".to_string()),
                    };
                    let body = Self::sexpr_to_par(&children[2])?;
                    Ok(Par {
                        receives: vec![Receive {
                            binds,
                            body: Some(body),
                            persistent: false,
                            peek: false,
                            bind_count: 0,
                            locally_free: Vec::new(),
                            connective_used: false,
                        }],
                        ..Default::default()
                    })
                }

                "new" => {
                    // New: (new x0 x1 ... body)
                    if children.len() < 3 {
                        return Err("New requires at least one binding and a body".to_string());
                    }
                    let bind_count = (children.len() - 2) as i32;
                    let body =
                        Self::sexpr_to_par(children.last().expect("New should have a body"))?;
                    Ok(Par {
                        news: vec![New {
                            bind_count,
                            p: Some(body),
                            uri: Vec::new(),
                            injections: std::collections::BTreeMap::new(),
                            locally_free: Vec::new(),
                        }],
                        ..Default::default()
                    })
                }

                "bundle" => {
                    // Bundle: (bundle body)
                    if children.len() < 2 {
                        return Err("Bundle requires a body".to_string());
                    }
                    let body = Self::sexpr_to_par(&children[1])?;
                    Ok(Par {
                        bundles: vec![Bundle {
                            body: Some(body),
                            write_flag: false,
                            read_flag: false,
                        }],
                        ..Default::default()
                    })
                }

                "tuple" => {
                    // Tuple: (tuple e1 e2 ...)
                    let elements: Vec<Par> = children[1..]
                        .iter()
                        .map(|c| Self::sexpr_to_par(c))
                        .collect::<Result<_, _>>()?;
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::ETupleBody(ETuple {
                                ps: elements,
                                locally_free: Vec::new(),
                                connective_used: false,
                            })),
                        }],
                        ..Default::default()
                    })
                }

                "set" => {
                    // Set: (set e1 e2 ...)
                    let elements: Vec<Par> = children[1..]
                        .iter()
                        .map(|c| Self::sexpr_to_par(c))
                        .collect::<Result<_, _>>()?;
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::ESetBody(
                                ParSetTypeMapper::par_set_to_eset(ParSet::new(
                                    elements,
                                    false,
                                    Vec::new(),
                                    None,
                                )),
                            )),
                        }],
                        ..Default::default()
                    })
                }

                "map" => {
                    // Map: (map (k1 : v1) (k2 : v2) ...)
                    let mut kvs = Vec::new();
                    for child in &children[1..] {
                        match child {
                            SExpr::List(pair_parts) if pair_parts.len() == 3 => {
                                let key = Self::sexpr_to_par(&pair_parts[0])?;
                                // pair_parts[1] is the ":" symbol
                                let value = Self::sexpr_to_par(&pair_parts[2])?;
                                kvs.push((key, value));
                            }
                            _ => return Err("Map entries must be (key : value)".to_string()),
                        }
                    }
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EMapBody(
                                ParMapTypeMapper::par_map_to_emap(ParMap::new(
                                    kvs,
                                    false,
                                    Vec::new(),
                                    None,
                                )),
                            )),
                        }],
                        ..Default::default()
                    })
                }

                "not" => {
                    // ENot: (not expr)
                    if children.len() != 2 {
                        return Err("not requires exactly one argument".to_string());
                    }
                    let p = Self::sexpr_to_par(&children[1])?;
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::ENotBody(ENot { p: Some(p) })),
                        }],
                        ..Default::default()
                    })
                }

                "*" => {
                    // EMult: (* lhs rhs)
                    if children.len() != 3 {
                        return Err("* requires exactly two arguments".to_string());
                    }
                    let p1 = Self::sexpr_to_par(&children[1])?;
                    let p2 = Self::sexpr_to_par(&children[2])?;
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EMultBody(EMult {
                                p1: Some(p1),
                                p2: Some(p2),
                            })),
                        }],
                        ..Default::default()
                    })
                }

                "/" => {
                    // EDiv: (/ lhs rhs)
                    if children.len() != 3 {
                        return Err("/ requires exactly two arguments".to_string());
                    }
                    let p1 = Self::sexpr_to_par(&children[1])?;
                    let p2 = Self::sexpr_to_par(&children[2])?;
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EDivBody(EDiv {
                                p1: Some(p1),
                                p2: Some(p2),
                            })),
                        }],
                        ..Default::default()
                    })
                }

                "+" => {
                    // EPlus: (+ lhs rhs)
                    if children.len() != 3 {
                        return Err("+ requires exactly two arguments".to_string());
                    }
                    let p1 = Self::sexpr_to_par(&children[1])?;
                    let p2 = Self::sexpr_to_par(&children[2])?;
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EPlusBody(EPlus {
                                p1: Some(p1),
                                p2: Some(p2),
                            })),
                        }],
                        ..Default::default()
                    })
                }

                "-" if children.len() == 2 => {
                    // ENeg (unary): (- expr)
                    let p = Self::sexpr_to_par(&children[1])?;
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::ENegBody(ENeg { p: Some(p) })),
                        }],
                        ..Default::default()
                    })
                }

                "-" if children.len() == 3 => {
                    // EMinus (binary): (- lhs rhs)
                    let p1 = Self::sexpr_to_par(&children[1])?;
                    let p2 = Self::sexpr_to_par(&children[2])?;
                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EMinusBody(EMinus {
                                p1: Some(p1),
                                p2: Some(p2),
                            })),
                        }],
                        ..Default::default()
                    })
                }

                // Method call: target.method(args) → encoded as (target.method arg1 arg2 ...)
                other if other.contains('.') => {
                    let dot_pos = other
                        .rfind('.')
                        .expect("Method tag should contain a dot");
                    let target_str = &other[..dot_pos];
                    let method_name = &other[dot_pos + 1..];

                    let target_sexpr = parse_sexpr(target_str);
                    let target = Self::sexpr_to_par(&target_sexpr)?;
                    let arguments: Vec<Par> = children[1..]
                        .iter()
                        .map(|c| Self::sexpr_to_par(c))
                        .collect::<Result<_, _>>()?;

                    Ok(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                                method_name: method_name.to_string(),
                                target: Some(target),
                                arguments,
                                locally_free: Vec::new(),
                                connective_used: false,
                            })),
                        }],
                        ..Default::default()
                    })
                }

                other => Err(format!("Unknown S-expression tag: {}", other)),
            },
            Some(SExpr::List(_)) => {
                // Nested list as the head — shouldn't occur in normal encoding
                Err("Expected tagged S-expression with symbol head".to_string())
            }
            None => {
                // Empty list → empty Par
                Ok(Par::default())
            }
        }
    }

    /// Parse a bracket-list literal: [e1 e2 ...] stored as a single symbol by parse_sexpr.
    fn bracket_list_to_par(inner: &str) -> Result<Par, String> {
        let inner = inner.trim();
        if inner.is_empty() {
            return Ok(Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::EListBody(EList {
                        ps: Vec::new(),
                        locally_free: Vec::new(),
                        connective_used: false,
                        remainder: None,
                    })),
                }],
                ..Default::default()
            });
        }

        let parts = split_sexpr(inner);
        let elements: Vec<Par> = parts
            .iter()
            .map(|p| {
                let sx = parse_sexpr(p);
                Self::sexpr_to_par(&sx)
            })
            .collect::<Result<_, _>>()?;

        Ok(Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EListBody(EList {
                    ps: elements,
                    locally_free: Vec::new(),
                    connective_used: false,
                    remainder: None,
                })),
            }],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::par_to_sexpr::ParToSExpr;
    use crate::rust::pathmap_integration::parse_sexpr as parse;

    fn round_trip(par: &Par) -> Par {
        let sexpr_str = ParToSExpr::par_to_sexpr(par);
        let sexpr = parse(&sexpr_str);
        let encoded = sexpr.encode();
        SExprToPar::decode_segment(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode segment: {}", e))
    }

    #[test]
    fn test_round_trip_gint() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GInt(42)),
            }],
            ..Default::default()
        };
        let decoded = round_trip(&par);
        assert_eq!(decoded.exprs[0].expr_instance, par.exprs[0].expr_instance);
    }

    #[test]
    fn test_round_trip_gstring() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString("hello".to_string())),
            }],
            ..Default::default()
        };
        let decoded = round_trip(&par);
        assert_eq!(decoded.exprs[0].expr_instance, par.exprs[0].expr_instance);
    }

    #[test]
    fn test_round_trip_gbool_true() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GBool(true)),
            }],
            ..Default::default()
        };
        let decoded = round_trip(&par);
        assert_eq!(decoded.exprs[0].expr_instance, par.exprs[0].expr_instance);
    }

    #[test]
    fn test_round_trip_gbool_false() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GBool(false)),
            }],
            ..Default::default()
        };
        let decoded = round_trip(&par);
        assert_eq!(decoded.exprs[0].expr_instance, par.exprs[0].expr_instance);
    }

    #[test]
    fn test_round_trip_nil() {
        let par = Par::default();
        let decoded = round_trip(&par);
        assert!(decoded.is_nil());
    }

    #[test]
    fn test_round_trip_guri() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GUri("rho:io:stdout".to_string())),
            }],
            ..Default::default()
        };
        let decoded = round_trip(&par);
        assert_eq!(decoded.exprs[0].expr_instance, par.exprs[0].expr_instance);
    }

    #[test]
    fn test_round_trip_list() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EListBody(EList {
                    ps: vec![
                        Par {
                            exprs: vec![Expr {
                                expr_instance: Some(ExprInstance::GString("a".to_string())),
                            }],
                            ..Default::default()
                        },
                        Par {
                            exprs: vec![Expr {
                                expr_instance: Some(ExprInstance::GInt(1)),
                            }],
                            ..Default::default()
                        },
                    ],
                    locally_free: Vec::new(),
                    connective_used: false,
                    remainder: None,
                })),
            }],
            ..Default::default()
        };
        let decoded = round_trip(&par);
        match (&decoded.exprs[0].expr_instance, &par.exprs[0].expr_instance) {
            (
                Some(ExprInstance::EListBody(dec_list)),
                Some(ExprInstance::EListBody(orig_list)),
            ) => {
                assert_eq!(dec_list.ps.len(), orig_list.ps.len());
                assert_eq!(
                    dec_list.ps[0].exprs[0].expr_instance,
                    orig_list.ps[0].exprs[0].expr_instance
                );
                assert_eq!(
                    dec_list.ps[1].exprs[0].expr_instance,
                    orig_list.ps[1].exprs[0].expr_instance
                );
            }
            _ => panic!("Expected EListBody"),
        }
    }
}
