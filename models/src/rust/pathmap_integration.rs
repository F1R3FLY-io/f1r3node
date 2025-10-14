//! Integration layer between Rholang Par types and the PathMap crate.

use pathmap::PathMap;
use crate::rhoapi::{Par, Var};

/// Type alias for our standard use case: PathMap from bytes to Rholang Par.
pub type RholangPathMap = PathMap<Par>;

use crate::rust::path_map_encoder::SExpr;
use crate::rust::par_to_sexpr::ParToSExpr;
use crate::rhoapi::expr::ExprInstance;

/// Convert a Par element into a path consisting of byte segments.
/// - Lists are interpreted as multi-segment paths.
/// - Non-list Pars are encoded as a single S-expression path segment.
pub fn par_to_path(par: &Par) -> Vec<Vec<u8>> {
    // If Par is a list, convert each inner element to a segment
    if let Some(path_segments) = extract_list_path(par) {
        return path_segments;
    }
    // Otherwise: treat as a single segment
    let sexpr_string = ParToSExpr::par_to_sexpr(par);
    let sexpr = parse_sexpr(&sexpr_string);
    vec![sexpr.encode()]
}

fn extract_list_path(par: &Par) -> Option<Vec<Vec<u8>>> {
    if par.exprs.len() == 1 {
        if let Some(ExprInstance::EListBody(list)) = &par.exprs[0].expr_instance {
            let segments: Vec<Vec<u8>> = list.ps.iter().map(|p| {
                let sexpr_string = ParToSExpr::par_to_sexpr(p);
                let sexpr = parse_sexpr(&sexpr_string);
                sexpr.encode()
            }).collect();
            return Some(segments);
        }
    }
    None
}

// Basic SExpr parser for structure encoding (copy your existing logic if complex)
fn parse_sexpr(s: &str) -> SExpr {
    let s = s.trim();
    if !s.starts_with('(') {
        return SExpr::Symbol(s.to_string());
    }
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len()-1];
        let parts = split_sexpr(inner);
        let children: Vec<SExpr> = parts.iter().map(|p| parse_sexpr(p)).collect();
        return SExpr::List(children);
    }
    SExpr::Symbol(s.to_string())
}

fn split_sexpr(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;
    for ch in s.chars() {
        if escape {
            current.push(ch);
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '(' if !in_string => { depth += 1; current.push(ch); }
            ')' if !in_string => { depth -= 1; current.push(ch); }
            ' ' | '\t' | '\n' if !in_string && depth == 0 => {
                if !current.is_empty() {
                    parts.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}
