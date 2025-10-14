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

/// Convenience return type—including the constructed map and related Rholang metadata.
pub struct PathMapCreationResult {
    pub map: RholangPathMap,
    pub connective_used: bool,
    pub locally_free: Vec<u8>,
}

/// Construct a RholangPathMap from a list of Par elements and an optional remainder.
/// This mirrors what the normalizer does when producing EPathMap from parsed elements.
pub fn create_pathmap_from_elements(elements: &[Par], remainder: Option<Var>) -> PathMapCreationResult {
    let mut map = RholangPathMap::new();
    let mut connective_used = false;
    let mut locally_free = Vec::new();

    for par in elements {
        // Update connective metadata
        if par.connective_used { connective_used = true; }
        locally_free = crate::rust::utils::union(locally_free.clone(), par.locally_free.clone());

        // Convert Par to path (Vec<Vec<u8>>)
        let segments = par_to_path(par);
        // To store in PathMap, flatten path segments into bytes (using a separator byte that can't appear in encoded input, e.g., 0xFF)
        let key: Vec<u8> = segments.into_iter().flat_map(|mut seg| {
            seg.push(0xFF); // separator
            seg
        }).collect();

        map.set_val_at(key, par.clone());
    }

    if remainder.is_some() {
        connective_used = true;
    }

    PathMapCreationResult {
        map,
        connective_used,
        locally_free,
    }
}
