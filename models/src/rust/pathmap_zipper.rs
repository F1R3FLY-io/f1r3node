//! Zipper wrapper types for integrating PathMap zippers with Rholang Par types.
//!
//! This module provides wrapper types that bridge PathMap's zipper API with Rholang's process-oriented
//! data model. Operations work on Par values as the unit of operation rather than raw bytes.

use pathmap::zipper::{ReadZipperUntracked, WriteZipperUntracked, ZipperHead};
use crate::rhoapi::{Par, EPathMap};
use super::pathmap_integration::{par_to_path, RholangPathMap};

/// Wrapper for PathMap ReadZipper that maintains Rholang context
pub struct RholangReadZipper<'a, 'path> {
    pub(crate) zipper: ReadZipperUntracked<'a, 'path, Par>,
    pub(crate) connective_used: bool,
    pub(crate) locally_free: Vec<u8>,
}

impl<'a, 'path> RholangReadZipper<'a, 'path> {
    /// Create a new read zipper from a PathMap at root
    pub fn new(map: &'a RholangPathMap, connective_used: bool, locally_free: Vec<u8>) -> Self {
        RholangReadZipper {
            zipper: map.read_zipper(),
            connective_used,
            locally_free,
        }
    }

    /// Create a new read zipper at a specific path
    pub fn new_at_path(
        map: &'a RholangPathMap,
        path: &Par,
        connective_used: bool,
        locally_free: Vec<u8>,
    ) -> Result<RholangReadZipper<'a, 'static>, String> {
        let segments = par_to_path(path);
        let key = flatten_segments(&segments);
        // Use the owned version since we can't return a reference to local key
        Ok(RholangReadZipper {
            zipper: map.read_zipper_at_path(key),
            connective_used,
            locally_free,
        })
    }

    /// Descend to a path specified as a Par (list of segments)
    pub fn descend_to(&mut self, path: &Par) -> Result<(), String> {
        use pathmap::zipper::ZipperMoving;
        
        let segments = par_to_path(path);
        let key = flatten_segments(&segments);
        self.zipper.descend_to(&key);
        Ok(())
    }

    /// Get the value at the current position
    pub fn get_val(&self) -> Option<&Par> {
        use pathmap::zipper::ZipperValues;
        self.zipper.val()
    }

    /// Check if there's a value at current position
    pub fn has_val(&self) -> bool {
        use pathmap::zipper::Zipper;
        self.zipper.is_val()
    }

    /// Check if the current path exists
    pub fn path_exists(&self) -> bool {
        use pathmap::zipper::Zipper;
        self.zipper.path_exists()
    }

    /// Convert zipper to Par representation
    /// This creates a special Par that represents the zipper state
    pub fn to_par(&self) -> Par {
        // For now, we'll represent the zipper as a special PathMap
        // In a full implementation, we'd need a custom Expr type for zippers
        // We'll create an empty PathMap as a placeholder since we can't easily
        // extract the underlying PathMap from the zipper
        let empty_pathmap = EPathMap {
            ps: vec![],
            locally_free: self.locally_free.clone(),
            connective_used: self.connective_used,
            remainder: None,
        };
        
        // Create a special Par that represents a read zipper
        // We'll use a special marker to identify it as a zipper
        Par::default().with_exprs(vec![crate::rhoapi::Expr {
            expr_instance: Some(crate::rhoapi::expr::ExprInstance::EPathmapBody(empty_pathmap)),
        }])
    }
}

/// Wrapper for PathMap WriteZipper that maintains Rholang context
pub struct RholangWriteZipper<'a, 'path> {
    pub(crate) zipper: WriteZipperUntracked<'a, 'path, Par>,
    #[allow(dead_code)]
    pub(crate) connective_used: bool,
    #[allow(dead_code)]
    pub(crate) locally_free: Vec<u8>,
}

impl<'a, 'path> RholangWriteZipper<'a, 'path> {
    /// Create a new write zipper from a PathMap at root
    pub fn new(map: &'a mut RholangPathMap, connective_used: bool, locally_free: Vec<u8>) -> Self {
        RholangWriteZipper {
            zipper: map.write_zipper(),
            connective_used,
            locally_free,
        }
    }

    /// Create a new write zipper at a specific path
    pub fn new_at_path(
        map: &'a mut RholangPathMap,
        path: &Par,
        connective_used: bool,
        locally_free: Vec<u8>,
    ) -> Result<Self, String> {
        use pathmap::zipper::ZipperMoving;
        
        let segments = par_to_path(path);
        let key = flatten_segments(&segments);
        // Create a write zipper at the constructed path
        let mut zipper = map.write_zipper();
        zipper.descend_to(&key);
        Ok(RholangWriteZipper {
            zipper,
            connective_used,
            locally_free,
        })
    }

    /// Descend to a path specified as a Par (list of segments)
    pub fn descend_to(&mut self, path: &Par) -> Result<(), String> {
        use pathmap::zipper::ZipperMoving;
        
        let segments = par_to_path(path);
        let key = flatten_segments(&segments);
        self.zipper.descend_to(&key);
        Ok(())
    }

    /// Set the value at the current position
    pub fn set_val(&mut self, value: Par) -> Option<Par> {
        use pathmap::zipper::ZipperWriting;
        self.zipper.set_val(value)
    }

    /// Get the value at the current position
    pub fn get_val(&self) -> Option<&Par> {
        use pathmap::zipper::ZipperValues;
        self.zipper.val()
    }

    /// Remove the value at the current position
    pub fn remove_val(&mut self) -> Option<Par> {
        use pathmap::zipper::ZipperWriting;
        self.zipper.remove_val(true)
    }

    /// Remove all branches below the current position
    pub fn remove_branches(&mut self) {
        use pathmap::zipper::ZipperWriting;
        self.zipper.remove_branches(true);
    }

    /// Check if there's a value at current position
    pub fn has_val(&self) -> bool {
        use pathmap::zipper::Zipper;
        self.zipper.is_val()
    }

    /// Check if the current path exists
    pub fn path_exists(&self) -> bool {
        use pathmap::zipper::Zipper;
        self.zipper.path_exists()
    }

    /// Graft a subtrie from a read zipper
    pub fn graft<'b, 'bpath>(&mut self, read_zipper: &RholangReadZipper<'b, 'bpath>) {
        use pathmap::zipper::ZipperWriting;
        self.zipper.graft(&read_zipper.zipper);
    }

    /// Join (union) a subtrie from a read zipper
    pub fn join_into<'b, 'bpath>(&mut self, read_zipper: &RholangReadZipper<'b, 'bpath>) {
        use pathmap::zipper::ZipperWriting;
        self.zipper.join_into(&read_zipper.zipper);
    }

    /// Reset zipper to root
    pub fn reset(&mut self) {
        use pathmap::zipper::ZipperMoving;
        self.zipper.reset();
    }
}

/// Wrapper for PathMap ZipperHead that maintains Rholang context
pub struct RholangZipperHead<'a> {
    #[allow(dead_code)]
    pub(crate) zipper_head: ZipperHead<'a, 'a, Par>,
    #[allow(dead_code)]
    pub(crate) connective_used: bool,
    #[allow(dead_code)]
    pub(crate) locally_free: Vec<u8>,
}

impl<'a> RholangZipperHead<'a> {
    /// Create a new zipper head from a PathMap
    pub fn new(map: &'a mut RholangPathMap, connective_used: bool, locally_free: Vec<u8>) -> Self {
        RholangZipperHead {
            zipper_head: map.zipper_head(),
            connective_used,
            locally_free,
        }
    }
}

/// Helper function to flatten path segments with 0xFF separator
pub(crate) fn flatten_segments(segments: &[Vec<u8>]) -> Vec<u8> {
    segments
        .iter()
        .flat_map(|seg| {
            let mut v = seg.clone();
            v.push(0xFF); // separator
            v
        })
        .collect()
}

/// Helper function to unflatten path segments (split by 0xFF separator)
#[allow(dead_code)]
pub(crate) fn unflatten_segments(flattened: &[u8]) -> Vec<Vec<u8>> {
    flattened
        .split(|&b| b == 0xFF)
        .filter(|seg| !seg.is_empty())
        .map(|seg| seg.to_vec())
        .collect()
}

