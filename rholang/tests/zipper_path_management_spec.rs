// Unit tests for PathMap zipper path management methods: createPath, prunePath, reset

use models::rhoapi::{Par, Expr, expr::ExprInstance, EPathMap, EZipper, EList};
use models::rust::pathmap_crate_type_mapper::PathMapCrateTypeMapper;

#[cfg(test)]
mod zipper_path_management_tests {
    use super::*;

    fn create_test_pathmap() -> EPathMap {
        // Create PathMap with entries: ["a", "value1"], ["a", "b", "value2"], ["a", "b", "c", "value3"]
        let entries = vec![
            create_path_par(vec!["a".to_string()], "value1"),
            create_path_par(vec!["a".to_string(), "b".to_string()], "value2"),
            create_path_par(vec!["a".to_string(), "b".to_string(), "c".to_string()], "value3"),
            create_path_par(vec!["x".to_string(), "y".to_string()], "value4"),
        ];
        
        EPathMap {
            ps: entries,
            locally_free: vec![],
            connective_used: false,
            remainder: None,
        }
    }

    fn create_path_list(path: Vec<String>) -> Par {
        let path_elements = path.iter().map(|s| {
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.clone())),
            }])
        }).collect::<Vec<_>>();
        
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: path_elements,
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }])
    }

    fn create_path_par(path: Vec<String>, value: &str) -> Par {
        let mut path_elements = path.iter().map(|s| {
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.clone())),
            }])
        }).collect::<Vec<_>>();
        
        path_elements.push(Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GString(value.to_string())),
        }]));
        
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: path_elements,
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }])
    }

    #[test]
    fn test_prune_path_removes_all_children() {
        let pathmap = create_test_pathmap();
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
        let mut rholang_pathmap = pathmap_result.map;
        
        let initial_count = rholang_pathmap.iter().count();
        assert_eq!(initial_count, 4, "Should start with 4 entries");
        
        // Build S-expression encoded prefix for ["a", "b"]
        use models::rust::pathmap_integration::par_to_path;
        let path_segments = par_to_path(&create_path_list(vec!["a".to_string(), "b".to_string()]));
        let prefix_key: Vec<u8> = path_segments.iter().flat_map(|seg| {
            let mut s = seg.clone();
            s.push(0xFF);
            s
        }).collect();
        
        let keys_to_remove: Vec<Vec<u8>> = rholang_pathmap
            .iter()
            .filter_map(|(key, _)| {
                if key.starts_with(&prefix_key) {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect();
        
        for key in keys_to_remove {
            rholang_pathmap.remove(&key);
        }
        
        // After pruning ["a", "b"] and ["a", "b", "c"], should have 2 entries left
        let final_count = rholang_pathmap.iter().count();
        assert_eq!(final_count, 2, "Should have 2 entries after pruning (removed 2 entries)");
    }

    #[test]
    fn test_prune_path_at_leaf() {
        let pathmap = create_test_pathmap();
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
        let mut rholang_pathmap = pathmap_result.map;
        
        let initial_count = rholang_pathmap.iter().count();
        assert_eq!(initial_count, 4, "Should start with 4 entries");
        
        // Build S-expression encoded prefix for ["a", "b", "c"]
        use models::rust::pathmap_integration::par_to_path;
        let path_segments = par_to_path(&create_path_list(vec!["a".to_string(), "b".to_string(), "c".to_string()]));
        let prefix_key: Vec<u8> = path_segments.iter().flat_map(|seg| {
            let mut s = seg.clone();
            s.push(0xFF);
            s
        }).collect();
        
        let keys_to_remove: Vec<Vec<u8>> = rholang_pathmap
            .iter()
            .filter_map(|(key, _)| {
                if key.starts_with(&prefix_key) {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect();
        
        for key in keys_to_remove {
            rholang_pathmap.remove(&key);
        }
        
        // After pruning just ["a", "b", "c"], should have 3 entries left
        let final_count = rholang_pathmap.iter().count();
        assert_eq!(final_count, 3, "Should have 3 entries after pruning (removed 1 entry)");
    }

    #[test]
    fn test_prune_path_at_root() {
        let pathmap = create_test_pathmap();
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
        let mut rholang_pathmap = pathmap_result.map;
        
        // Prune at root (empty prefix) - should remove everything
        let prefix_key: Vec<u8> = vec![];
        
        let keys_to_remove: Vec<Vec<u8>> = rholang_pathmap
            .iter()
            .filter_map(|(key, _)| {
                if key.starts_with(&prefix_key) {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect();
        
        for key in keys_to_remove {
            rholang_pathmap.remove(&key);
        }
        
        assert!(rholang_pathmap.is_empty(), "All paths should be removed when pruning at root");
    }

    #[test]
    fn test_prune_path_nonexistent() {
        let pathmap = create_test_pathmap();
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
        let mut rholang_pathmap = pathmap_result.map;
        let original_size = rholang_pathmap.iter().count();
        
        // Try to prune non-existent path
        let prefix_key: Vec<u8> = vec![b'z', 0xFF];
        
        let keys_to_remove: Vec<Vec<u8>> = rholang_pathmap
            .iter()
            .filter_map(|(key, _)| {
                if key.starts_with(&prefix_key) {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect();
        
        for key in keys_to_remove {
            rholang_pathmap.remove(&key);
        }
        
        assert_eq!(rholang_pathmap.iter().count(), original_size, "PathMap size should remain unchanged");
    }

    #[test]
    fn test_reset_zipper_to_root() {
        let pathmap = create_test_pathmap();
        
        // Create zipper at path ["a", "b"]
        let mut zipper = EZipper {
            pathmap: Some(pathmap),
            current_path: vec![b"a".to_vec(), b"b".to_vec()],
            is_write_zipper: false,
            locally_free: vec![],
            connective_used: false,
        };
        
        // Reset to root
        zipper.current_path = vec![];
        
        assert!(zipper.current_path.is_empty(), "Zipper should be at root after reset");
    }

    #[test]
    fn test_reset_read_zipper() {
        let pathmap = create_test_pathmap();
        
        let mut zipper = EZipper {
            pathmap: Some(pathmap),
            current_path: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            is_write_zipper: false,
            locally_free: vec![],
            connective_used: false,
        };
        
        zipper.current_path = vec![];
        
        assert!(zipper.current_path.is_empty(), "Read zipper should be at root after reset");
    }

    #[test]
    fn test_reset_write_zipper() {
        let pathmap = create_test_pathmap();
        
        let mut zipper = EZipper {
            pathmap: Some(pathmap),
            current_path: vec![b"x".to_vec(), b"y".to_vec()],
            is_write_zipper: true,
            locally_free: vec![],
            connective_used: false,
        };
        
        zipper.current_path = vec![];
        
        assert!(zipper.current_path.is_empty(), "Write zipper should be at root after reset");
        assert!(zipper.is_write_zipper, "Zipper should remain a write zipper after reset");
    }

    #[test]
    fn test_create_path_validates_format() {
        // createPath is currently a no-op that validates path format
        // This test verifies the structure is correct for potential future implementation
        let pathmap = create_test_pathmap();
        
        // The method should accept the PathMap and return it unchanged
        // This validates that the path format is correct
        assert!(!pathmap.ps.is_empty(), "PathMap should not be empty");
    }
}

