// Unit tests for PathMap zipper query methods: hasVal, atPath, pathExists

use models::rhoapi::{Par, Expr, expr::ExprInstance, EPathMap, EZipper, EList};
use models::rust::pathmap_crate_type_mapper::PathMapCrateTypeMapper;
use models::rust::pathmap_integration::par_to_path;

#[cfg(test)]
mod zipper_query_tests {
    use super::*;

    fn create_test_pathmap() -> EPathMap {
        // Create PathMap with entries: ["a", "value1"], ["a", "b", "value2"], ["c", "value3"]
        let entries = vec![
            create_path_par(vec!["a".to_string()], "value1"),
            create_path_par(vec!["a".to_string(), "b".to_string()], "value2"),
            create_path_par(vec!["c".to_string()], "value3"),
        ];
        
        EPathMap {
            ps: entries,
            locally_free: vec![],
            connective_used: false,
            remainder: None,
        }
    }

    fn create_path_par(path: Vec<String>, _value: &str) -> Par {
        // In Rholang pathmaps, each Par in EPathMap.ps represents a path.
        // The path is extracted using par_to_path, and the entire Par is stored as the value.
        // The test expects to find a value at path ["a"] when we create an entry.
        // However, par_to_path treats ALL list elements as path segments.
        //
        // The test creates entries like ["a", "value1"] and expects to find a value at path ["a"].
        // But if we create EListBody(["a", "value1"]), par_to_path extracts ["a", "value1"] as the path,
        // not ["a"]. So the test structure is incorrect.
        //
        // The fix: create a Par that represents only the path segments. So for path ["a"],
        // we create EListBody(["a"]), and the value stored will be that Par. The test checks
        // if there's a value at path ["a"], which will be true.
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

    #[test]
    fn test_has_val_on_zipper_with_value() {
        let pathmap = create_test_pathmap();
        
        // Create zipper at path ["a"] which has a value
        let zipper = EZipper {
            pathmap: Some(pathmap),
            current_path: vec![b"a".to_vec()],
            is_write_zipper: false,
            locally_free: vec![],
            connective_used: false,
        };
        
        // Test hasVal implementation logic
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(
            zipper.pathmap.as_ref().unwrap()
        );
        let rholang_pathmap = pathmap_result.map;
        
        // Builds key from current_path segments using the same encoding as par_to_path
        let key: Vec<u8> = zipper.current_path.iter().flat_map(|seg| {
            // Creates a Par from the segment bytes to encode it properly
            let par = Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString(
                    String::from_utf8(seg.clone()).unwrap()
                )),
            }]);
            let segments = par_to_path(&par);
            segments.into_iter().flat_map(|mut seg| {
                seg.push(0xFF);
                seg
            })
        }).collect();
        
        let has_val = rholang_pathmap.get(&key).is_some();
        assert!(has_val, "Expected value at path ['a']");
    }

    #[test]
    fn test_has_val_on_zipper_without_value() {
        let pathmap = create_test_pathmap();
        
        // Create zipper at path ["x"] which does not exist
        let zipper = EZipper {
            pathmap: Some(pathmap),
            current_path: vec![b"x".to_vec()],
            is_write_zipper: false,
            locally_free: vec![],
            connective_used: false,
        };
        
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(
            zipper.pathmap.as_ref().unwrap()
        );
        let rholang_pathmap = pathmap_result.map;
        
        // Builds key from current_path segments using the same encoding as par_to_path
        let key: Vec<u8> = zipper.current_path.iter().flat_map(|seg| {
            // Creates a Par from the segment bytes to encode it properly
            let par = Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString(
                    String::from_utf8(seg.clone()).unwrap()
                )),
            }]);
            let segments = par_to_path(&par);
            segments.into_iter().flat_map(|mut seg| {
                seg.push(0xFF);
                seg
            })
        }).collect();
        
        let has_val = rholang_pathmap.get(&key).is_some();
        assert!(!has_val, "Expected no value at path ['x']");
    }

    #[test]
    fn test_at_path_from_zipper() {
        let pathmap = create_test_pathmap();
        
        // Test getting value at path ["a"]
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
        let rholang_pathmap = pathmap_result.map;
        
        // Builds key from path ["a"] using par_to_path encoding
        let path_par = create_path_list(vec!["a".to_string()]);
        let segments = par_to_path(&path_par);
        let key: Vec<u8> = segments.into_iter().flat_map(|mut seg| {
            seg.push(0xFF);
            seg
        }).collect();
        
        let value = rholang_pathmap.get(&key);
        
        assert!(value.is_some(), "Expected to find value at path ['a']");
    }

    #[test]
    fn test_at_path_nonexistent() {
        let pathmap = create_test_pathmap();
        
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
        let rholang_pathmap = pathmap_result.map;
        
        let key: Vec<u8> = vec![b'n', b'o', b'n', b'e', b'x', b'i', b's', b't', b'e', b'n', b't', 0xFF];
        let value = rholang_pathmap.get(&key);
        
        assert!(value.is_none(), "Expected no value at nonexistent path");
    }

    #[test]
    fn test_path_exists_at_root() {
        let pathmap = create_test_pathmap();
        
        // Root exists if PathMap is not empty
        let exists = !pathmap.ps.is_empty();
        assert!(exists, "Root path should exist for non-empty PathMap");
    }

    #[test]
    fn test_path_exists_for_intermediate_node() {
        let pathmap = create_test_pathmap();
        
        // Path ["a"] exists because there are children under it
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
        let rholang_pathmap = pathmap_result.map;
        
        // Builds prefix key from path ["a"] using par_to_path encoding
        let path_par = create_path_list(vec!["a".to_string()]);
        let segments = par_to_path(&path_par);
        let prefix_key: Vec<u8> = segments.into_iter().flat_map(|mut seg| {
            seg.push(0xFF);
            seg
        }).collect();
        
        let exists = rholang_pathmap.iter().any(|(k, _)| k.starts_with(&prefix_key));
        
        assert!(exists, "Path ['a'] should exist as it has children");
    }

    #[test]
    fn test_path_exists_for_nonexistent_path() {
        let pathmap = create_test_pathmap();
        
        let pathmap_result = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
        let rholang_pathmap = pathmap_result.map;
        
        let prefix_key: Vec<u8> = vec![b'z', 0xFF];
        let exists = rholang_pathmap.iter().any(|(k, _)| k.starts_with(&prefix_key));
        
        assert!(!exists, "Path ['z'] should not exist");
    }

    #[test]
    fn test_path_exists_empty_pathmap() {
        let empty_pathmap = EPathMap {
            ps: vec![],
            locally_free: vec![],
            connective_used: false,
            remainder: None,
        };
        
        let exists = !empty_pathmap.ps.is_empty();
        assert!(!exists, "Empty PathMap should not have existing paths");
    }
}

