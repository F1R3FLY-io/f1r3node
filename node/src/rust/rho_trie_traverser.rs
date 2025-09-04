use casper::rust::{genesis::contracts::standard_deploys, util::rholang::tools::Tools};
use crypto::rust::hash::keccak256::Keccak256;
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::g_unforgeable::UnfInstance;
use models::rhoapi::{EList, ETuple, Expr, GPrivate, GUnforgeable, Par};
use models::rust::par_map::ParMap;
use models::rust::par_map_type_mapper::ParMapTypeMapper;
use prost::Message;
use rholang::rust::interpreter::errors::InterpreterError;
use rholang::rust::interpreter::rho_runtime::RhoRuntime;
use std::collections::HashMap;

pub type ReadParams = (Vec<Vec<i32>>, i32, Par, Vec<ParMap>);

/// Traverse a Rholang Trie.
///
/// This is a 1:1 port of the Scala RhoTrieTraverser from:
/// https://github.com/rchain/rchain/blob/19880674b9c50aa29efe91d77f70b06b861ca7a8/casper/src/main/resources/Registry.rho
///
/// According to the trie implementation in Rholang, the methods below are hacks to traverse the trie
/// structure used in the Registry implementation.
///
/// # Example
///
/// ```rust
/// use node::rust::rho_trie_traverser::RhoTrieTraverser;
///
/// // Create a Par from a string and hash it
/// let key = RhoTrieTraverser::keccak_key("example");
///
/// // Convert byte array to nybble list for trie traversal
/// let byte_array = /* some Par with byte array */;
/// let nybbles = RhoTrieTraverser::byte_array_to_nybble_list(&byte_array, 0, 4, vec![]);
/// ```
pub struct RhoTrieTraverser;

impl RhoTrieTraverser {
    /// Depth indices for trie traversal
    const DEPTH_EACH: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    /// Powers of 2 for bit manipulation during trie traversal
    const POWERS: [i64; 17] = [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    ];

    /// Create a Keccak256 hash and wrap it in a Par with GByteArray
    fn keccak_hash(input: &[u8]) -> Par {
        let hash = Keccak256::hash(input.to_vec());
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(hash)),
            }],
            ..Default::default()
        }
    }

    /// Convert a byte array Par to nybble list for trie traversal.
    ///
    /// This is the Rust version of:
    /// https://github.com/rchain/rchain/blob/19880674b9c50aa29efe91d77f70b06b861ca7a8/casper/src/main/resources/Registry.rho#L72-L78
    ///
    /// # Arguments
    /// * `binary_array` - Par containing a byte array
    /// * `n` - Current position in the byte array
    /// * `length` - Total length to process
    /// * `acc` - Accumulator for the nybble list
    ///
    /// # Returns
    /// Vector of nybbles (4-bit values) extracted from the byte array
    pub fn byte_array_to_nybble_list(
        binary_array: &Par,
        n: usize,
        length: usize,
        mut acc: Vec<i32>,
    ) -> Vec<i32> {
        if n == length {
            acc
        } else {
            let nth = Self::nth_of_par(binary_array, n);
            acc.push(nth % 16); // Lower nybble
            acc.push(nth / 16); // Upper nybble
            Self::byte_array_to_nybble_list(binary_array, n + 1, length, acc)
        }
    }

    /// Extract the nth byte from a Par containing a byte array
    ///
    /// # Arguments
    /// * `p` - Par containing a GByteArray
    /// * `nth` - Index of the byte to extract
    ///
    /// # Returns
    /// The byte value as i32 (converted to unsigned)
    ///
    /// # Panics
    /// Panics if the Par is not a valid byte array or index is out of bounds
    pub fn nth_of_par(p: &Par, nth: usize) -> i32 {
        if let Some(expr) = p.exprs.first() {
            if let Some(ExprInstance::GByteArray(bs)) = &expr.expr_instance {
                if nth < bs.len() {
                    return (bs[nth] & 0xff) as i32; // convert to unsigned
                }
            }
        }
        panic!("Par {:?} is not valid for nth_of_par method", p);
    }

    /// Create a Par containing a string
    fn par_string(s: &str) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.to_string())),
            }],
            ..Default::default()
        }
    }

    /// Serialize a Par to byte array using protobuf encoding
    fn par_to_byte_array(p: &Par) -> Vec<u8> {
        p.encode_to_vec()
    }

    /// Create a Keccak256 hash of a string wrapped as a Par
    ///
    /// # Arguments
    /// * `s` - String to hash
    ///
    /// # Returns
    /// Par containing the Keccak256 hash as a byte array
    pub fn keccak_key(s: &str) -> Par {
        let par_str = Self::par_string(s);
        let byte_array = Self::par_to_byte_array(&par_str);
        Self::keccak_hash(&byte_array)
    }

    /// Create a Keccak256 hash of a string and return as raw bytes
    ///
    /// # Arguments
    /// * `s` - String to hash
    ///
    /// # Returns
    /// Raw byte vector of the Keccak256 hash
    pub fn keccak_par_string(s: &str) -> Vec<u8> {
        let par_str = Self::par_string(s);
        let byte_array = Self::par_to_byte_array(&par_str);
        Keccak256::hash(byte_array)
    }

    /// Create a Par containing a list of integers from nybble list
    ///
    /// # Arguments
    /// * `nyb_list` - Vector of nybble values
    ///
    /// # Returns
    /// Par containing an EList of the nybbles as GInt values
    pub fn node_list(nyb_list: &[i32]) -> Par {
        let ps: Vec<Par> = nyb_list
            .iter()
            .map(|&n| Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::GInt(n as i64)),
                }],
                ..Default::default()
            })
            .collect();

        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EListBody(EList {
                    ps,
                    locally_free: vec![],
                    connective_used: false,
                    remainder: None,
                })),
            }],
            ..Default::default()
        }
    }

    /// Create a node map list for trie query
    fn node_map_list(map: &Par, nyb_list: &[i32]) -> Par {
        let map_with_nyb = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::ETupleBody(ETuple {
                    ps: vec![map.clone(), Self::node_list(nyb_list)],
                    locally_free: vec![],
                    connective_used: false,
                })),
            }],
            ..Default::default()
        };
        Self::node_map_store(&map_with_nyb)
    }

    /// Wrap the map with store token for runtime query
    fn node_map_store(map_with_nyb: &Par) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EListBody(EList {
                    ps: vec![map_with_nyb.clone(), Self::store_token_unforgeable()],
                    locally_free: vec![],
                    connective_used: false,
                    remainder: None,
                })),
            }],
            ..Default::default()
        }
    }

    /// Create the store token unforgeable name
    ///
    fn store_token_unforgeable() -> Par {
        let mut rand = Tools::unforgeable_name_rng(
            &standard_deploys::REGISTRY_PUB_KEY,
            standard_deploys::REGISTRY_TIMESTAMP,
        );

        // Call rand.next() 6 times (0 to 5 inclusive)
        for _ in 0..6 {
            rand.next();
        }

        // Split with index 6
        let mut new_rand = rand.split_short(6);

        // Call newRand.next() 7 times (0 to 6 inclusive)
        for _ in 0..7 {
            new_rand.next();
        }

        // Get the final target
        let target = new_rand.next();

        Par {
            unforgeables: vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                    id: target.iter().map(|&b| b as u8).collect(),
                })),
            }],
            ..Default::default()
        }
    }

    /// Get data from the trie at a specific node
    ///
    /// # Arguments
    /// * `map_par` - The trie map Par
    /// * `nyb_list` - The nybble path to query
    /// * `runtime` - RhoRuntime instance for data access
    ///
    /// # Returns
    /// Result containing either an integer value (left) or a ParMap (right) if found, None otherwise
    fn tree_hash_map_getter<R: RhoRuntime>(
        map_par: &Par,
        nyb_list: &[i32],
        runtime: &R,
    ) -> Result<Option<Result<i64, ParMap>>, InterpreterError> {
        let query = Self::node_map_list(map_par, nyb_list);
        let result = runtime.get_data(&query);

        if let Some(first) = result.first() {
            if let Some(head_par) = first.a.pars.first() {
                if let Some(head_expr) = head_par.exprs.first() {
                    match &head_expr.expr_instance {
                        Some(ExprInstance::GInt(i)) => return Ok(Some(Ok(*i))),
                        Some(ExprInstance::EMapBody(emap)) => {
                            // Convert EMap to ParMap using the existing mapper
                            let par_map = ParMapTypeMapper::emap_to_par_map(emap.clone());
                            return Ok(Some(Err(par_map)));
                        }
                        _ => return Ok(None),
                    }
                }
            }
        }
        Ok(None)
    }

    /// Convert a vector of ParMaps to a HashMap using provided key/value extractors
    ///
    /// # Arguments
    /// * `values` - Vector of ParMaps to convert
    /// * `get_key` - Function to extract key from Par
    /// * `get_value` - Function to extract value from Par
    ///
    /// # Returns
    /// HashMap with extracted keys and values
    pub fn vec_par_map_to_map<K, V, F, G>(
        values: &[ParMap],
        get_key: F,
        get_value: G,
    ) -> HashMap<K, V>
    where
        F: Fn(&Par) -> K,
        G: Fn(&Par) -> V,
        K: std::hash::Hash + Eq,
    {
        let mut result = HashMap::new();
        for par_map in values {
            for (k_par, v_par) in &par_map.ps.sorted_list {
                let key = get_key(k_par);
                let value = get_value(v_par);
                result.insert(key, value);
            }
        }
        result
    }

    /// Traverse a Rholang trie and collect all ParMaps
    ///
    /// # Arguments
    /// * `depth` - Maximum depth to traverse
    /// * `map_par` - The root trie map Par
    /// * `runtime` - RhoRuntime instance for data access
    ///
    /// # Returns
    /// Result containing vector of all ParMaps found during traversal, or an error
    pub fn traverse_trie<R: RhoRuntime>(
        depth: i32,
        map_par: &Par,
        runtime: &R,
    ) -> Result<Vec<ParMap>, InterpreterError> {
        let start_params: ReadParams = (vec![vec![]], depth * 2, map_par.clone(), vec![]);
        Self::traverse_trie_rec(start_params, runtime)
    }

    /// Extend a key based on bit patterns in the value
    fn extend_key(head: &[i32], value: i64) -> Vec<Vec<i32>> {
        Self::DEPTH_EACH
            .iter()
            .filter(|&&i| (value / Self::POWERS[i as usize]) % 2 != 0)
            .map(|&i| {
                let mut new_key = head.to_vec();
                new_key.push(i);
                new_key
            })
            .collect()
    }

    /// Recursive helper for trie traversal
    ///
    /// This uses iteration instead of recursion to avoid stack overflow issues
    /// and implements the same logic as the Scala tailrec version.
    ///
    /// # Returns
    /// Result containing Either continuation parameters (Left) or final result (Right)
    /// In practice, this iterative version always returns Right with the final result
    fn traverse_trie_rec<R: RhoRuntime>(
        read_params: ReadParams,
        runtime: &R,
    ) -> Result<Vec<ParMap>, InterpreterError> {
        let (mut keys, depth, map_par, mut collected_results) = read_params;

        while let Some(key) = keys.pop() {
            let current_node = Self::tree_hash_map_getter(&map_par, &key, runtime)?;

            match current_node {
                Some(Ok(i)) => {
                    if key.is_empty() {
                        let mut extended = Self::extend_key(&key, i);
                        keys.append(&mut extended);
                    } else if depth == key.len() as i32 {
                        // Skip this key - we've reached max depth
                    } else {
                        let mut extended = Self::extend_key(&key, i);
                        keys.append(&mut extended);
                    }
                }
                Some(Err(map)) => {
                    // Found a ParMap - collect it
                    collected_results.push(map);
                }
                None => {
                    // No data found for this key - skip
                }
            }
        }
        Ok(collected_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_array_to_nybble_list() {
        let byte_array = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(vec![0x12, 0x34])),
            }],
            ..Default::default()
        };

        let result = RhoTrieTraverser::byte_array_to_nybble_list(&byte_array, 0, 2, vec![]);
        assert_eq!(result, vec![2, 1, 4, 3]); // 0x12 -> [2, 1], 0x34 -> [4, 3]
    }

    #[test]
    fn test_nth_of_par() {
        let byte_array = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(vec![0x12, 0x34, 0x56])),
            }],
            ..Default::default()
        };

        assert_eq!(RhoTrieTraverser::nth_of_par(&byte_array, 0), 0x12);
        assert_eq!(RhoTrieTraverser::nth_of_par(&byte_array, 1), 0x34);
        assert_eq!(RhoTrieTraverser::nth_of_par(&byte_array, 2), 0x56);
    }

    #[test]
    fn test_keccak_key() {
        let key = RhoTrieTraverser::keccak_key("test");
        // Verify it's a byte array
        if let Some(expr) = key.exprs.first() {
            assert!(matches!(
                expr.expr_instance,
                Some(ExprInstance::GByteArray(_))
            ));
        }
    }

    #[test]
    fn test_node_list() {
        let nyb_list = vec![1, 2, 3];
        let result = RhoTrieTraverser::node_list(&nyb_list);

        if let Some(expr) = result.exprs.first() {
            if let Some(ExprInstance::EListBody(elist)) = &expr.expr_instance {
                assert_eq!(elist.ps.len(), 3);
                for (i, par) in elist.ps.iter().enumerate() {
                    if let Some(par_expr) = par.exprs.first() {
                        if let Some(ExprInstance::GInt(val)) = &par_expr.expr_instance {
                            assert_eq!(*val, nyb_list[i] as i64);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_keccak_par_string() {
        let hash1 = RhoTrieTraverser::keccak_par_string("test");
        let hash2 = RhoTrieTraverser::keccak_par_string("test");
        let hash3 = RhoTrieTraverser::keccak_par_string("different");

        // Same input should produce same hash
        assert_eq!(hash1, hash2);
        // Different input should produce different hash
        assert_ne!(hash1, hash3);
        // Hash should be 32 bytes for Keccak256
        assert_eq!(hash1.len(), 32);
    }

    #[test]
    fn test_store_token_unforgeable() {
        // Test that store_token_unforgeable produces deterministic results
        let token1 = RhoTrieTraverser::store_token_unforgeable();
        let token2 = RhoTrieTraverser::store_token_unforgeable();

        // Should be deterministic - same inputs should produce same outputs
        assert_eq!(token1, token2);

        // Verify the structure
        assert_eq!(token1.unforgeables.len(), 1);
        if let Some(unforgeable) = token1.unforgeables.first() {
            if let Some(UnfInstance::GPrivateBody(private)) = &unforgeable.unf_instance {
                // Verify the ID is the expected length (32 bytes for Blake2b512Random.next())
                assert_eq!(private.id.len(), 32);

                // Verify it's not empty or all zeros
                assert!(!private.id.is_empty());
                assert!(private.id.iter().any(|&b| b != 0));

                println!(
                    "✓ Generated deterministic store token with ID: {:?}",
                    private
                        .id
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<String>()
                );
            } else {
                panic!("Expected GPrivateBody in unforgeable");
            }
        } else {
            panic!("Expected at least one unforgeable");
        }
    }
}
