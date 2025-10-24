// See rspace/src/main/scala/coop/rchain/rspace/history/RadixTree.scala

use crate::rspace::errors::RadixTreeError;
use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::history::Either;
use crate::rspace::history::history_action::HistoryActionTrait;
use dashmap::DashMap;
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::rust::Byte;
use shared::rust::ByteVector;
use shared::rust::store::key_value_store::KeyValueStore;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::history_action::DeleteAction;
use super::history_action::HistoryAction;
use super::history_action::InsertAction;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub enum Item {
    EmptyItem,

    Leaf {
        prefix: ByteVector,
        value: ByteVector,
    },

    NodePtr {
        prefix: ByteVector,
        ptr: ByteVector,
    },
}

pub type Node = Vec<Item>;

const NUM_ITEMS: usize = 256;

pub fn empty_node() -> Node {
    vec![Item::EmptyItem; NUM_ITEMS]
}

/**
* Binary codecs for serializing/deserializing Node in Radix tree
*
* {{{
* Coding structure for items:
*   EmptyItem                   - Empty (not encode)

*   Leaf(prefix,value)    -> [item index] [second byte] [prefix0]..[prefixM] [value0]..[value31]
*                               where is: [second byte] -> bit7 = 0 (Leaf identifier)
*                                                          bit6..bit0 - prefix length = M (from 0 to 127)
*
*   NodePtr(prefix,ptr)   -> [item index] [second byte] [prefix0]..[prefixM] [ptr0]..[ptr31]
*                               where is: [second byte] -> bit7 = 1 (NodePtr identifier)
*                                                          bit6..bit0 - prefix length = M (from 0 to 127)
*
* For example encode this Node which contains 2 non-empty items (index 1 and index 2):
* (0)[Empty] (1)[Leaf(prefix:0xFFFF,value:0x00..0001)] (2)[NodePtr(prefix:empty,value:0xFF..FFFF)] (3)...(255)[Empty].
* Encoded data = 0x0102FFFF0000000000000000000000000000000000000000000000000000000000000001
*                  0280FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
* where: item 1 (index_secondByte_prefix_value) = 01_02_FFFF_00..0001
*        item 2 (index_secondByte_prefix_value) = 02_80_empty_FF..FFFF
* }}}
*/

// Default size for non-empty item data
const DEF_SIZE: usize = 32;
// 2 bytes: first - item index, second - second byte
const HEAD_SIZE: usize = 2;

/** Serialization [[Node]] to [[ByteVector]]
 */
pub fn encode(node: &Node) -> ByteVector {
    // Calculate the size of the serialized data
    let calc_size = node.iter().fold(0, |size, item| match item {
        Item::EmptyItem => size,

        Item::Leaf {
            prefix: leaf_prefix,
            value,
        } => {
            let size_prefix = leaf_prefix.len();
            assert!(
                size_prefix <= 127,
                "Error during serialization: size of prefix more than 127."
            );
            assert!(
                value.len() == DEF_SIZE,
                "Error during serialization: size of leafValue not equal 32."
            );
            size + HEAD_SIZE + size_prefix + DEF_SIZE
        }

        Item::NodePtr {
            prefix: ptr_prefix,
            ptr,
        } => {
            let size_prefix = ptr_prefix.len();
            assert!(
                size_prefix <= 127,
                "Error during serialization: size of prefix more than 127."
            );
            assert!(
                ptr.len() == DEF_SIZE,
                "Error during serialization: size of ptrPrefix not equal 32."
            );
            size + HEAD_SIZE + size_prefix + DEF_SIZE
        }
    });

    let buf = vec![0u8; calc_size]; //Allocation memory

    // Leaf second byte: Leaf identifier (most significant bit = 0) and prefixSize (lower 7 bits = size)
    fn encode_leaf_second_byte(prefix_size: usize) -> u8 {
        (prefix_size & 0x7F) as u8 // (size & 01111111b)
    }

    // NodePtr second byte: NodePtr identifier (most significant bit = 1) and prefixSize (lower 7 bits = size)
    fn encode_node_ptr_second_byte(prefix_size: usize) -> u8 {
        (0x80 | (prefix_size & 0x7F)) as u8 // 10000000b | (size & 01111111b)
    }

    fn put_item_into_array(
        idx_item: usize,
        pos0: usize,
        mut arr: Vec<u8>,
        node: &Node,
        calc_size: usize,
    ) -> Vec<u8> {
        if pos0 == calc_size {
            arr // Happy end (return serializing data).
        } else {
            match &node[idx_item] {
                // If current item is empty - just skip serialization of this item
                Item::EmptyItem => put_item_into_array(idx_item + 1, pos0, arr, node, calc_size), // Loop to the next item.

                Item::Leaf { prefix, value } => {
                    // Fill first byte - item index
                    arr[pos0] = idx_item as u8;
                    // Fill second byte - Leaf identifier
                    let pos_second_byte = pos0 + 1;
                    let prefix_size = prefix.len();
                    arr[pos_second_byte] = encode_leaf_second_byte(prefix_size);
                    // Fill prefix
                    let pos_prefix_start = pos_second_byte + 1;
                    for i in 0..prefix_size {
                        arr[pos_prefix_start + i] = prefix[i];
                    }
                    // Fill leafValue
                    let pos_value_start = pos_prefix_start + prefix_size;
                    for i in 0..DEF_SIZE {
                        arr[pos_value_start + i] = value[i];
                    }

                    // Loop to the next item.
                    put_item_into_array(
                        idx_item + 1,
                        pos_value_start + DEF_SIZE,
                        arr,
                        node,
                        calc_size,
                    )
                }

                Item::NodePtr { prefix, ptr } => {
                    // Fill first byte - item index
                    arr[pos0] = idx_item as u8;
                    // Fill second byte - NodePtr identifier (most significant bit = 1) and prefixSize (lower 7 bits = size)
                    let pos_second_byte = pos0 + 1;
                    let prefix_size = prefix.len();
                    arr[pos_second_byte] = encode_node_ptr_second_byte(prefix_size);
                    // Fill prefix
                    let pos_prefix_start = pos_second_byte + 1;
                    for i in 0..prefix_size {
                        arr[pos_prefix_start + i] = prefix[i];
                    }
                    // Fill ptr
                    let pos_ptr_start = pos_prefix_start + prefix_size;
                    for i in 0..DEF_SIZE {
                        arr[pos_ptr_start + i] = ptr[i];
                    }

                    // Loop to the next item.
                    put_item_into_array(
                        idx_item + 1,
                        pos_ptr_start + DEF_SIZE,
                        arr,
                        node,
                        calc_size,
                    )
                }
            }
        }
    }

    put_item_into_array(0, 0, buf, node, calc_size)
}

/** Deserialization [[ByteVector]] to [[Node]]
 */
pub fn decode(bv: ByteVector) -> Node {
    let max_size = bv.len();

    // If first bit 0 - return true, otherwise false.
    fn is_leaf(second_byte: u8) -> bool {
        (second_byte & 0x80) == 0x00
    }

    // Each loop decodes one non-empty item.
    fn decode_item(pos0: usize, node: Node, max_size: usize, arr: Vec<u8>) -> Node {
        if pos0 == max_size {
            node // End of deserialization
        } else {
            let idx_item = byte_to_int(arr[pos0]); // Take first byte - it's item's index
            assert!(
                node[idx_item] == Item::EmptyItem,
                "Error during deserialization: wrong index of item."
            );
            let pos1 = pos0 + 1;
            let second_byte = arr[pos1]; // Take second byte

            // Decoding prefix
            let prefix_size = (second_byte & 0x7F) as usize; // Lower 7 bits - it's size of prefix (0..127).
            let mut prefix = vec![0u8; prefix_size];
            let pos_prefix_start = pos1 + 1;
            for i in 0..prefix_size {
                prefix[i] = arr[pos_prefix_start + i]; // Take prefix
            }

            // Decoding leaf or nodePtr data
            let mut val_or_ptr = vec![0u8; DEF_SIZE];
            let pos_val_or_ptr_start = pos_prefix_start + prefix_size;
            for i in 0..DEF_SIZE {
                val_or_ptr[i] = arr[pos_val_or_ptr_start + i]; // Take next 32 bytes - it's data
            }

            let pos0_next = pos_val_or_ptr_start + DEF_SIZE; // Calculating start position for next loop

            // Decoding type of non-empty item
            let item = if is_leaf(second_byte) {
                Item::Leaf {
                    prefix,
                    value: val_or_ptr,
                }
            } else {
                Item::NodePtr {
                    prefix,
                    ptr: val_or_ptr,
                }
            };

            let mut node_next = node.clone();
            node_next[idx_item] = item;

            decode_item(pos0_next, node_next, max_size, arr) // Try to decode next item.
        }
    }

    decode_item(0, empty_node(), max_size, bv)
}

fn common_prefix(b1: ByteVector, b2: ByteVector) -> (ByteVector, ByteVector, ByteVector) {
    fn go(
        common: ByteVector,
        l: ByteVector,
        r: ByteVector,
    ) -> (ByteVector, ByteVector, ByteVector) {
        if r.is_empty() || l.is_empty() {
            (common, l, r)
        } else {
            let (l_head, l_tail) = l.split_first().unwrap();
            let (r_head, r_tail) = r.split_first().unwrap();
            if l_head == r_head {
                let mut new_common = common.clone();
                new_common.push(*l_head);
                go(new_common, l_tail.to_vec(), r_tail.to_vec())
            } else {
                (common, l, r)
            }
        }
    }
    go(Vec::new(), b1, b2)
}

pub fn hash_node(node: &Node) -> (ByteVector, ByteVector) {
    let node_bytes = encode(node);
    let hash = Blake2b256Hash::new(&node_bytes);
    (hash.bytes(), node_bytes)
}

fn byte_to_int(b: u8) -> usize {
    b as usize & 0xff
}

/**
 * Data returned after export
 *
 * @param nodePrefixes Node prefixes
 * @param nodeKeys Node KVDB keys
 * @param nodeValues Node KVDB values
 * @param leafPrefixes Leaf prefixes
 * @param leafValues Leaf values (it's pointer for data in datastore)
 */
#[derive(Clone, Debug)]
pub struct ExportData {
    pub node_prefixes: Vec<ByteVector>,
    pub node_keys: Vec<ByteVector>,
    pub node_values: Vec<ByteVector>,
    pub leaf_prefixes: Vec<ByteVector>,
    pub leaf_values: Vec<ByteVector>,
}

impl ExportData {
    fn new(
        node_prefixes: Vec<ByteVector>,
        node_keys: Vec<ByteVector>,
        node_values: Vec<ByteVector>,
        leaf_prefixes: Vec<ByteVector>,
        leaf_values: Vec<ByteVector>,
    ) -> Self {
        ExportData {
            node_prefixes,
            node_keys,
            node_values,
            leaf_prefixes,
            leaf_values,
        }
    }
}

/**
 * Settings for [[ExportData]]
 *
 * If false - data will not be exported.
 */
#[derive(Clone)]
pub struct ExportDataSettings {
    pub flag_node_prefixes: bool,
    pub flag_node_keys: bool,
    pub flag_node_values: bool,
    pub flag_leaf_prefixes: bool,
    pub flag_leaf_values: bool,
}

/**
 * Sequential export algorithm
 *
 * @param rootHash Root node hash, starting point
 * @param lastPrefix Describes the path of root to last processed element (if None - start from root)
 * @param skipSize Describes how many elements to skip
 * @param takeSize Describes how many elements to take
 * @param getNodeDataFromStore Function to get data from storage
 * @param settings [[ExportDataSettings]]
 *
 * @return
 * Return the data and prefix of the last processed item.
 * If all bonds in the tree are processed, returns None as prefix.
 * {{{
 * prefix - Prefix that describes the path of root to node
 * decoded - Deserialized data (from parsing)
 * lastItemIndex - Last processed item index
 * }}}
 */
pub fn sequential_export(
    root_hash: ByteVector,
    last_prefix: Option<ByteVector>,
    skip_size: i32,
    take_size: i32,
    get_node_data_from_store: Arc<dyn Fn(&ByteVector) -> Option<ByteVector>>,
    settings: ExportDataSettings,
) -> Result<(ExportData, Option<ByteVector>), RadixTreeError> {
    #[derive(Clone)]
    struct NodeData {
        prefix: ByteVector,
        decoded: Node,
        last_item_index: Option<Byte>,
    }

    impl NodeData {
        fn new(prefix: ByteVector, decoded: Node, last_item_index: Option<Byte>) -> Self {
            NodeData {
                prefix,
                decoded,
                last_item_index,
            }
        }
    }

    type Path = Vec<NodeData>; // Sequence used in recursions

    struct NodePathData {
        hash: ByteVector,        // Hash of node for load
        node_prefix: ByteVector, // Prefix of this node
        rest_prefix: ByteVector, // Prefix that describes the rest of the Path
        path: Path,              // Return path
    }

    /*
     * Create path from root to lastPrefix node
     */
    let init_node_path: Box<
        dyn Fn(NodePathData) -> Result<Either<NodePathData, Path>, RadixTreeError>,
    > = Box::new(|p: NodePathData| {
        let process_child_item = |node: Node| {
            let item_idx = byte_to_int(*p.rest_prefix.first().unwrap());
            match node.get(item_idx) {
                Some(Item::NodePtr {
                    prefix: ptr_prefix,
                    ptr,
                }) => {
                    let (_, rest_prefix_tail) = p.rest_prefix.split_first().unwrap();
                    let (mut prefix_common, prefix_rest, ptr_prefix_rest) =
                        common_prefix(rest_prefix_tail.to_vec(), ptr_prefix.to_vec());

                    assert!(ptr_prefix_rest.is_empty(), "{}", {
                        let mut node_prefix_cloned = p.node_prefix.clone();
                        let mut rest_prefix_cloned = p.rest_prefix.clone();
                        node_prefix_cloned.append(&mut rest_prefix_cloned);

                        format!(
                            "Radix Tree - Export error: node with prefix {} not found.",
                            hex::encode(node_prefix_cloned)
                        )
                    });

                    let mut node_prefix_appended = p.node_prefix.clone();
                    node_prefix_appended.push(*p.rest_prefix.first().unwrap());
                    node_prefix_appended.append(&mut prefix_common);

                    Either::Left(NodePathData {
                        hash: ptr.to_vec(),
                        node_prefix: node_prefix_appended,
                        rest_prefix: prefix_rest,
                        path: {
                            let mut new_path = Vec::new();
                            new_path.push(NodeData::new(
                                p.node_prefix.clone(),
                                node,
                                p.rest_prefix.first().copied(),
                            ));

                            new_path.extend(p.path.clone());
                            new_path
                        },
                    })
                }
                _ => {
                    assert!(false, "{}", {
                        let mut node_prefix_cloned = p.node_prefix.clone();
                        let mut rest_prefix_cloned = p.rest_prefix.clone();
                        node_prefix_cloned.append(&mut rest_prefix_cloned);

                        format!(
                            "Radix Tree - Export error: node with prefix {} not found.",
                            hex::encode(node_prefix_cloned)
                        )
                    });

                    Either::Right(Vec::<NodeData>::new()) // Not found
                }
            }
        };

        let node_opt = get_node_data_from_store(&p.hash);
        if node_opt.is_none() {
            Err(RadixTreeError::KeyNotFound(format!(
                "Radix Tree - Export error: node with key {} not found.",
                { hex::encode(p.hash) }
            )))
        } else {
            let decoded_node = decode(node_opt.unwrap());
            if p.rest_prefix.is_empty() {
                let mut new_path = Vec::new();
                new_path.push(NodeData::new(p.node_prefix, decoded_node, None));

                new_path.extend(p.path);
                Ok(Either::Right(new_path)) // Happy end
            } else {
                Ok(process_child_item(decoded_node)) // Go dipper
            }
        }
    });

    /*
     * Find next non-empty item.
     *
     * @param node Node to look for
     * @param lastIdxOpt Last found index (if this node was not searched - [[None]])
     * @param settings ExportDataSettings from outer scope
     * @return [[Some]](idxItem, [[Item]]) if item found, [[None]] if non-empty item not found
     */
    fn find_next_non_empty_item(
        node: Node,
        last_idx_opt: Option<Byte>,
        settings: ExportDataSettings,
    ) -> Option<(Byte, Item)> {
        if last_idx_opt == Some(0xFF) {
            None
        } else {
            let cur_idx_int = last_idx_opt.map(|b| byte_to_int(b) + 1).unwrap_or(0);
            let cur_item = node.get(cur_idx_int).unwrap();
            let cur_idx = cur_idx_int as u8;
            match cur_item {
                Item::EmptyItem => find_next_non_empty_item(node, Some(cur_idx), settings),
                Item::Leaf {
                    prefix: _,
                    value: _,
                } => {
                    if settings.flag_leaf_prefixes || settings.flag_leaf_values {
                        Some((cur_idx, cur_item.clone()))
                    } else {
                        find_next_non_empty_item(node, Some(cur_idx), settings)
                    }
                }
                Item::NodePtr { prefix: _, ptr: _ } => Some((cur_idx, cur_item.clone())),
            }
        }
    }

    #[derive(Clone)]
    struct StepData {
        path: Path,           // Path of node from current to root
        skip: i32,            // Skip counter
        take: i32,            // Take counter
        exp_data: ExportData, // Result of export
    }

    impl StepData {
        fn new(path: Path, skip: i32, take: i32, exp_data: ExportData) -> Self {
            StepData {
                path,
                skip,
                take,
                exp_data,
            }
        }
    }

    let add_leaf: Box<
        dyn Fn(StepData, ByteVector, ByteVector, Byte, ByteVector, Vec<NodeData>) -> StepData,
    > = Box::new(
        |p: StepData,
         leaf_prefix: ByteVector,
         leaf_value: ByteVector,
         item_index: Byte,
         curr_node_prefix: ByteVector,
         new_path: Vec<NodeData>| {
            if p.skip > 0 {
                StepData::new(new_path, p.skip, p.take, p.exp_data)
            } else {
                let new_lp = if settings.flag_leaf_prefixes {
                    let mut new_single_lp = curr_node_prefix;
                    let mut leaf_prefix_copy = leaf_prefix;
                    new_single_lp.push(item_index);
                    new_single_lp.append(&mut leaf_prefix_copy);

                    let mut leaf_prefixes_copy = p.exp_data.leaf_prefixes;
                    leaf_prefixes_copy.push(new_single_lp);
                    leaf_prefixes_copy
                } else {
                    Vec::new()
                };

                let new_lv = if settings.flag_leaf_values {
                    let mut leaf_values_copied = p.exp_data.leaf_values;
                    leaf_values_copied.push(leaf_value);
                    leaf_values_copied
                } else {
                    Vec::new()
                };

                let new_export_data = ExportData::new(
                    p.exp_data.node_prefixes,
                    p.exp_data.node_keys,
                    p.exp_data.node_values,
                    new_lp,
                    new_lv,
                );

                StepData::new(new_path, p.skip, p.take, new_export_data)
            }
        },
    );

    let add_node_ptr: Box<
        dyn Fn(
            StepData,
            ByteVector,
            ByteVector,
            Byte,
            ByteVector,
            Vec<NodeData>,
        ) -> Result<StepData, RadixTreeError>,
    > = Box::new(
        |p: StepData,
         ptr_prefix: ByteVector,
         ptr: ByteVector,
         item_index: Byte,
         curr_node_prefix: ByteVector,
         new_path: Vec<NodeData>| {
            let construct_node_ptr_data =
                |child_path: Vec<NodeData>, child_np: ByteVector, child_nv: ByteVector| {
                    let new_np = if settings.flag_node_prefixes {
                        let mut new = p.exp_data.node_prefixes.clone();
                        new.push(child_np);
                        new
                    } else {
                        Vec::new()
                    };

                    let new_nk = if settings.flag_node_keys {
                        let mut new = p.exp_data.node_keys.clone();
                        new.push(ptr.clone());
                        new
                    } else {
                        Vec::new()
                    };

                    let new_nv = if settings.flag_node_values {
                        let mut new = p.exp_data.node_values.clone();
                        new.push(child_nv);
                        new
                    } else {
                        Vec::new()
                    };

                    let new_data = ExportData::new(
                        new_np,
                        new_nk,
                        new_nv,
                        p.exp_data.leaf_prefixes.clone(),
                        p.exp_data.leaf_values.clone(),
                    );

                    StepData::new(child_path, p.skip, p.take - 1, new_data)
                };

            let child_node_opt = get_node_data_from_store(&ptr);
            if child_node_opt.is_none() {
                Err(RadixTreeError::KeyNotFound(format!(
                    "Radix Tree - Export error: node with key {} not found. here",
                    { hex::encode(ptr) }
                )))
            } else {
                let child_nv = child_node_opt.unwrap();
                let child_decoded = decode(child_nv.clone());
                let mut child_np = curr_node_prefix;
                child_np.push(item_index);
                child_np.append(&mut ptr_prefix.clone());

                let child_node_data = NodeData::new(child_np.clone(), child_decoded, None);
                let mut child_path = vec![child_node_data];
                child_path.append(&mut new_path.clone());

                if p.skip > 0 {
                    Ok(StepData::new(child_path, p.skip - 1, p.take, p.exp_data))
                } else {
                    Ok(construct_node_ptr_data(child_path, child_np, child_nv))
                }
            }
        },
    );

    let add_element: Box<
        dyn Fn(StepData, u8, Item, Vec<Item>, Vec<u8>) -> Result<StepData, RadixTreeError>,
    > = Box::new(
        |p: StepData,
         item_index: Byte,
         item: Item,
         curr_node: Node,
         curr_node_prefix: ByteVector| {
            let new_curr_node_data =
                NodeData::new(curr_node_prefix.clone(), curr_node, Some(item_index));
            let (_, path_tail) = p.path.split_first().unwrap();
            let mut new_path = vec![new_curr_node_data];
            new_path.append(&mut path_tail.to_vec());

            match item {
                Item::EmptyItem => Ok(StepData::new(new_path, p.skip, p.take, p.exp_data)),
                Item::Leaf { prefix, value } => {
                    Ok(add_leaf(p, prefix, value, item_index, curr_node_prefix, new_path))
                }
                Item::NodePtr { prefix, ptr } => {
                    Ok(add_node_ptr(p, prefix, ptr, item_index, curr_node_prefix, new_path)?)
                }
            }
        },
    );

    /*
     * Export one element (Node or Leaf) and recursively move to the next step.
     */
    let export_step: Box<
        dyn Fn(
            StepData,
        )
            -> Result<Either<StepData, (ExportData, Option<ByteVector>)>, RadixTreeError>,
    > = Box::new(|p: StepData| {
        if p.path.is_empty() {
            // End of Tree
            Ok(Either::Right((p.exp_data, None)))
        } else {
            let curr_node_data = p.path.first().unwrap();
            let curr_node_prefix = curr_node_data.prefix.clone();
            let curr_node = curr_node_data.decoded.clone();

            if (p.skip, p.take) == (0, 0) {
                // End of skip&take counter
                Ok(Either::Right((p.exp_data, Some(curr_node_prefix))))
            } else {
                let next_not_empty_item_opt = find_next_non_empty_item(
                    curr_node.clone(),
                    curr_node_data.last_item_index,
                    settings.clone(),
                );

                let new_step_data = match next_not_empty_item_opt {
                    Some((item_index, item)) => {
                        add_element(p, item_index, item, curr_node, curr_node_prefix)
                    }
                    None => Ok({
                        let (_, path_tail) = p.path.split_first().unwrap();
                        StepData::new(path_tail.to_vec(), p.skip, p.take, p.exp_data)
                    }),
                };

                Ok(Either::Left(new_step_data?))
            }
        }
    });

    fn init_conditions_exception() -> () {
        panic!("Export error: invalid initial conditions (skipSize, takeSize)==(0,0).")
    }

    fn empty_export_data() -> ExportData {
        ExportData::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }

    fn empty_result() -> (ExportData, Option<ByteVector>) {
        (empty_export_data(), None)
    }

    let do_export: Box<
        dyn Fn(ByteVector) -> Result<(ExportData, Option<ByteVector>), RadixTreeError>,
    > = Box::new(|root_node_ser: ByteVector| {
        let root_params = NodePathData {
            hash: root_hash.clone(),
            node_prefix: Vec::new(),
            rest_prefix: last_prefix.clone().unwrap_or(Vec::new()),
            path: Vec::new(),
        };

        let no_root_start = (empty_export_data(), skip_size, take_size); // Start from next node after lastPrefix
        let skipped_start = (empty_export_data(), skip_size - 1, take_size); // Skipped node start
        let root_export_data = {
            let new_np = if settings.flag_node_prefixes {
                vec![Vec::<u8>::new()]
            } else {
                Vec::new()
            };

            let new_nk = if settings.flag_node_keys {
                vec![root_hash.clone()]
            } else {
                Vec::new()
            };

            let new_nv = if settings.flag_leaf_values {
                vec![root_node_ser]
            } else {
                Vec::new()
            };

            ExportData::new(new_np, new_nk, new_nv, Vec::new(), Vec::new())
        };

        let root_start = (root_export_data, skip_size, take_size - 1); // Take root

        // Defining init data
        let (init_export_data, init_skip_size, init_take_size) = match last_prefix {
            Some(_) => no_root_start,
            None => {
                if skip_size > 0 {
                    skipped_start
                } else {
                    root_start
                }
            }
        };

        let mut state = root_params;
        let path = loop {
            match init_node_path(state)? {
                Either::Left(new_state) => state = new_state,
                Either::Right(final_state) => break final_state,
            }
        };

        let start_params: StepData =
            StepData::new(path, init_skip_size, init_take_size, init_export_data);

        let mut state = start_params;
        let r = loop {
            match export_step(state)? {
                Either::Left(new_state) => state = new_state,
                Either::Right(final_state) => break final_state,
            }
        };

        Ok(r)
    });

    if (skip_size, take_size) == (0, 0) {
        init_conditions_exception()
    }

    let root_node_ser_opt = get_node_data_from_store(&root_hash);
    match root_node_ser_opt {
        Some(bytes) => do_export(bytes),
        None => Ok(empty_result()),
    }
}

/**
 * Radix Tree implementation
 */
#[derive(Clone)]
pub struct RadixTreeImpl {
    pub store: Arc<dyn KeyValueStore>,
    /**
     * Cache for storing read and decoded nodes.
     *
     * Cache stores kv-pairs (hash, node).
     * Where hash - Blake2b256Hash of serializing nodes data,
     *       node - deserialized data of this node.
     */
    pub cache_r: DashMap<ByteVector, Node>,
    /**
     * Cache for storing serializing nodes. For subsequent unloading in KVDB
     *
     * Cache stores kv-pairs (hash, bytes).
     * Where hash -  Blake2b256Hash of bytes,
     *       bytes - serializing data of nodes.
     */
    pub cache_w: DashMap<ByteVector, ByteVector>,
}

impl RadixTreeImpl {
    pub fn new(store: Arc<dyn KeyValueStore>) -> Self {
        RadixTreeImpl {
            store,
            cache_r: DashMap::new(),
            cache_w: DashMap::new(),
        }
    }

    /**
     * Load and decode serializing data from KVDB.
     */
    fn load_node_from_store(&self, node_ptr: &ByteVector) -> Result<Option<Node>, RadixTreeError> {
        let get_result = self.store.get_one(&node_ptr)?;

        match get_result {
            Some(bytes) => {
                let deserialized_node = decode(bytes);
                Ok(Some(deserialized_node))
            }
            None => Ok(None),
        }
    }

    /**
     * Load one node from [[cacheR]].
     *
     * If there is no such record in cache - load and decode from KVDB, then save to cacheR.
     * If there is no such record in KVDB - execute assert (if set noAssert flag - return emptyNode).
     */
    pub fn load_node(
        &self,
        node_ptr: ByteVector,
        no_assert: Option<bool>,
    ) -> Result<Node, RadixTreeError> {
        let no_assert = no_assert.unwrap_or(false);

        let error_msg = |node_ptr: &[u8]| {
            assert!(
                no_assert,
                "Missing node in database. ptr={:?}",
                node_ptr
                    .iter()
                    .map(|byte| format!("{:02x}", byte))
                    .collect::<Vec<_>>()
            );
        };

        let cache_miss: Box<dyn Fn(ByteVector) -> Result<Node, RadixTreeError>> =
            Box::new(|node_ptr: ByteVector| {
                let store_node_opt = self.load_node_from_store(&node_ptr)?;

                match store_node_opt {
                    Some(ref node) => {
                        self.cache_r.insert(node_ptr.clone(), node.to_vec());
                    }
                    None => error_msg(&node_ptr),
                };

                match store_node_opt {
                    Some(node) => Ok(node),
                    None => Ok(empty_node()),
                }
            });

        let cache_node_opt = self.cache_r.get(&node_ptr);
        match cache_node_opt {
            Some(node) => Ok(node.to_vec()),
            None => cache_miss(node_ptr),
        }
    }

    /**
     * Clear [[cacheR]] (cache for storing read nodes).
     */
    pub fn clear_read_cache(&self) -> () {
        self.cache_r.clear()
    }

    /**
     * Serializing and hashing one [[Node]].
     *
     * Serializing data load in [[cacheW]].
     * If detected collision with older cache data - executing assert
     */
    pub fn save_node(&self, node: Node) -> ByteVector {
        let (node_bytes_hash, node_bytes) = hash_node(&node);
        let check_collision = |v: Node| {
            assert!(
                v == node,
                "Radix Tree - Collision in cache: record with key = ${} has already existed.",
                hex::encode(node_bytes_hash.clone())
            )
        };

        match self.cache_r.get(&node_bytes_hash) {
            Some(node) => check_collision(node.value().to_vec()),
            None => {
                self.cache_r.insert(node_bytes_hash.clone(), node);
            }
        };

        self.cache_w.insert(node_bytes_hash.clone(), node_bytes);
        node_bytes_hash
    }

    /**
     * Save all [[cacheW]] to [[store]]
     *
     * If detected collision with older KVDB data - execute Exception
     */
    pub fn commit(&self) -> Result<(), RadixTreeError> {
        fn collision_panic(collisions: Vec<(ByteVector, ByteVector)>) -> RadixTreeError {
            RadixTreeError::CollisionError(format!(
                "{} collisions in KVDB (first collision with key = {}.",
                collisions.len(),
                hex::encode(collisions.first().unwrap().0.clone()),
            ))
        }

        let kv_pairs: Vec<(ByteVector, ByteVector)> = self
            .cache_w
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let if_absent: Vec<bool> = self
            .store
            .contains(&kv_pairs.clone().into_iter().map(|(k, _)| k).collect_vec())?;
        let kv_if_absent: Vec<((ByteVector, ByteVector), bool)> =
            kv_pairs.into_iter().zip(if_absent.into_iter()).collect();

        let kv_exist: Vec<(ByteVector, ByteVector)> = kv_if_absent
            .iter()
            .filter(|&(_, absent)| *absent)
            .map(|(kv, _)| kv.clone())
            .collect();

        let value_exist_in_store: Vec<Option<ByteVector>> = self
            .store
            .get(&kv_exist.clone().into_iter().map(|(k, _)| k).collect_vec())?;

        let kvv_exist: Vec<((ByteVector, ByteVector), ByteVector)> = kv_exist
            .into_iter()
            .zip(
                value_exist_in_store
                    .into_iter()
                    .map(|v| v.unwrap_or_else(|| Vec::new())),
            )
            .collect();

        let kv_collision: Vec<(ByteVector, ByteVector)> = kvv_exist
            .into_iter()
            .filter(|kvv| !(kvv.0.1 == kvv.1))
            .map(|(kv, _)| kv)
            .collect();

        if !kv_collision.is_empty() {
            return Err(collision_panic(kv_collision));
        }

        let kv_absent: Vec<(ByteVector, ByteVector)> = kv_if_absent
            .into_iter()
            .filter(|&(_, absent)| !absent)
            .map(|(kv, _)| kv)
            .collect();

        let serialized_kv_absent = kv_absent
            .into_iter()
            .map(|(key, value)| (key, value))
            .collect();

        self.store.put(serialized_kv_absent)?;
        Ok(())
    }

    /**
     * Clear [[cacheW]] (cache for storing data to write in KVDB).
     */
    pub fn clear_write_cache(&self) -> () {
        self.cache_w.clear()
    }

    /**
     * Read leaf data with prefix. If data not found, returned [[None]]
     */
    pub fn read(
        &self,
        start_node: Node,
        start_prefix: ByteVector,
    ) -> Result<Option<ByteVector>, RadixTreeError> {
        type Params = (Node, ByteVector);

        let loops: Box<
            dyn Fn(Params) -> Result<Either<Params, Option<ByteVector>>, RadixTreeError>,
        > = Box::new(|params: Params| {
            match params {
                (_, ref prefix) if prefix.is_empty() => Ok(Either::Right(None)), // Not found
                (cur_node, prefix) => match &cur_node[byte_to_int(*prefix.first().unwrap())] {
                    Item::EmptyItem => Ok(Either::Right(None)), // Not found,
                    Item::Leaf {
                        prefix: leaf_prefix,
                        value,
                    } => {
                        let (_, prefix_tail) = prefix.split_first().unwrap();
                        if leaf_prefix == prefix_tail {
                            Ok(Either::Right(Some(value.clone()))) // Happy end
                        } else {
                            Ok(Either::Right(None)) // Not found
                        }
                    }
                    Item::NodePtr {
                        prefix: ptr_prefix,
                        ptr,
                    } => {
                        let (_, prefix_tail) = prefix.split_first().unwrap();
                        let (_, prefix_rest, ptr_prefix_rest) =
                            common_prefix(prefix_tail.to_vec(), ptr_prefix.to_vec());

                        if ptr_prefix_rest.is_empty() {
                            // Deeper
                            self.load_node(ptr.to_vec(), None)
                                .map(|n| Ok(Either::Left((n, prefix_rest))))?
                        } else {
                            Ok(Either::Right(None)) // Not found
                        }
                    }
                },
            }
        });

        tail_rec_m((start_node, start_prefix), loops)
    }

    fn create_node_from_item(&self, item: Item) -> Node {
        match item {
            Item::EmptyItem => empty_node(),
            Item::Leaf {
                prefix: leaf_prefix,
                value: leaf_value,
            } => {
                assert!(
                    !leaf_prefix.is_empty(),
                    "Impossible to create a node. LeafPrefix should be non empty."
                );

                let index = byte_to_int(*leaf_prefix.first().unwrap());
                let (_, leaf_prefix_tail) = leaf_prefix.split_first().unwrap();
                let mut empty_node = empty_node();
                empty_node[index] = Item::Leaf {
                    prefix: leaf_prefix_tail.to_vec(),
                    value: leaf_value,
                };

                empty_node
            }
            Item::NodePtr {
                prefix: node_ptr_prefix,
                ptr,
            } => {
                assert!(
                    !node_ptr_prefix.is_empty(),
                    "Impossible to create a node. NodePtrPrefix should be non empty."
                );

                let index = byte_to_int(*node_ptr_prefix.first().unwrap());
                let (_, node_ptr_prefix_tail) = node_ptr_prefix.split_first().unwrap();
                let mut empty_node = empty_node();
                empty_node[index] = Item::NodePtr {
                    prefix: node_ptr_prefix_tail.to_vec(),
                    ptr,
                };

                empty_node
            }
        }
    }

    /**
     * Create node from [[Item]].
     *
     * If item is NodePtr and prefix is empty - load child node
     */
    fn construct_node_from_item(&self, item: Item) -> Result<Node, RadixTreeError> {
        match item {
            Item::NodePtr { prefix, ptr } if prefix.is_empty() => self.load_node(ptr, None),
            _ => Ok(self.create_node_from_item(item)),
        }
    }

    /**
     * Optimize and save Node, create item from this Node
     */
    fn save_node_and_create_item(&self, node: Node, prefix: ByteVector, compaction: bool) -> Item {
        if compaction {
            let non_empty_items: Vec<Item> = node
                .clone()
                .into_iter()
                .filter(|item| *item != Item::EmptyItem)
                .take(2)
                .collect();

            match non_empty_items.len() {
                0 => Item::EmptyItem, // All items are empty.
                1 => {
                    // Only one item is not empty - merge child and parent nodes.
                    let idx_item = node
                        .iter()
                        .position(|item| *item == non_empty_items[0])
                        .unwrap();

                    match &non_empty_items[0] {
                        Item::EmptyItem => Item::EmptyItem,
                        Item::Leaf {
                            prefix: leaf_prefix,
                            value,
                        } => {
                            let mut new_prefix = prefix.clone();
                            new_prefix.push(idx_item as u8);
                            new_prefix.append(&mut leaf_prefix.clone());
                            Item::Leaf {
                                prefix: new_prefix,
                                value: value.clone(),
                            }
                        }
                        Item::NodePtr {
                            prefix: node_ptr_prefix,
                            ptr,
                        } => {
                            let mut new_prefix = prefix.clone();
                            new_prefix.push(idx_item as u8);
                            new_prefix.append(&mut node_ptr_prefix.clone());
                            Item::NodePtr {
                                prefix: new_prefix,
                                ptr: ptr.to_vec(),
                            }
                        }
                    }
                }
                2 => {
                    Item::NodePtr {
                        // 2 or more items are not empty.
                        prefix,
                        ptr: self.save_node(node),
                    }
                }
                _ => unreachable!(),
            }
        } else {
            Item::NodePtr {
                prefix,
                ptr: self.save_node(node),
            }
        }
    }

    /**
     * Save new leaf value to this part of tree (start from curItems).
     * Rehash and save all depend node to [[cacheW]].
     *
     * If exist leaf with same prefix but different value - update leaf value.
     * If exist leaf with same prefix and same value - return [[None]].
     * @return Updated current item.
     */
    fn update(
        &self,
        curr_item: Item,
        ins_prefix: ByteVector,
        ins_value: ByteVector,
    ) -> Result<Option<Item>, RadixTreeError> {
        let insert_new_node_to_child: Box<
            dyn Fn(ByteVector, ByteVector, ByteVector) -> Result<Option<Item>, RadixTreeError>,
        > = Box::new(|child_ptr: ByteVector, child_prefix: ByteVector, ins_prefix: ByteVector| {
            let child_node = self.load_node(child_ptr, None)?;
            let (ins_prefix_head, ins_prefix_tail) = ins_prefix.split_first().unwrap();
            let (child_item_idx, child_ins_prefix) =
                (byte_to_int(*ins_prefix_head), ins_prefix_tail);

            let child_item_opt = self.update(
                child_node.get(child_item_idx).unwrap().clone(),
                child_ins_prefix.to_vec(),
                ins_value.clone(),
            )?; // Deeper

            match child_item_opt {
                Some(child_item) => {
                    let mut updated_child_node = child_node.clone();
                    updated_child_node[child_item_idx] = child_item;
                    let returned_item =
                        self.save_node_and_create_item(updated_child_node, child_prefix, false);
                    Ok(Some(returned_item))
                }
                None => Ok(None),
            }
        });

        match curr_item {
            Item::EmptyItem => Ok(Some(Item::Leaf {
                prefix: ins_prefix,
                value: ins_value.clone(),
            })),
            Item::Leaf {
                prefix: leaf_prefix,
                value: leaf_value,
            } => {
                assert!(
                    leaf_prefix.len() == ins_prefix.len(),
                    "Radix Tree: All Radix keys should be same length."
                );

                if leaf_prefix == ins_prefix {
                    if ins_value == leaf_value {
                        Ok(None)
                    } else {
                        Ok(Some(Item::Leaf {
                            prefix: ins_prefix,
                            value: ins_value.clone(),
                        })) // Update Leaf.
                    }
                } else {
                    // Create child node, insert existing and new leaf in this node.
                    // Intentionally not recursive for speed up.
                    let (comm_prefix, ins_prefix_rest, leaf_prefix_rest) =
                        common_prefix(ins_prefix, leaf_prefix);

                    let mut new_node = empty_node();
                    let (leaf_prefix_rest_head, leaf_prefix_rest_tail) =
                        leaf_prefix_rest.split_first().unwrap();

                    new_node[byte_to_int(*leaf_prefix_rest_head)] = Item::Leaf {
                        prefix: leaf_prefix_rest_tail.to_vec(),
                        value: leaf_value,
                    };

                    let (ins_prefix_rest_head, ins_prefix_rest_tail) =
                        ins_prefix_rest.split_first().unwrap();

                    new_node[byte_to_int(*ins_prefix_rest_head)] = Item::Leaf {
                        prefix: ins_prefix_rest_tail.to_vec(),
                        value: ins_value.clone(),
                    };

                    Ok(Some(self.save_node_and_create_item(new_node, comm_prefix, false)))
                }
            }
            Item::NodePtr {
                prefix: ptr_prefix,
                ptr,
            } => {
                assert!(
                    ptr_prefix.len() < ins_prefix.len(),
                    "Radix Tree: Radix key should be longer than NodePtr key."
                );

                let (comm_prefix, ins_prefix_rest, ptr_prefix_rest) =
                    common_prefix(ins_prefix, ptr_prefix);

                if ptr_prefix_rest.is_empty() {
                    insert_new_node_to_child(ptr, comm_prefix, ins_prefix_rest) // Add new node to existing child node.
                } else {
                    // Create child node, insert existing Ptr and new leaf in this node.
                    let mut new_node = empty_node();
                    let (ptr_prefix_rest_head, ptr_prefix_rest_tail) =
                        ptr_prefix_rest.split_first().unwrap();

                    new_node[byte_to_int(*ptr_prefix_rest_head)] = Item::NodePtr {
                        prefix: ptr_prefix_rest_tail.to_vec(),
                        ptr: ptr.clone(),
                    };

                    let (ins_prefix_rest_head, ins_prefix_rest_tail) =
                        ins_prefix_rest.split_first().unwrap();

                    new_node[byte_to_int(*ins_prefix_rest_head)] = Item::Leaf {
                        prefix: ins_prefix_rest_tail.to_vec(),
                        value: ins_value.clone(),
                    };

                    Ok(Some(self.save_node_and_create_item(new_node, comm_prefix, false)))
                }
            }
        }
    }

    /**
     * Delete leaf value from this part of tree (start from curItem).
     * Rehash and save all depend node to cacheW.
     *
     * If not found leaf with  delPrefix - return [[None]].
     * @return Updated current item.
     */
    fn delete(
        &self,
        curr_item: Item,
        del_prefix: ByteVector,
    ) -> Result<Option<Item>, RadixTreeError> {
        let delete_from_child_node: Box<
            dyn Fn(ByteVector, ByteVector, ByteVector) -> Result<Option<Item>, RadixTreeError>,
        > = Box::new(|child_ptr: ByteVector, child_prefix: ByteVector, del_prefix: ByteVector| {
            let child_node = self.load_node(child_ptr, None)?;
            let (del_prefix_head, del_prefix_tail) = del_prefix.split_first().unwrap();
            let (del_item_idx, del_item_prefix) = (byte_to_int(*del_prefix_head), del_prefix_tail);
            let child_item_opt = self
                .delete(child_node.get(del_item_idx).unwrap().clone(), del_item_prefix.to_vec())?;

            match child_item_opt {
                Some(child_item) => {
                    let mut child_node_updated = child_node.clone();
                    // child_node_updated.insert(del_item_idx, child_item);
                    child_node_updated[del_item_idx] = child_item;
                    Ok(Some(self.save_node_and_create_item(child_node_updated, child_prefix, true)))
                }
                None => Ok(None),
            }
        });

        match curr_item {
            Item::EmptyItem => Ok(None), // Not found
            Item::Leaf {
                prefix: leaf_prefix,
                value: _,
            } => {
                if leaf_prefix == del_prefix {
                    Ok(Some(Item::EmptyItem)) // Happy end
                } else {
                    Ok(None) // Not found
                }
            }
            Item::NodePtr {
                prefix: ptr_prefix,
                ptr,
            } => {
                let (comm_prefix, del_prefix_rest, ptr_prefix_rest) =
                    common_prefix(del_prefix, ptr_prefix);

                if !ptr_prefix_rest.is_empty() || del_prefix_rest.is_empty() {
                    Ok(None) // Not found
                } else {
                    delete_from_child_node(ptr, comm_prefix, del_prefix_rest) // Deeper
                }
            }
        }
    }

    /**
     * Parallel processing of [[HistoryAction]]s in this part of tree (start from curNode).
     *
     * New data load to [[cacheW]].
     * @return Updated curNode. if no action was taken - return [[None]].
     */
    pub fn make_actions(
        &self,
        curr_node: Node,
        actions: Vec<HistoryAction>,
    ) -> Result<Option<Node>, RadixTreeError> {
        // If we have 1 action in group.
        // We can't parallel next and we should use sequential traversing with help update() or delete().
        let process_one_action: Box<
            dyn Fn(HistoryAction, Item, i32) -> Result<(i32, Option<Item>), RadixTreeError>
                + Send
                + Sync,
        > = Box::new(|action: HistoryAction, item: Item, item_idx: i32| {
            // println!("\nhit process_one_action");
            let new_item = match action {
                HistoryAction::Insert(InsertAction { key, hash }) => {
                    let (_, key_tail) = key.split_first().unwrap();
                    self.update(item, key_tail.to_vec(), hash.bytes())
                }
                HistoryAction::Delete(DeleteAction { key }) => {
                    let (_, key_tail) = key.split_first().unwrap();
                    self.delete(item, key_tail.to_vec())
                }
            }?;

            Ok((item_idx, new_item))
        });

        fn clearing_delete_actions(actions: Vec<HistoryAction>, item: Item) -> Vec<HistoryAction> {
            let not_exist_insert_action = actions
                .iter()
                .find_map(|action| {
                    if let HistoryAction::Insert(_) = action {
                        Some(true)
                    } else {
                        None
                    }
                })
                .is_none();

            if item == Item::EmptyItem && not_exist_insert_action {
                Vec::new()
            } else {
                actions
            }
        }

        fn trim_keys(actions: Vec<HistoryAction>) -> Vec<HistoryAction> {
            actions
                .iter()
                .map(|action| match action {
                    HistoryAction::Insert(InsertAction { key, hash }) => {
                        let (_, key_tail) = key.split_first().unwrap();
                        HistoryAction::Insert(InsertAction {
                            key: key_tail.to_vec(),
                            hash: hash.clone(),
                        })
                    }
                    HistoryAction::Delete(DeleteAction { key }) => {
                        let (_, key_tail) = key.split_first().unwrap();
                        HistoryAction::Delete(DeleteAction {
                            key: key_tail.to_vec(),
                        })
                    }
                })
                .collect()
        }

        let process_non_empty_actions: Box<
            dyn Fn(Vec<HistoryAction>, i32) -> Result<(i32, Option<Item>), RadixTreeError>
                + Send
                + Sync,
        > = Box::new(|actions: Vec<HistoryAction>, item_idx: i32| {
            // println!("\nactions in process_non_empty_actions: {:?}", actions);
            // println!("item_idx in process_non_empty_actions: {:?}", item_idx);
            // println!("\ncurr_node: {:?}", curr_node);

            let created_node =
                self.construct_node_from_item(curr_node.get(item_idx as usize).unwrap().clone())?;

            // println!("\ncreated_node in process_non_empty_actions: {:?}", created_node);

            let new_actions = trim_keys(actions);
            let new_node_opt = self.make_actions(created_node, new_actions)?;

            // println!(
            //     "\nnew_node_opt in process_non_empty_actions: {:?}",
            //     new_node_opt.clone().unwrap().len()
            // );

            let new_item = match new_node_opt {
                Some(new_node) => Some(self.save_node_and_create_item(new_node, Vec::new(), true)),
                None => None,
            };

            // println!("\nnew_item: {:?}", new_item);

            Ok((item_idx, new_item))
        });

        // If we have more than 1 action. We can create more parallel processes.
        let process_several_actions: Box<
            dyn Fn(Vec<HistoryAction>, Item, i32) -> Result<(i32, Option<Item>), RadixTreeError>
                + Send
                + Sync,
        > = Box::new(|actions: Vec<HistoryAction>, item: Item, item_idx: i32| {
            let cleared_actions = clearing_delete_actions(actions, item);
            if cleared_actions.is_empty() {
                Ok((item_idx, None))
            } else {
                process_non_empty_actions(cleared_actions, item_idx)
            }
        });

        // Process actions within each group.
        let process_grouped_actions: Box<
            dyn Fn(
                Vec<(Byte, Vec<HistoryAction>)>,
                Node,
            ) -> Vec<Result<(i32, Option<Item>), RadixTreeError>>,
        > = Box::new(|grouped_actions: Vec<(Byte, Vec<HistoryAction>)>, curr_node: Node| {
            grouped_actions
                .par_iter()
                .map(|grouped_action| match grouped_action {
                    (group_idx, actions_in_group) => {
                        let item_idx = byte_to_int(*group_idx);
                        // println!("item_idx: {:?}", item_idx);
                        let item = curr_node[item_idx].clone();
                        if actions_in_group.len() == 1 {
                            process_one_action(
                                actions_in_group.first().unwrap().clone(),
                                item,
                                item_idx as i32,
                            )
                        } else {
                            process_several_actions(
                                actions_in_group.to_vec(),
                                item,
                                item_idx as i32,
                            )
                        }
                    }
                })
                .collect()
        });

        // Group the actions by the first byte of the prefix.
        fn grouping(
            actions: Vec<HistoryAction>,
        ) -> Result<Vec<(Byte, Vec<HistoryAction>)>, RadixTreeError> {
            let mut groups: BTreeMap<Byte, Vec<HistoryAction>> = BTreeMap::new();
            for action in actions {
                let first_byte = match action.key().first() {
                    Some(b) => *b,
                    None => {
                        return Err(RadixTreeError::PrefixError(
                            "The length of all prefixes in the subtree must be the same."
                                .to_string(),
                        ));
                    }
                };
                groups
                    .entry(first_byte)
                    .or_insert_with(Vec::new)
                    .push(action);
            }
            Ok(groups.into_iter().collect())
        }

        // Group the actions by the first byte of the prefix.
        let grouped_actions = grouping(actions)?;

        // println!("\ngrouped_actions: {:?}", grouped_actions);
        // println!("\ncurr_node: {:?}", curr_node);

        // Process actions within each group.
        // TODO: Update to handle parallel execution. See Scala side
        let new_group_items_results = process_grouped_actions(grouped_actions, curr_node.clone());

        let mut new_group_items = Vec::with_capacity(new_group_items_results.len());
        for result in new_group_items_results {
            let value = result?;
            new_group_items.push(value);
        }

        // println!("\nnew_group_items: {:?}", new_group_items);

        // Update all changed items in current node.
        let mut new_cur_node = curr_node.clone();
        for (index, new_item_opt) in new_group_items {
            new_cur_node = match new_item_opt {
                Some(new_item) => {
                    new_cur_node[index as usize] = new_item;
                    new_cur_node
                }
                None => new_cur_node,
            };
        }

        // If current node changing return new node, otherwise return none.
        if new_cur_node != curr_node {
            Ok(Some(new_cur_node))
        } else {
            Ok(None)
        }
    }

    pub fn print_tree(&self, node: &Node, prefix: Vec<u8>) -> Result<String, RadixTreeError> {
        let mut output = String::new();
        for (_, item) in node.iter().enumerate() {
            match item {
                Item::EmptyItem => {
                    // Skip empty items
                }
                Item::Leaf {
                    prefix: leaf_prefix,
                    value,
                } => {
                    let full_prefix = [prefix.clone(), leaf_prefix.clone()].concat();
                    let prefix_str = format!("{:02X?}", full_prefix);
                    let value_str = format!("{:02X?}", value);
                    output.push_str(&format!("Leaf at prefix:{}: data: {}", prefix_str, value_str));
                }
                Item::NodePtr {
                    prefix: node_prefix,
                    ptr,
                } => {
                    let full_prefix = [prefix.clone(), node_prefix.clone()].concat();
                    let prefix_str = format!("{:02X?}", full_prefix);
                    output.push_str(&format!("NodePtr at {}: ", prefix_str));

                    // Retrieve the serialized child node using the store's get method
                    let serialized_child_nodes = self.store.get(&vec![ptr.clone()])?;
                    if let Some(serialized_child_node) = serialized_child_nodes.get(0) {
                        if let Some(child_node_bytes) = serialized_child_node {
                            let child_node = bincode::deserialize::<Node>(
                                child_node_bytes.as_ref(),
                            )
                            .map_err(|_| {
                                RadixTreeError::SerializationError(
                                    "Radix Tree: Failed to deserialize".to_string(),
                                )
                            })?;
                            output.push_str(&self.print_tree(&child_node, full_prefix)?);
                        }
                    }
                }
            }
        }
        Ok(output)
    }
}

fn tail_rec_m<A, B, F>(initial_state: A, mut func: F) -> Result<B, RadixTreeError>
where
    F: FnMut(A) -> Result<Either<A, B>, RadixTreeError>,
{
    let mut state = initial_state;
    loop {
        match func(state)? {
            Either::Left(new_state) => state = new_state,
            Either::Right(final_state) => return Ok(final_state),
        }
    }
}
