// use blake2::{digest::consts::U32, Blake2b, Digest};
// use hex::ToHex;
use std::error::Error;

use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::history::radix_tree::{empty_node, hash_node};

// fn blake2b256_hash(data: &[u8]) -> Vec<u8> {
//     let mut hasher = Blake2b::<U32>::new();
//     hasher.update(b"hello world");
//     let hash = hasher.finalize();
//     hash.to_vec()
// }

fn main() -> Result<(), Box<dyn Error>> {
    // let node_bytes = encode(&empty_node());

    // let hash = blake2b256_hash(&node_bytes);
    // let hex_hash = hash.encode_hex::<String>();

    let (node_hash_bytes, _node_bytes) = hash_node(&empty_node());

    println!("\nNode Hash: {}", Blake2b256Hash::from_bytes(node_hash_bytes));

    Ok(())
}
