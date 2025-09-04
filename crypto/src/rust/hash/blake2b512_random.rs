use super::blake2b512_block::Blake2b512Block;
use byteorder::{ByteOrder, LittleEndian};
use rand::Rng;

/** Blake2b512 based splittable and mergeable random number generator
 * specialized for generating 256-bit unforgeable names.
 * splitByte and splitShort are the interfaces to make the random number
 * generator diverge.
 * Blake2b512.merge uses online tree hashing to merge two random generator
 * states.
 *
 */

// See crypto/src/main/scala/coop/rchain/crypto/hash/Blake2b512Random.scala

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Blake2b512Random {
    pub digest: Blake2b512Block,
    pub last_block: Vec<i8>,
    pub path_view: Vec<u8>,
    pub count_view: Vec<u64>,
    pub hash_array: [i8; 64],
    pub position: i64,
    pub path_position: usize,
}

impl Blake2b512Random {
    fn new(digest: Blake2b512Block, last_block: Vec<i8>) -> Blake2b512Random {
        Blake2b512Random {
            digest,
            last_block: last_block.clone(),
            path_view: vec![0; 112],
            count_view: {
                let len = last_block.len();
                let slice = &last_block[0..len];
                let long_count = slice.len() / 8;
                let mut long_buffer: Vec<u64> = Vec::with_capacity(long_count);
                for i in 0..long_count {
                    let u8_slice: &[u8] = &slice[i * 8..(i + 1) * 8]
                        .iter()
                        .map(|&x| x as u8)
                        .collect::<Vec<u8>>();
                    long_buffer.push(LittleEndian::read_u64(u8_slice));
                }

                long_buffer
            },
            hash_array: [0; 64],
            position: 0,
            path_position: 0,
        }
    }

    fn create(init: &[u8], offset: usize, length: usize) -> Blake2b512Random {
        let init_i8: Vec<i8> = init.iter().map(|&x| x as i8).collect();
        let mut result = Self::new(Blake2b512Block::new_from_fanout(0), vec![0; 128]);
        let range: Vec<usize> = if length > 127 {
            (offset..(offset + length - 127)).step_by(128).collect()
        } else {
            Vec::new()
        };

        for &base in &range {
            result.digest.update(&init_i8, base);
        }

        // println!("\nresult.digest: {:?}", result.digest);
        let partial_base = if range.is_empty() {
            offset
        } else {
            range.last().unwrap() + 128
        };

        // If there is any remainder:
        if offset + length != partial_base {
            let mut padded = vec![0i8; 128];
            let remainder_length = (offset + length) - partial_base;
            padded[..remainder_length]
                .copy_from_slice(&init_i8[partial_base..(partial_base + remainder_length)]);
            // println!("\npadded: {:?}", padded);
            result.digest.update(&padded, 0);
        }

        // result.debug_str();
        result
    }

    pub fn create_from_length(length: i32) -> Blake2b512Random {
        let mut bytes = vec![0u8; length as usize];
        rand::thread_rng().fill(&mut bytes[..]);
        Self::create(&bytes, 0, length as usize)
    }

    pub fn create_from_bytes(init: &[u8]) -> Blake2b512Random {
        Self::create(init, 0, init.len())
    }

    pub fn split_byte(&self, index: i8) -> Blake2b512Random {
        let mut split = self.copy();
        split.add_byte(index);
        split
    }

    pub fn split_short(&self, index: i16) -> Blake2b512Random {
        let mut split = self.copy();
        let packed: [u8; 2] = index.to_le_bytes();
        let packed_signed: [i8; 2] = [packed[0] as i8, packed[1] as i8];
        // println!("\npacked: {:?}", packed_signed);
        split.add_byte(packed_signed[0]);
        split.add_byte(packed_signed[1]);
        split
    }

    fn add_byte(&mut self, index: i8) {
        if self.path_position == 112 {
            // println!("\npath_position is 112");
            self.digest.update(&self.last_block, 0);
            self.last_block.fill(0);
            self.path_position = 0;
        }
        if self.path_position < 112 {
            self.path_view[self.path_position] = index as u8;
            self.last_block[self.path_position] = index;
            self.path_position += 1;
        } else {
            self.last_block.copy_within(1..112, 0); // Shift left
            self.last_block[111] = index; // Add new byte at the end
        }
    }

    fn copy(&self) -> Blake2b512Random {
        let mut clone_block = vec![0; 128];
        clone_block.copy_from_slice(&self.last_block);
        // In Rust, there's no need to rewind a Vec, as it doesn't have a position like a ByteBuffer.
        let mut result = Self::new(
            Blake2b512Block::new_from_src(self.digest.clone()),
            clone_block,
        );
        result.path_position = self.path_position;
        // println!("\nresult in copy");
        // result.debug_str();
        result
    }

    fn hash(&mut self) {
        // println!("\nhash_array: {:?}", self.hash_array);
        self.digest
            .peek_final_root(self.last_block.as_slice(), 0, &mut self.hash_array, 0);
        let low = self.count_view[0];
        // println!("\nlow: {}", low);
        if low == u64::MAX {
            let high = self.count_view[1];
            self.count_view[0] = 0;
            self.last_block[0] = 0;
            self.count_view[1] = high + 1;
            self.last_block[1] = high as i8 + 1;
        } else {
            self.count_view[0] = low + 1;
            self.last_block[112] = low as i8 + 1; // TODO: review this hardcoded index
                                                  // println!("\ncount_view: {:?}", self.count_view);
        }
    }

    pub fn next(&mut self) -> Vec<i8> {
        if self.position == 0 {
            self.hash();
            self.position = 32;
            // println!("\nhash_array 0-32: {:?}", self.hash_array);
            self.hash_array[0..32].to_vec()
        } else {
            self.position = 0;
            // println!("\nhash_array 32-64: {:?}", self.hash_array);
            self.hash_array[32..64].to_vec()
        }
    }

    pub fn merge(mut children: Vec<Blake2b512Random>) -> Blake2b512Random {
        assert!(
            children.len() >= 2,
            "Blake2b512Random should have at least 2 inputs to merge."
        );

        let mut squashed_builder = Vec::new();
        let mut chain_block = vec![0i8; 128];

        while children.len() > 1 {
            let chunks = children.chunks_mut(255).collect::<Vec<_>>();

            for slice in chunks {
                let mut result = Self::new(
                    Blake2b512Block::new_from_fanout(slice.len() as u8),
                    vec![0; 128],
                );
                for quad in slice.chunks_mut(4) {
                    for (i, child) in quad.iter_mut().enumerate() {
                        child.digest.finalize_internal(
                            child.last_block.as_slice(),
                            0,
                            &mut chain_block,
                            i * 32,
                        );
                    }
                    if quad.len() != 4 {
                        let blank_length = (4 - quad.len()) * 32;
                        if blank_length > 0 {
                            let mut blank = Blake2b512Random::BLANK_BLOCK;
                            let chain_block_i8: Vec<i8> =
                                chain_block.iter().map(|&b| b as i8).collect();
                            let chain_block_u8: Vec<u8> =
                                chain_block_i8.iter().map(|&b| b as u8).collect();
                            blank
                                .get_mut(..blank_length)
                                .unwrap()
                                .copy_from_slice(&chain_block_u8[quad.len() * 32..]);
                        }
                    }
                    let chain_block_i8: Vec<i8> = chain_block.iter().map(|&b| b as i8).collect();
                    result.digest.update(&chain_block_i8, 0);
                }
                squashed_builder.push(result.clone());
            }
            children = squashed_builder;
            squashed_builder = Vec::new();
        }

        children.remove(0)
    }

    const BLANK_BLOCK: [u8; 128] = [0; 128];

    pub fn debug_str(&self) -> () {
        let rot_position = ((self.position - 1) & 0x3f) + 1;
        self.digest.debug_str();
        println!("last_block: {:?}", self.last_block);
        println!("path_position: {:?}", self.path_position);
        // println!("count_view: {:?}", self.count_view);
        println!("position: {}", self.position);
        println!("rot_position: {}", rot_position);
        println!(
            "remainder: {:?}",
            &self.hash_array[rot_position as usize..64]
        );
        println!("hash_array: {:?}", &self.hash_array);
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize the digest
        bytes.extend_from_slice(&self.digest.to_bytes());

        // Serialize last_block
        bytes.extend_from_slice(&(self.last_block.len() as u32).to_le_bytes());
        for &value in &self.last_block {
            bytes.push(value as u8);
        }

        // Serialize path_view
        bytes.extend_from_slice(&(self.path_view.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.path_view);

        // Serialize count_view
        bytes.extend_from_slice(&(self.count_view.len() as u32).to_le_bytes());
        for &value in &self.count_view {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        // Serialize hash_array
        bytes.extend_from_slice(
            &self
                .hash_array
                .iter()
                .map(|&x| x as u8)
                .collect::<Vec<u8>>(),
        );

        // Serialize position
        bytes.extend_from_slice(&self.position.to_le_bytes());

        // Serialize path_position
        bytes.extend_from_slice(&(self.path_position as u32).to_le_bytes());

        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Blake2b512Random {
        let mut offset = 0;

        // Deserialize the digest
        let digest = Blake2b512Block::from_bytes(&bytes[offset..offset + 80]);
        offset += 80;

        // Deserialize last_block
        let last_block_len =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let last_block = bytes[offset..offset + last_block_len]
            .iter()
            .map(|&x| x as i8)
            .collect();
        offset += last_block_len;

        // Deserialize path_view
        let path_view_len =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let path_view = bytes[offset..offset + path_view_len].to_vec();
        offset += path_view_len;

        // Deserialize count_view
        let count_view_len =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut count_view = Vec::with_capacity(count_view_len);
        for _ in 0..count_view_len {
            let value = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            count_view.push(value);
            offset += 8;
        }

        // Deserialize hash_array
        let hash_array = bytes[offset..offset + 64]
            .iter()
            .map(|&x| x as i8)
            .collect::<Vec<i8>>()
            .try_into()
            .unwrap();
        offset += 64;

        // Deserialize position
        let position = i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Deserialize path_position
        let path_position =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;

        Blake2b512Random {
            digest,
            last_block,
            path_view,
            count_view,
            hash_array,
            position,
            path_position,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake2b512_random_should_correctly_create_from_string() {
        let rand = Blake2b512Random::create_from_bytes("testing".as_bytes());
        assert_eq!(
            rand.digest.chain_value,
            [
                -8360952207520771867,
                7767624583803025097,
                -5596376288506801394,
                -2188126373757815809,
                -4217540961114877052,
                6292414413621190116,
                1078587327606774112,
                3428071545592311926
            ]
        )
    }

    #[test]
    fn blake2b512_random_should_handle_split_short() {
        let rand = Blake2b512Random::create_from_bytes(&[]);
        let mut split = rand.split_short(0x6487);
        // split.debug_str();

        let res1 = split.next();
        let res2 = split.next();
        // println!("\nres1: {:?}", res1);
        // println!("\nres2: {:?}", res2);

        assert_eq!(
            res1,
            [
                116, 92, -32, -11, -102, -89, -21, -83, -61, 28, 9, 113, 38, -84, -123, -121, 12,
                51, 100, -75, 97, -47, -40, 25, 53, -21, 1, -17, 89, 104, -44, -77
            ]
        );
        assert_eq!(
            res2,
            [
                45, 123, -46, 25, -28, -50, 30, 24, -29, -116, 6, -18, -51, -15, 112, -104, -19,
                73, -42, 104, -112, 8, 109, 25, 84, 58, -124, -2, -120, -40, 13, 103
            ]
        );
    }

    #[test]
    fn blake2b512_random_should_handle_merge() {
        let b2random_base = Blake2b512Random::create_from_bytes(&[]);
        let b2random_0 = b2random_base.split_byte(0);
        // b2random_0.debug_str();
        let b2random_1 = b2random_base.split_byte(1);
        // b2random_1.debug_str();
        let mut single_merge_random = Blake2b512Random::merge(vec![b2random_0, b2random_1]);
        // single_merge_random.debug_str();

        let res1 = single_merge_random.next();
        let res2 = single_merge_random.next();
        // println!("\nres1: {:?}", res1);
        // println!("\nres2: {:?}", res2);

        assert_eq!(
            res1,
            [
                -50, 25, 15, 66, -125, -44, -79, 22, 83, -53, 120, -18, -113, -68, 104, -91, -72,
                -53, 98, 81, 26, 31, 46, -45, -24, 54, 64, 14, 98, 20, 79, -87
            ]
        );
        assert_eq!(
            res2,
            [
                70, 14, -111, 63, -74, -14, 37, 15, -79, -82, 44, -42, -50, -21, 85, 1, -80, -40,
                59, 41, -85, -43, 56, -45, 80, -114, -58, -124, 89, 4, 52, 45
            ]
        );
    }

    #[test]
    fn blake2b512_random_should_correctly_implement_wraparound() {
        let mut rand = Blake2b512Random::create_from_bytes(&[]);
        let res1 = rand.next();
        // println!("\nres1: {:?}", res1);
        // rand.debug_str();
        assert_eq!(
            res1,
            [
                82, -120, 78, -100, -6, -9, 56, 112, -99, 39, 30, -100, 2, 104, -16, 89, 100, 57,
                86, 120, -39, -52, -42, 27, 24, 125, 103, 34, 74, 70, 66, 48
            ]
        );

        let res2 = rand.next();
        // println!("\nres2: {:?}", res2);
        // rand.debug_str();
        assert_eq!(
            res2,
            [
                -49, -94, -21, -71, 17, -123, -23, 118, 74, 90, -17, 108, 127, 90, 103, 86, -49,
                -32, -12, -118, 51, -59, -65, -85, -1, -38, -54, -59, 93, 50, -14, 74
            ]
        );

        let res3 = rand.next();
        // println!("\nres3: {:?}", res3);
        assert_eq!(
            res3,
            [
                -92, 36, -34, 33, 10, 100, -106, 105, -1, -111, -105, 111, -35, -127, -27, 89, -77,
                -75, -64, -38, 118, -116, -68, -68, 48, -46, 94, -24, 110, 62, 7, -77
            ]
        );

        let res4 = rand.next();
        // println!("\nres4: {:?}", res4);
        assert_eq!(
            res4,
            [
                12, 5, -2, -6, 45, 30, -53, -102, -12, 120, -73, 79, 94, -103, 112, 60, 66, -64,
                124, -75, 89, -90, -96, -114, 24, -2, 77, -7, -120, -24, -108, 96
            ]
        );
    }

    #[test]
    fn blake2b512_random_should_be_able_to_go_into_and_out_of_bytes() {
        let mut split_rand_result = Blake2b512Random::create_from_bytes(&[]).split_byte(3);
        split_rand_result.next();
        let merge_rand = Blake2b512Random::merge(vec![
            split_rand_result.split_byte(1),
            split_rand_result.split_byte(0),
        ]);
        // merge_rand.debug_str();

        let bytes = merge_rand.to_bytes();
        let from_bytes = Blake2b512Random::from_bytes(&bytes);
        // from_bytes.debug_str();

        assert_eq!(merge_rand, from_bytes);
    }
}
