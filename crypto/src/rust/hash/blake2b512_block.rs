/* See crypto/src/main/scala/coop/rchain/crypto/hash/Blake2b512Block.scala */

/**
Block oriented Blake2b512 class.

We're using some of the tree hashing parameters to achieve online tree hashing,
where the structure of the tree is provided by the application. This precludes
using the depth counters. Also, because we are online, we can't know before-hand
that a node is the last in a level, or is offset. When we hash the root, we know
that it is the root, but since we are only publishing the root hash, and not the
sub-tree, this is sufficient protection against extension.

I've (not spreston8, old dev) checked that we should be fine against the criteria here.
See: "Sufficient conditions for sound tree and sequential hashing modes" by
Guido Bertoni, Joan Daemen, Michaël Peeters, and Gilles Van Assche

We also have data at every level, so we're using the fanout parameter to
distinguish child-hashes from data. To make it convenient for block-orientation,
we will null-pad an odd number of child hashes. This means that the fanout varies per-node
rather than being set for the whole instance. Where applicable, we are setting
the other tree-parameters accordingly. Namely, max-depth to 255 (unlimited) and
inner hash length to 64.

This class is an abbreviated version of Blake2bDigest.java from BouncyCastle
https://github.com/bcgit/bc-java/blob/master/core/src/main/java/org/bouncycastle/crypto/digests/Blake2bDigest.java
  */

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Blake2b512Block {
    pub chain_value: Vec<i64>,
    pub t0: i64,
    pub t1: i64,
}

impl Blake2b512Block {
    pub fn new_from_fanout(fanout: u8) -> Blake2b512Block {
        let param0_with_fanout: i64 = Self::PARAM_VALUE_0 | ((fanout as i64 & 0xff) << 16);
        // println!("\nparam0_with_fanout: {:?}", param0_with_fanout);

        let mut result = Blake2b512Block {
            chain_value: vec![0; Self::CHAIN_VALUE_LENGTH],
            t0: 0,
            t1: 0,
        };

        result
            .chain_value
            .copy_from_slice(&Self::IV[0..Self::CHAIN_VALUE_LENGTH]);
        // println!("\nchain_value: {:?}", result.chain_value);
        result.chain_value[0] ^= param0_with_fanout;
        result.chain_value[2] ^= Self::PARAM_VALUE_2;
        // result.debug_str();
        result
    }

    pub fn new_from_src(src: Blake2b512Block) -> Blake2b512Block {
        let mut result = Blake2b512Block {
            chain_value: vec![0; Self::CHAIN_VALUE_LENGTH],
            t0: 0,
            t1: 0,
        };

        result
            .chain_value
            .copy_from_slice(&src.chain_value[0..Self::CHAIN_VALUE_LENGTH]);
        result.t0 = src.t0;
        result.t1 = src.t1;
        result
    }

    // block must be 128 bytes long
    pub fn update(&mut self, block: &[i8], offset: usize) {
        let cv = &mut self.chain_value.clone();
        self.compress(block, offset, cv, false, false, false);
    }

    // gives the output as if block were the last block processed, but does not
    // invalidate or modify internal state
    pub fn peek_final_root(
        &mut self,
        block: &[i8],
        in_offset: usize,
        output: &mut [i8],
        out_offset: usize,
    ) {
        let mut temp_chain_value = vec![0i64; 8];
        self.compress(block, in_offset, &mut temp_chain_value, true, true, true);
        // println!(
        //     "\ntemp_chain_value in peek_final_root: {:?}",
        //     temp_chain_value
        // );
        // println!("\noutput in peek_final_root: {:?}", output);
        self.long_vec_to_little_endian(&temp_chain_value, output, out_offset);
    }

    pub fn finalize_internal(
        &mut self,
        block: &[i8],
        in_offset: usize,
        output: &mut [i8],
        out_offset: usize,
    ) {
        let mut temp_chain_value = vec![0i64; 8];
        self.compress(block, in_offset, &mut temp_chain_value, true, true, false);
        for i in 0..4 {
            self.long_to_little_endian(temp_chain_value[i], output, out_offset + i * 8);
        }
    }

    fn long_to_little_endian(&self, value: i64, output: &mut [i8], offset: usize) {
        for i in 0..8 {
            output[offset + i] = (value >> (i * 8)) as i8;
        }
    }

    fn long_vec_to_little_endian(&self, values: &[i64], output: &mut [i8], offset: usize) {
        for (i, &value) in values.iter().enumerate() {
            self.long_to_little_endian(value as i64, output, offset + i * 8);
        }
    }

    pub fn compress(
        &mut self,
        msg: &[i8],
        offset: usize,
        new_chain_value: &mut Vec<i64>,
        peek: bool,
        finalize: bool,
        root_finalize: bool,
    ) {
        // println!("\nhit compress");
        let mut internal_state = [0i64; 16];
        let new_t0 = self.t0 + Self::BLOCK_LENGTH_BYTES;
        let new_t1 = if new_t0 == 0 { self.t1 + 1 } else { self.t1 };

        // Initialize internal state
        internal_state[..8].copy_from_slice(&self.chain_value);
        internal_state[8..16].copy_from_slice(&Self::IV); // Adjusted to copy the full IV

        let f0 = if finalize {
            0xFFFFFFFFFFFFFFFFu64 as i64
        } else {
            0
        };
        let f1 = if root_finalize {
            0xFFFFFFFFFFFFFFFFu64 as i64
        } else {
            0
        };

        internal_state[12] = new_t0 ^ Self::IV[4];
        internal_state[13] = new_t1 ^ Self::IV[5];
        internal_state[14] = f0 ^ Self::IV[6];
        internal_state[15] = f1 ^ Self::IV[7];

        // println!("\ninternal_state: {:?}", internal_state);
        // println!("\nchain_value: {:?}", self.chain_value);
        // println!("\nmsg: {:?}", msg);

        let mut m = [0i64; 16];

        for i in 0..Self::BLOCK_LENGTH_LONGS {
            if offset + i * 8 + 8 <= msg.len() {
                m[i] = i64::from_le_bytes(
                    msg[offset + i * 8..offset + (i + 1) * 8]
                        .iter()
                        .map(|&b| b as u8)
                        .collect::<Vec<u8>>()
                        .as_slice()
                        .try_into()
                        .unwrap(),
                );
            } else {
                m[i] = 0; // Ensure uninitialized values are set to 0
            }
        }

        // println!("\ninternal_state: {:?}", internal_state);

        for round in 0..Self::ROUNDS {
            // Columns
            self.g(
                m[Self::SIGMA[round][0]],
                m[Self::SIGMA[round][1]],
                0,
                4,
                8,
                12,
                &mut internal_state,
            );
            self.g(
                m[Self::SIGMA[round][2]],
                m[Self::SIGMA[round][3]],
                1,
                5,
                9,
                13,
                &mut internal_state,
            );
            self.g(
                m[Self::SIGMA[round][4]],
                m[Self::SIGMA[round][5]],
                2,
                6,
                10,
                14,
                &mut internal_state,
            );
            self.g(
                m[Self::SIGMA[round][6]],
                m[Self::SIGMA[round][7]],
                3,
                7,
                11,
                15,
                &mut internal_state,
            );

            // Diagonals
            self.g(
                m[Self::SIGMA[round][8]],
                m[Self::SIGMA[round][9]],
                0,
                5,
                10,
                15,
                &mut internal_state,
            );
            self.g(
                m[Self::SIGMA[round][10]],
                m[Self::SIGMA[round][11]],
                1,
                6,
                11,
                12,
                &mut internal_state,
            );
            self.g(
                m[Self::SIGMA[round][12]],
                m[Self::SIGMA[round][13]],
                2,
                7,
                8,
                13,
                &mut internal_state,
            );
            self.g(
                m[Self::SIGMA[round][14]],
                m[Self::SIGMA[round][15]],
                3,
                4,
                9,
                14,
                &mut internal_state,
            );
        }

        // println!("\ninternal_state: {:?}", internal_state);

        if !peek {
            self.t0 = new_t0;
            self.t1 = new_t1;

            for i in 0..Self::CHAIN_VALUE_LENGTH {
                self.chain_value[i] =
                    self.chain_value[i] ^ internal_state[i] ^ internal_state[i + 8];
            }
        } else {
            for i in 0..Self::CHAIN_VALUE_LENGTH {
                new_chain_value[i] =
                    self.chain_value[i] ^ internal_state[i] ^ internal_state[i + 8];
            }
        }
    }

    fn g(
        &self,
        m1: i64,
        m2: i64,
        pos_a: usize,
        pos_b: usize,
        pos_c: usize,
        pos_d: usize,
        internal_state: &mut [i64; 16],
    ) {
        let rotr64 = |x: i64, rot: u32| {
            // println!("\nx: {:?}", x);
            // println!("rot: {:?}", rot);
            let result = x.rotate_right(rot);
            // println!("result: {:?}", result);
            result
        };

        internal_state[pos_a] = internal_state[pos_a]
            .wrapping_add(internal_state[pos_b])
            .wrapping_add(m1);

        internal_state[pos_d] = rotr64(internal_state[pos_d] ^ internal_state[pos_a], 32);
        internal_state[pos_c] = internal_state[pos_c].wrapping_add(internal_state[pos_d]);
        internal_state[pos_b] = rotr64(internal_state[pos_b] ^ internal_state[pos_c], 24);

        internal_state[pos_a] = internal_state[pos_a]
            .wrapping_add(internal_state[pos_b])
            .wrapping_add(m2);

        internal_state[pos_d] = rotr64(internal_state[pos_d] ^ internal_state[pos_a], 16);
        internal_state[pos_c] = internal_state[pos_c].wrapping_add(internal_state[pos_d]);
        internal_state[pos_b] = rotr64(internal_state[pos_b] ^ internal_state[pos_c], 63);

        // println!("\ninternal_state: {:?}", internal_state);
    }

    // Produced from the square root of primes 2, 3, 5, 7, 11, 13, 17, 19.
    // The same as SHA-512 IV.
    const IV: [i64; 8] = [
        0x6A09E667F3BCC908u64 as i64,
        0xBB67AE8584CAA73Bu64 as i64,
        0x3C6EF372FE94F82Bu64 as i64,
        0xA54FF53A5F1D36F1u64 as i64,
        0x510E527FADE682D1u64 as i64,
        0x9B05688C2B3E6C1Fu64 as i64,
        0x1F83D9ABFB41BD6Bu64 as i64,
        0x5BE0CD19137E2179u64 as i64,
    ];

    // Message word permutations:
    pub const SIGMA: [[usize; 16]; 12] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    ];

    const ROUNDS: usize = 12;
    const CHAIN_VALUE_LENGTH: usize = 8;
    const BLOCK_LENGTH_BYTES: i64 = 128;
    const BLOCK_LENGTH_LONGS: usize = 16;
    // Depth = 255, Fanout = ??, Keylength = 0, Digest length = 64 bytes
    const PARAM_VALUE_0: i64 = 0xFF000040;
    // Inner length = 32 bytes
    const PARAM_VALUE_2: i64 = 0x2000;

    pub fn debug_str(&self) -> () {
        println!(
            "Blake2b512Block {{\n  chain_value: {:?},\n  t0: {},\n  t1: {}\n}}",
            self.chain_value, self.t0, self.t1
        );
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 * 8 + 2 * 8); // 8 i64 for chain_value, 2 i64 for t0 and t1
        for &value in &self.chain_value {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&self.t0.to_le_bytes());
        bytes.extend_from_slice(&self.t1.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Blake2b512Block {
        assert!(
            bytes.len() == 8 * 8 + 2 * 8,
            "Invalid byte length for Blake2b512Block"
        );

        let mut chain_value = Vec::with_capacity(8);
        for i in 0..8 {
            let start = i * 8;
            let end = start + 8;
            let value = i64::from_le_bytes(bytes[start..end].try_into().unwrap());
            chain_value.push(value);
        }

        let t0 = i64::from_le_bytes(bytes[64..72].try_into().unwrap());
        let t1 = i64::from_le_bytes(bytes[72..80].try_into().unwrap());

        Blake2b512Block {
            chain_value,
            t0,
            t1,
        }
    }
}
