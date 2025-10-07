use crypto::rust::{
    hash::{
        blake2b256::{Blake2b256, HASH_LENGTH},
        keccak256::Keccak256,
    },
    public_key::PublicKey,
};
use models::rhoapi::GPrivate;
use prost::Message;

use super::base58;

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/util/AddressTools.scala
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Address {
    prefix: Vec<u8>,
    key_hash: Vec<u8>,
    checksum: Vec<u8>,
}

impl Address {
    pub fn to_base58(&self) -> String {
        let payload: Vec<u8> = [&self.prefix[..], &self.key_hash[..]].concat();
        let address: Vec<u8> = [&payload[..], &self.checksum[..]].concat();
        base58::encode(&address)
    }
}

#[derive(Debug)]
pub struct AddressTools {
    pub prefix: Vec<u8>,
    pub key_length: usize,
    pub checksum_length: usize,
}

impl AddressTools {
    fn compute_checksum(&self, to_check: &[u8]) -> Vec<u8> {
        let hash = Blake2b256::hash(to_check.to_vec());
        hash.into_iter().take(self.checksum_length).collect()
    }

    /**
     * Creates an Address given a public key.
     *
     * @param pk the public key from which the address is derived
     * @return None if the key length is invalid or Some if the address was created successfully
     */
    pub fn from_public_key(&self, pk: &PublicKey) -> Option<Address> {
        if self.key_length == pk.bytes.len() {
            let encoded = hex::encode(Keccak256::hash(pk.bytes[1..].to_vec()));
            let eth_address = encoded[encoded.len() - 40..].to_string();

            self.from_eth_address(&eth_address)
        } else {
            None
        }
    }

    pub fn from_eth_address(&self, eth_address: &str) -> Option<Address> {
        let eth_address_length = 40;

        let eth_address_without_prefix = if eth_address.starts_with("0x") {
            &eth_address[2..]
        } else {
            eth_address
        };

        if eth_address_without_prefix.len() == eth_address_length {
            let key_hash = Keccak256::hash(
                hex::decode(eth_address_without_prefix).expect("Invalid hex string"),
            );

            let payload = [&self.prefix[..], &key_hash[..]].concat();

            Some(Address {
                prefix: self.prefix.clone(),
                key_hash: key_hash.to_vec(),
                checksum: self.compute_checksum(&payload),
            })
        } else {
            None
        }
    }

    pub fn from_unforgeable(&self, gprivate: &GPrivate) -> Address {
        let key_hash = Keccak256::hash(gprivate.encode_to_vec());
        let payload = [&self.prefix[..], &key_hash[..]].concat();

        Address {
            prefix: self.prefix.clone(),
            key_hash: key_hash.to_vec(),
            checksum: self.compute_checksum(&payload),
        }
    }

    pub fn parse(&self, address: &str) -> Result<Address, String> {
        let checksum_start = self.prefix.len() + HASH_LENGTH;
        let address_length = self.prefix.len() + HASH_LENGTH + self.checksum_length;

        let decoded_address = match base58::decode(address) {
            Ok(bytes) => bytes,
            Err(_) => return Err("Invalid Base58 encoding".to_string()),
        };

        // validateLength
        if decoded_address.len() != address_length {
            return Err("Invalid address length".to_string());
        }

        // validateChecksum
        let (payload, checksum) = decoded_address.split_at(checksum_start);
        let computed_checksum = self.compute_checksum(payload);

        if computed_checksum != checksum {
            return Err("Invalid checksum".to_string());
        }

        // parseKeyHash
        let (actual_prefix, key_hash) = payload.split_at(self.prefix.len());

        if actual_prefix != self.prefix.as_slice() {
            return Err("Invalid prefix".to_string());
        }

        Ok(Address {
            prefix: self.prefix.clone(),
            key_hash: key_hash.to_vec(),
            checksum: checksum.to_vec(),
        })
    }
}
