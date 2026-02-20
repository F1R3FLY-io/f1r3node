use crypto::rust::public_key::PublicKey;
use hex;
use models::rust::validator;

use super::address_tools::{Address, AddressTools};
use models::rhoapi::GPrivate;

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/util/VaultAddress.scala
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VaultAddress {
    address: Address,
}

pub const COIN_ID: &str = "000000";

pub const VERSION: &str = "00";

fn prefix() -> Vec<u8> {
    hex::decode(format!("{}{}", COIN_ID, VERSION)).expect("Invalid hex string")
}

fn tools() -> AddressTools {
    AddressTools {
        prefix: prefix(),
        key_length: validator::LENGTH,
        checksum_length: 4,
    }
}

impl VaultAddress {
    pub fn to_base58(&self) -> String {
        self.address.to_base58()
    }

    pub fn from_deployer_id(deployer_id: Vec<u8>) -> Option<VaultAddress> {
        VaultAddress::from_public_key(&PublicKey::from_bytes(&deployer_id))
    }

    pub fn from_public_key(pk: &PublicKey) -> Option<VaultAddress> {
        match tools().from_public_key(pk) {
            Some(address) => Some(VaultAddress { address }),
            None => None,
        }
    }

    pub fn from_eth_address(eth_address: &str) -> Option<VaultAddress> {
        match tools().from_eth_address(eth_address) {
            Some(address) => Some(VaultAddress { address }),
            None => None,
        }
    }

    pub fn from_unforgeable(gprivate: &GPrivate) -> VaultAddress {
        VaultAddress {
            address: { tools().from_unforgeable(gprivate) },
        }
    }

    pub fn parse(address: &str) -> Result<VaultAddress, String> {
        match tools().parse(address) {
            Ok(address) => Ok(VaultAddress { address }),
            Err(err) => Err(err),
        }
    }

    pub fn is_valid(address: &str) -> bool {
        VaultAddress::parse(address).is_ok()
    }
}
