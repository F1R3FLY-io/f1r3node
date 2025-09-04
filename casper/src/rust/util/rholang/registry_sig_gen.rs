// See casper/src/main/scala/coop/rchain/casper/util/rholang/RegistrySigGen.scala

use std::fmt;

use crypto::rust::{
    hash::{blake2b256::Blake2b256, blake2b512_random::Blake2b512Random},
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::secp256k1::Secp256k1,
    signatures::signatures_alg::SignaturesAlg,
};
use models::{
    rhoapi::{g_unforgeable::UnfInstance, GPrivate, GUnforgeable, Par},
    rust::utils::{new_bundle_par, new_etuple_par, new_gint_par},
};
use prost::Message;

use super::tools::Tools;
use rholang::rust::interpreter::pretty_printer::PrettyPrinter;
use rholang::rust::interpreter::registry::registry::Registry;

/// Helper wrapper providing hex string formatting for byte arrays.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Hex(pub Vec<u8>);

impl Hex {
    pub fn from_slice(slice: &[u8]) -> Self {
        Self(slice.to_vec())
    }
}

impl fmt::Display for Hex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0))
    }
}

/// Simple representation of a Rholang contract (only variable name + body).
#[derive(Clone, Debug)]
pub struct Contract {
    pub var_name: String,
    pub p: Par,
}

/// Data that will be inserted into the RChain registry via `rho:registry:insertSigned:secp256k1`.
#[derive(Clone, Debug)]
pub struct InsertSigned {
    pub pk: Hex,
    pub nonce: i64,
    pub contract: Contract,
    pub sig: Hex,
}

impl InsertSigned {
    pub fn new(pk: Hex, nonce: i64, contract: Contract, sig: Hex) -> Self {
        Self {
            pk,
            nonce,
            contract,
            sig,
        }
    }
}

impl fmt::Display for InsertSigned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"
new
  {var}, rs(`rho:registry:insertSigned:secp256k1`), uriOut
in {{
  contract {var}(...) = {{
     ...
  }} |
  rs!(
    \"{pk}\".hexToBytes(),
    ({nonce}, bundle+{{*{var}}}),
    \"{sig}\".hexToBytes(),
    *uriOut
  )
}}"#,
            var = self.contract.var_name,
            pk = self.pk,
            nonce = self.nonce,
            sig = self.sig
        )
    }
}

/// Full derivation details – mirrors the Scala `Derivation` case class.
#[derive(Clone, Debug)]
pub struct Derivation {
    pub sk: Hex,
    pub timestamp: i64,
    pub uname: Par,
    pub to_sign: Par,
    pub result: InsertSigned,
    pub uri: String,
}

impl fmt::Display for Derivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Pretty-print helper for Par values.
        let mut pp_uname = PrettyPrinter::new();
        let uname_str = pp_uname.build_channel_string(&self.uname);

        let mut pp_value = PrettyPrinter::new();
        let value_str = pp_value.build_channel_string(&self.to_sign);

        let to_sign_hex = Hex::from_slice(&self.to_sign.encode_to_vec());

        write!(
            f,
            r#"
 /*
 The table below describes the required computations and their dependencies

 No. | Dependency | Computation method | Result
 ----+------------+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------
 1.  |            | given              | sk = {sk}
 2.  |            | given              | timestamp = {ts}
 3.  |            | lastNonce          | nonce = {nonce}
 4.  | 1,         | secp256k1          | pk = {pk}
 5.  | 4, 2,      | genIds             | uname = {uname}
 6.  | 3, 5,      | registry           | value = {value}
 7.  | 6,         | protobuf           | toSign = {to_sign_hex}
 8.  | 7, 1,      | secp256k1          | sig = {sig}
 9.  | 4,         | registry           | uri = {uri}
 ----+------------+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------
 */

 {result}
"#,
            sk = self.sk,
            ts = self.timestamp,
            nonce = self.result.nonce,
            pk = self.result.pk,
            uname = uname_str,
            value = value_str,
            to_sign_hex = to_sign_hex,
            sig = self.result.sig,
            uri = self.uri,
            result = self.result
        )
    }
}

/// Parameters required for the derivation process (mirrors Scala `Args`).
pub struct Args {
    pub key_pair: (PrivateKey, PublicKey),
    pub timestamp: i64,
    pub unforgeable_name: Par,
    pub contract_name: String,
}

impl Args {
    /// Equivalent to Scala `Args.apply` with optional parameters.
    pub fn new(
        contract_name: Option<String>,
        timestamp: Option<i64>,
        sk_option: Option<PrivateKey>,
        unforgeable_name_bytes: Option<Vec<u8>>, // raw id bytes if provided explicitly
    ) -> Self {
        let contract_name = contract_name.unwrap_or_else(|| "CONTRACT".to_string());
        let timestamp = timestamp.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as i64
        });

        // Generate / derive key pair
        let secp256k1 = Secp256k1;
        let key_pair = sk_option
            .map(|sk| {
                let pk = secp256k1.to_public(&sk);
                (sk, pk)
            })
            .unwrap_or_else(|| secp256k1.new_key_pair());

        // Determine unforgeable name id
        let id_bytes = unforgeable_name_bytes.unwrap_or_else(|| {
            RegistrySigGen::generate_unforgeable_name_id(&key_pair.1, timestamp)
        });

        let unforgeable_name = Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate { id: id_bytes })),
        }]);

        Self {
            key_pair,
            timestamp,
            unforgeable_name,
            contract_name,
        }
    }

    /// Parse CLI-like argv array similar to Scala implementation.
    pub fn parse(argv: &[String]) -> Self {
        match argv.len() {
            4 => Self::new(
                Some(argv[0].clone()),
                Some(argv[1].parse().expect("timestamp")),
                Some(PrivateKey::from_bytes(
                    &hex::decode(&argv[2]).expect("hex private key"),
                )),
                Some(hex::decode(&argv[3]).expect("unforgeable name")),
            ),
            3 => Self::new(
                Some(argv[0].clone()),
                Some(argv[1].parse().expect("timestamp")),
                Some(PrivateKey::from_bytes(
                    &hex::decode(&argv[2]).expect("hex private key"),
                )),
                None,
            ),
            2 => Self::new(
                Some(argv[0].clone()),
                Some(argv[1].parse().expect("timestamp")),
                None,
                None,
            ),
            1 => Self::new(Some(argv[0].clone()), None, None, None),
            _ => Self::new(None, None, None, None),
        }
    }
}

/// Ported implementation of Scala `RegistrySigGen` helper.
pub struct RegistrySigGen;

impl RegistrySigGen {
    pub const MAX_LONG: i64 = (1i64 << 62) + ((1i64 << 62) - 1); // Long.MaxValue (2^63 - 1)

    /// Generate deterministic unforgeable name sequence seed based on deploy parameters.
    pub fn generate_unforgeable_name_id(deployer: &PublicKey, timestamp: i64) -> Vec<u8> {
        let mut rnd: Blake2b512Random = Tools::unforgeable_name_rng(deployer, timestamp);
        rnd.next().into_iter().map(|b| b as u8).collect()
    }

    /// Main derivation logic – equivalent to Scala `deriveFrom`.
    pub fn derive_from(args: &Args) -> Derivation {
        let secp256k1 = Secp256k1;
        let (sec_key, pub_key) = &args.key_pair;

        // Wrap contract inside bundle to prevent unauthorized reads
        let access: Par = new_bundle_par(args.unforgeable_name.clone(), true, false);
        let contract = Contract {
            var_name: args.contract_name.clone(),
            p: access.clone(),
        };

        // Use maximum nonce to prevent unauthorized updates
        let last_nonce = Self::MAX_LONG;

        // Prepare value to sign (tuple of nonce and bundled contract)
        let to_sign: Par = new_etuple_par(vec![
            new_gint_par(last_nonce, Vec::new(), false),
            access.clone(),
        ]);

        // Serialize with Protobuf and hash with Blake2b256
        let sign_bytes = Blake2b256::hash(to_sign.encode_to_vec());
        let sig_bytes = secp256k1.sign(&sign_bytes, &sec_key.bytes);

        // Compute registry URI from deployer key hash
        let key_hash = Blake2b256::hash(pub_key.bytes.to_vec());
        let uri = Registry::build_uri(&key_hash);

        let result = InsertSigned::new(
            Hex::from_slice(&pub_key.bytes),
            last_nonce,
            contract,
            Hex::from_slice(&sig_bytes),
        );

        Derivation {
            sk: Hex::from_slice(&sec_key.bytes),
            timestamp: args.timestamp,
            uname: args.unforgeable_name.clone(),
            to_sign,
            result,
            uri,
        }
    }
}
