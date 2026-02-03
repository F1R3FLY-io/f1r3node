// See casper/src/main/scala/coop/rchain/casper/util/rholang/RegistrySigGen.scala

use std::fmt;

use crypto::rust::{
    hash::blake2b256::Blake2b256,
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::secp256k1::Secp256k1,
    signatures::signatures_alg::SignaturesAlg,
};
use models::rhoapi::{expr::ExprInstance, Expr, Par};
use models::rust::utils::{new_etuple_par, new_gint_par};
use prost::Message;

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

#[derive(Clone, Debug)]
pub struct Contract {
    pub var_name: String,
}

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

#[derive(Clone, Debug)]
pub struct Derivation {
    pub sk: Hex,
    pub timestamp: i64,
    pub to_sign: Par,
    pub result: InsertSigned,
    pub uri: String,
}

impl fmt::Display for Derivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Pretty-print helper for Par values.
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
 5.  | 2, 4, 3,   | registry           | value = {value}
 6.  | 5,         | protobuf           | toSign = {to_sign_hex}
 7.  | 6, 1,      | secp256k1          | sig = {sig}
 8.  | 4,         | registry           | uri = {uri}
 ----+------------+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------
 */

 {result}
"#,
            sk = self.sk,
            ts = self.timestamp,
            nonce = self.result.nonce,
            pk = self.result.pk,
            value = value_str,
            to_sign_hex = to_sign_hex,
            sig = self.result.sig,
            uri = self.uri,
            result = self.result
        )
    }
}

pub struct Args {
    pub key_pair: (PrivateKey, PublicKey),
    pub timestamp: i64,
    pub contract_name: String,
}

impl Args {
    pub fn new(
        contract_name: Option<String>,
        timestamp: Option<i64>,
        sk_option: Option<PrivateKey>,
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

        Self {
            key_pair,
            timestamp,
            contract_name,
        }
    }

    pub fn parse(argv: &[String]) -> Self {
        match argv.len() {
            3 => Self::new(
                Some(argv[0].clone()),
                Some(argv[1].parse().expect("timestamp")),
                Some(PrivateKey::from_bytes(
                    &hex::decode(&argv[2]).expect("hex private key"),
                )),
            ),
            2 => Self::new(
                Some(argv[0].clone()),
                Some(argv[1].parse().expect("timestamp")),
                None,
            ),
            1 => Self::new(Some(argv[0].clone()), None, None),
            _ => Self::new(None, None, None),
        }
    }
}

pub struct RegistrySigGen;

impl RegistrySigGen {
    pub const MAX_LONG: i64 = (1i64 << 62) + ((1i64 << 62) - 1); // Long.MaxValue (2^63 - 1)

    pub fn derive_from(args: &Args) -> Derivation {
        let secp256k1 = Secp256k1;
        let (sec_key, pub_key) = &args.key_pair;

        let contract = Contract {
            var_name: args.contract_name.clone(),
        };

        // Use maximum nonce to prevent unauthorized updates
        let last_nonce = Self::MAX_LONG;

        // Prepare value to sign (tuple of timestamp, deployerPubKey, version)
        let to_sign: Par = new_etuple_par(vec![
            new_gint_par(args.timestamp, Vec::new(), false),
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(pub_key.bytes.to_vec())),
            }]),
            new_gint_par(last_nonce, Vec::new(), false),
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
            to_sign,
            result,
            uri,
        }
    }
}
