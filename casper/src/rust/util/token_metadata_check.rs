// Startup validation that the on-chain TokenMetadata contract values match
// the node's local `native-token-*` configuration.
//
// This guards against the "lying API" scenario where a node joins an existing
// network but has mismatched token metadata in its config: the protocol would
// still work (the values on-chain are the only ones that matter), but the
// node's `/api/status` responses would advertise values that disagree with
// what was baked into genesis state.
//
// Caught here, the node logs a clear error explaining which value(s) disagree
// and refuses to continue. Caught at genesis ceremony time instead, the node
// would fail to sign the UnapprovedBlock and the ceremony would stall without
// a clear reason.

use models::rhoapi::Par;
use models::rust::block::state_hash::StateHash;
use rholang::rust::interpreter::rho_type::{RhoNumber, RhoString};

use crate::rust::errors::CasperError;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;

const TOKEN_METADATA_QUERY: &str = r#"
    new ret, rl(`rho:registry:lookup`), tmCh in {
      rl!(`rho:system:tokenMetadata`, *tmCh) |
      for (@(_, TokenMetadata) <- tmCh) {
        @TokenMetadata!("all", *ret)
      }
    }
"#;

/// Queries the on-chain TokenMetadata contract and returns
/// `(name, symbol, decimals)` read from the `rho:system:tokenMetadata`
/// registry entry.
pub async fn read_on_chain_token_metadata(
    runtime_manager: &RuntimeManager,
    post_state_hash: &StateHash,
) -> Result<(String, String, u32), CasperError> {
    let (result, _cost) = runtime_manager
        .play_exploratory_deploy(TOKEN_METADATA_QUERY.to_string(), post_state_hash)
        .await?;

    // The contract's "all" method returns a single tuple `(name, symbol, decimals)`
    // on the exploratory deploy return channel.
    let tuple_par = result.first().ok_or_else(|| {
        CasperError::RuntimeError(
            "TokenMetadata exploratory deploy returned no values".to_string(),
        )
    })?;

    parse_all_tuple(tuple_par).ok_or_else(|| {
        CasperError::RuntimeError(format!(
            "TokenMetadata contract returned an unexpected shape; expected (String, String, Int), got: {:?}",
            tuple_par
        ))
    })
}

fn parse_all_tuple(par: &Par) -> Option<(String, String, u32)> {
    let expr = par.exprs.first()?;
    let etuple = match expr.expr_instance.as_ref()? {
        models::rhoapi::expr::ExprInstance::ETupleBody(t) => t,
        _ => return None,
    };

    if etuple.ps.len() != 3 {
        return None;
    }

    let name = RhoString::unapply(&etuple.ps[0])?;
    let symbol = RhoString::unapply(&etuple.ps[1])?;
    let decimals = RhoNumber::unapply(&etuple.ps[2])?;

    if decimals < 0 || decimals > i64::from(u32::MAX) {
        return None;
    }

    Some((name, symbol, decimals as u32))
}

/// Compares the on-chain token metadata against the node's local config.
/// Returns `Err` with a descriptive message if any field disagrees.
///
/// This is called once after the node transitions to Running state. A mismatch
/// means the operator's config does not reflect the values baked into this
/// chain's genesis block; the safest behaviour is to abort the node so the
/// API never reports misleading values to clients.
pub async fn verify_token_metadata_matches_config(
    runtime_manager: &RuntimeManager,
    post_state_hash: &StateHash,
    config_name: &str,
    config_symbol: &str,
    config_decimals: u32,
) -> Result<(), CasperError> {
    let (on_chain_name, on_chain_symbol, on_chain_decimals) =
        read_on_chain_token_metadata(runtime_manager, post_state_hash).await?;

    // Track mismatches both as a machine-parseable list of field names
    // (used by integration tests via structured log fields) and as a
    // human-readable description (used in the returned error message).
    let mut mismatched_fields: Vec<&'static str> = Vec::new();
    let mut mismatch_descriptions: Vec<String> = Vec::new();
    if on_chain_name != config_name {
        mismatched_fields.push("native-token-name");
        mismatch_descriptions.push(format!(
            "native-token-name: config={:?}, on-chain={:?}",
            config_name, on_chain_name
        ));
    }
    if on_chain_symbol != config_symbol {
        mismatched_fields.push("native-token-symbol");
        mismatch_descriptions.push(format!(
            "native-token-symbol: config={:?}, on-chain={:?}",
            config_symbol, on_chain_symbol
        ));
    }
    if on_chain_decimals != config_decimals {
        mismatched_fields.push("native-token-decimals");
        mismatch_descriptions.push(format!(
            "native-token-decimals: config={}, on-chain={}",
            config_decimals, on_chain_decimals
        ));
    }

    if !mismatched_fields.is_empty() {
        // Emit a structured log event BEFORE returning the error so that
        // integration tests (and operators) can grep the JSON-formatted logs
        // for a stable event without regex-parsing English error text.
        // Field names are stable identifiers matching the HOCON key names.
        //
        // mismatched_fields is joined into a comma-separated string so that
        // the tracing JSON layer serializes it as a plain JSON string that
        // consumers (tests, log pipelines) can split on ',' rather than
        // parsing Rust Debug-formatted Vec syntax.
        let mismatched_fields_joined = mismatched_fields.join(",");
        tracing::error!(
            event = "native_token_metadata_mismatch",
            mismatched_fields = %mismatched_fields_joined,
            config_name = %config_name,
            on_chain_name = %on_chain_name,
            config_symbol = %config_symbol,
            on_chain_symbol = %on_chain_symbol,
            config_decimals = config_decimals,
            on_chain_decimals = on_chain_decimals,
            "native token metadata mismatch: configured values do not match \
             values baked into this network's genesis state"
        );

        return Err(CasperError::RuntimeError(format!(
            "Configured native token metadata does not match the values baked \
             into this network's genesis state. Mismatches: [{}]. \
             Update casper.genesis-block-data.native-token-* in your config to \
             match the on-chain values, or connect to a network whose genesis \
             was created with your configured values.",
            mismatch_descriptions.join("; ")
        )));
    }

    tracing::info!(
        event = "native_token_metadata_verified",
        native_token_name = %config_name,
        native_token_symbol = %config_symbol,
        native_token_decimals = config_decimals,
        "Verified on-chain token metadata matches local configuration"
    );

    Ok(())
}
