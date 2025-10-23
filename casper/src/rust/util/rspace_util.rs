// See casper/src/test/scala/coop/rchain/casper/util/RSpaceUtil.scala

use crate::rust::util::{proto_util, rholang::runtime_manager::RuntimeManager};
use models::{
    rhoapi::{expr::ExprInstance, g_unforgeable::UnfInstance, Expr, GPrivate, GUnforgeable, Par},
    rust::{
        block::state_hash::StateHash, casper::protocol::casper_message::BlockMessage,
        string_ops::StringOps,
    },
};
use rholang::rust::interpreter::pretty_printer::PrettyPrinter;

pub async fn get_data_at_public_channel(
    hash: &StateHash,
    channel: i64,
    runtime_manager: &RuntimeManager,
) -> Vec<String> {
    let channel_par = Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GInt(channel)),
        }],
        ..Default::default()
    };

    get_data_at(hash, &channel_par, runtime_manager).await
}

pub async fn get_data_at_public_channel_block(
    block: &BlockMessage,
    channel: i64,
    runtime_manager: &RuntimeManager,
) -> Vec<String> {
    let post_state_hash = proto_util::post_state_hash(block);
    get_data_at_public_channel(&post_state_hash, channel, runtime_manager).await
}

pub async fn get_data_at_private_channel(
    block: &BlockMessage,
    channel: &str,
    runtime_manager: &RuntimeManager,
) -> Vec<String> {
    let name = StringOps::unsafe_decode_hex(channel.to_string());
    let channel_par = Par {
        unforgeables: vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate { id: name })),
        }],
        ..Default::default()
    };

    let post_state_hash = proto_util::post_state_hash(block);
    get_data_at(&post_state_hash, &channel_par, runtime_manager).await
}

pub async fn get_data_at(
    hash: &StateHash,
    channel: &Par,
    runtime_manager: &RuntimeManager,
) -> Vec<String> {
    let data = runtime_manager
        .get_data(hash.clone(), channel)
        .await
        .unwrap();

    data.into_iter()
        .flat_map(|par| {
            par.exprs
                .into_iter()
                .map(|expr| PrettyPrinter::new().build_string_from_expr(&expr))
        })
        .collect()
}
