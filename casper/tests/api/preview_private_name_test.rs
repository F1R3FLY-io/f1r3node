// See casper/src/test/scala/coop/rchain/casper/api/PreviewPrivateNameTest.scala

use casper::rust::api::block_api::BlockAPI;
use shared::rust::ByteString;

fn preview_id(pk_hex: &str, timestamp: i64, nth: i32) -> String {
    let deployer_bytes: ByteString = if pk_hex.is_empty() {
        vec![]
    } else {
        hex::decode(pk_hex).expect("Failed to decode hex")
    };

    let preview = BlockAPI::preview_private_names(&deployer_bytes, timestamp, nth + 1)
        .expect("Failed to preview private names");

    hex::encode(&preview[nth as usize])
}

const MY_NODE_PK: &str = "464f6780d71b724525be14348b59c53dc8795346dfd7576c9f01c397ee7523e6";

#[test]
fn preview_private_names_should_work_in_one_case() {
    // Scala comments:
    // When we deploy `new x ...` code from a javascript gRPC client,
    // we get this private name id in the log:
    // 16:41:08.995 [node-runner-15] INFO  c.r.casper.MultiParentCasperImpl - Received Deploy #1542308065454 -- new x0, x1 in {
    //   @{x1}!(...
    // [Unforgeable(0xb5630d1bfb836635126ee7f2770873937933679e38146b1ddfbfcc14d7d8a787), bundle+ {   Unforgeable(0x00) }]
    // 2018-11-15T18:54:25.454Z
    assert_eq!(
        preview_id(MY_NODE_PK, 1542308065454, 0),
        "b5630d1bfb836635126ee7f2770873937933679e38146b1ddfbfcc14d7d8a787"
    );
}

#[test]
fn preview_private_names_should_work_for_another_timestamp() {
    assert_eq!(
        preview_id(MY_NODE_PK, 1542315551822, 0),
        "d472acf9c61e276e460de567a2b709bc9b97ff6135a812abcbaa60106d2744f9"
    );
}

#[test]
fn preview_private_names_should_handle_empty_user_public_key() {
    assert_eq!(
        preview_id("", 1542308065454, 0),
        "a249b81b82572b32e9a8adc9d708be08bc85fdf19e4aca3c316e51d30b97c993"
    );
}

#[test]
fn preview_private_names_should_work_for_more_than_one_name() {
    assert_eq!(
        preview_id(MY_NODE_PK, 1542308065454, 1),
        "cdaba23ba96f28c7f443a84086e260b839cc33068d0f685648ba2ae08fd7f9da"
    );
}

