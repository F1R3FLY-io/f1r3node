extern crate tonic_prost_build;

// https://docs.rs/prost-build/latest/prost_build/struct.Config.html
// https://docs.rs/tonic-build/latest/tonic_build/struct.Builder.html#

use std::{env, path::Path};

fn main() {
    let manifest_dir = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).to_path_buf();
    let proto_src_dir = manifest_dir.join("src/main/protobuf/coop/rchain/comm/protocol");
    let scala_proto_base_dir = manifest_dir.join("src");

    let proto_files = ["kademlia.proto"];

    let absolute_proto_files: Vec<_> = proto_files.iter().map(|f| proto_src_dir.join(f)).collect();

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .btree_map(&".")
        .message_attribute(".", "#[repr(C)]")
        .enum_attribute(".", "#[repr(C)]")
        .bytes(&".")
        .compile_protos(
            &absolute_proto_files,
            &[proto_src_dir, manifest_dir, scala_proto_base_dir],
        )
        .expect("Failed to compile proto files");
}
