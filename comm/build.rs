extern crate tonic_build;

// https://docs.rs/prost-build/latest/prost_build/struct.Config.html
// https://docs.rs/tonic-build/latest/tonic_build/struct.Builder.html#

use std::{env, path::Path};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_src_dir =
        Path::new(&manifest_dir).join("src/main/protobuf/coop/rchain/comm/protocol");
    let scala_proto_base_dir = Path::new(&manifest_dir).join("src");

    let proto_files = ["kademlia.proto"];

    let absolute_proto_files: Vec<_> = proto_files.iter().map(|f| proto_src_dir.join(f)).collect();
    let proto_include_path = proto_src_dir.to_str().unwrap().to_string();
    let manifest_dir_path = manifest_dir.clone();
    let scala_proto_include_path = scala_proto_base_dir.to_str().unwrap().to_string();

    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .btree_map(&["."])
        .message_attribute(".", "#[repr(C)]")
        .enum_attribute(".", "#[repr(C)]")
        .bytes(&["."])
        .compile_protos(
            &absolute_proto_files,
            &[
                &proto_include_path,
                &manifest_dir_path,
                &scala_proto_include_path,
            ],
        )
        .expect("Failed to compile proto files");
}
