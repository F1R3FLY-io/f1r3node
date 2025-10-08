extern crate tonic_prost_build;

use std::{env, path::Path};

fn main() {
    let manifest_dir = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).to_path_buf();
    let proto_src_dir = manifest_dir.join("src/main/protobuf");
    let scala_proto_base_dir = manifest_dir.join("src");

    // Rerun build script if proto directory would have any changes
    println!("cargo:rerun-if-changed={}", proto_src_dir.display());

    let proto_files = ["lsp.proto", "repl.proto"];

    let absolute_proto_files: Vec<_> = proto_files.iter().map(|f| proto_src_dir.join(f)).collect();

    // Rerun if any of the proto files would be changed
    for entry in absolute_proto_files.iter() {
        println!("cargo:rerun-if-changed={}", entry.display());
    }

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
