extern crate tonic_prost_build;

use std::{env, path::Path, process::Command};

fn main() {
    let manifest_dir = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).to_path_buf();
    let proto_src_dir = manifest_dir.join("src/main/protobuf");
    let scala_proto_base_dir = manifest_dir.join("src");

    let proto_src_models_dir = manifest_dir.join("../models/src/main/protobuf");

    let proto_files = ["lsp.proto", "repl.proto"];

    let models_proto_files = ["DeployServiceV1.proto", "ProposeServiceV1.proto"];

    let absolute_proto_files: Vec<_> = proto_files.iter().map(|f| proto_src_dir.join(f)).collect();
    let absolute_models_proto_files: Vec<_> = models_proto_files
        .iter()
        .map(|f| proto_src_models_dir.join(f))
        .collect();

    let proto_files = [absolute_proto_files, absolute_models_proto_files].concat();

    // Rerun if any of the proto files would be changed
    for entry in proto_files.iter() {
        println!("cargo:rerun-if-changed={}", entry.display());
    }
    // Also watch the scalapb proto used for imports
    println!(
        "cargo:rerun-if-changed={}",
        scala_proto_base_dir.join("scalapb/scalapb.proto").display()
    );

    tonic_prost_build::configure()
        .file_descriptor_set_path("build/descriptors/reflection_protos.bin")
        .build_client(true)
        .build_server(true)
        .btree_map(&".")
        .message_attribute(".", "#[repr(C)]")
        .bytes(&".")
        .compile_protos(
            &proto_files,
            &[
                proto_src_dir.clone(),
                proto_src_models_dir.clone(),
                manifest_dir.clone(),
                scala_proto_base_dir.clone(),
            ],
        )
        .expect("Failed to compile proto files");

    let git_hash = Command::new("git")
        .args(&["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|hash| hash.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_HASH_SHORT={}", git_hash);
    println!("cargo:rerun-if-changed=.git/HEAD");
}
