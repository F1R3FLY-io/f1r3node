extern crate tonic_prost_build;

// https://docs.rs/prost-build/latest/prost_build/struct.Config.html
// https://docs.rs/tonic-build/latest/tonic_build/struct.Builder.html#

use std::{env, fs, path::Path};

fn main() {
    let manifest_dir = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).to_path_buf();
    let proto_src_dir = manifest_dir.join("src/main/protobuf");
    let scala_proto_base_dir = manifest_dir.join("src");

    let proto_files = [
        "CasperMessage.proto",
        "DeployServiceCommon.proto",
        "DeployServiceV1.proto",
        "ProposeServiceCommon.proto",
        "ProposeServiceV1.proto",
        "RholangScalaRustTypes.proto",
        "RhoTypes.proto",
        "RSpacePlusPlusTypes.proto",
        "ServiceError.proto",
        "ExternalCommunicationServiceCommon.proto",
        "ExternalCommunicationServiceV1.proto",
        "routing.proto",
    ];

    let absolute_proto_files: Vec<_> = proto_files.iter().map(|f| proto_src_dir.join(f)).collect();

    // Tell Cargo to only rerun this build script if proto files change
    for proto_file in &absolute_proto_files {
        println!("cargo:rerun-if-changed={}", proto_file.display());
    }
    // Also watch the scalapb proto used for imports
    println!(
        "cargo:rerun-if-changed={}",
        scala_proto_base_dir.join("scalapb/scalapb.proto").display()
    );

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .btree_map(".")
        // Apply to messages only (not enums or oneofs - they are handled separately)
        .message_attribute(
            ".rhoapi",
            "#[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]",
        )
        .message_attribute(".rhoapi", "#[derive(Eq, Ord, PartialOrd)]")
        .message_attribute(".rhoapi", "#[repr(C)]")
        // Apply serde/utoipa to enums (but NOT Eq/Ord/PartialOrd - prost already derives them)
        .enum_attribute(
            ".rhoapi",
            "#[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]",
        )
        .bytes(".casper")
        .bytes(".routing")
        // needed for grpc services from deploy_grpc_service_v1.rs to avoid upper camel case warnings
        .server_mod_attribute(".", "#[allow(non_camel_case_types)]")
        .client_mod_attribute(".", "#[allow(non_camel_case_types)]")
        .compile_protos(
            &absolute_proto_files,
            &[proto_src_dir, manifest_dir, scala_proto_base_dir],
        )
        .expect("Failed to compile proto files");

    // Post-process generated rhoapi.rs:
    // 1. Remove PartialEq from Messages and Oneofs (we provide manual impls in lib.rs)
    // 2. Remove Hash from Oneofs that have manual Hash impls
    // 3. Add Eq, Ord, PartialOrd to Oneofs (so messages containing them can derive these traits)
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let file_path = format!("{}/rhoapi.rs", out_dir);
    let content = fs::read_to_string(&file_path).expect("Unable to read file");

    let modified_content = content
        .lines()
        .map(|line| {
            // Messages: Remove PartialEq (manual impl in lib.rs)
            if line.contains("#[derive(Clone, PartialEq, ::prost::Message)]") {
                line.replace("PartialEq,", "")
            } else if line.contains("#[derive(Clone, Copy, PartialEq, ::prost::Message)]") {
                line.replace("PartialEq,", "")
            } else if line.contains("#[derive(Clone, Copy, PartialEq, Eq, Hash, ::prost::Message)]") {
                line.replace("PartialEq, Eq, Hash,", "")
            } else if line.contains("#[derive(Clone, PartialEq, Eq, Hash, ::prost::Message)]") {
                line.replace("PartialEq, Eq, Hash,", "")
            }
            // Oneofs: Remove PartialEq (manual impl in lib.rs), add Eq, Ord, PartialOrd
            // For Oneofs with Hash: also remove Hash (VarInstance, UnfInstance have manual Hash)
            else if line.contains("#[derive(Clone, PartialEq, ::prost::Oneof)]") {
                line.replace(
                    "#[derive(Clone, PartialEq, ::prost::Oneof)]",
                    "#[derive(Clone, Eq, Ord, PartialOrd, ::prost::Oneof)]",
                )
            } else if line.contains("#[derive(Clone, Copy, PartialEq, ::prost::Oneof)]") {
                line.replace(
                    "#[derive(Clone, Copy, PartialEq, ::prost::Oneof)]",
                    "#[derive(Clone, Copy, Eq, Ord, PartialOrd, ::prost::Oneof)]",
                )
            } else if line.contains("#[derive(Clone, Copy, PartialEq, Eq, Hash, ::prost::Oneof)]") {
                // VarInstance: has manual Hash impl
                line.replace(
                    "#[derive(Clone, Copy, PartialEq, Eq, Hash, ::prost::Oneof)]",
                    "#[derive(Clone, Copy, Eq, Ord, PartialOrd, ::prost::Oneof)]",
                )
            } else if line.contains("#[derive(Clone, PartialEq, Eq, Hash, ::prost::Oneof)]") {
                // UnfInstance: has manual Hash impl
                line.replace(
                    "#[derive(Clone, PartialEq, Eq, Hash, ::prost::Oneof)]",
                    "#[derive(Clone, Eq, Ord, PartialOrd, ::prost::Oneof)]",
                )
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<String>>()
        .join("\n");

    fs::write(file_path, modified_content).expect("Unable to write file");
}
