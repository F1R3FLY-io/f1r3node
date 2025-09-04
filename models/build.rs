extern crate tonic_build;

// https://docs.rs/prost-build/latest/prost_build/struct.Config.html
// https://docs.rs/tonic-build/latest/tonic_build/struct.Builder.html#

use std::{env, fs, path::Path};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_src_dir = Path::new(&manifest_dir).join("src/main/protobuf");
    let scala_proto_base_dir = Path::new(&manifest_dir).join("src");

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
    let proto_include_path = proto_src_dir.to_str().unwrap().to_string();
    let manifest_dir_path = manifest_dir.clone();
    let scala_proto_include_path = scala_proto_base_dir.to_str().unwrap().to_string();

    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .btree_map(&["."])
        .message_attribute(".rhoapi", "#[derive(serde::Serialize, serde::Deserialize)]")
        .message_attribute(".rhoapi", "#[derive(Eq, Ord, PartialOrd)]")
        .message_attribute(".rhoapi", "#[repr(C)]")
        .enum_attribute(".rhoapi", "#[derive(serde::Serialize, serde::Deserialize)]")
        .enum_attribute(".rhoapi", "#[derive(Eq, Ord, PartialOrd)]")
        .enum_attribute(".rhoapi", "#[repr(C)]")
        .bytes(&[".casper", ".routing"])
        .compile_protos(
            &absolute_proto_files,
            &[
                &proto_include_path,
                &manifest_dir_path,
                &scala_proto_include_path,
            ],
        )
        .expect("Failed to compile proto files");

    // Remove PartialEq from specific generated structs from rhoapi.rs
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let file_path = format!("{}/rhoapi.rs", out_dir);
    let content = fs::read_to_string(&file_path).expect("Unable to read file");

    let modified_content = content
        .lines()
        .map(|line| {
            if line.contains("#[derive(Clone, PartialEq, ::prost::Message)]")
                || line.contains("#[derive(Clone, PartialEq, ::prost::Oneof)]")
                || line.contains("#[derive(Clone, Copy, PartialEq, ::prost::Message)]")
                || line.contains("#[derive(Clone, Copy, PartialEq, ::prost::Oneof)]")
            {
                line.replace("PartialEq,", "")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<String>>()
        .join("\n");

    fs::write(file_path, modified_content).expect("Unable to write file");
}
