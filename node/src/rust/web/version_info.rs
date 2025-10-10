use std::env;

pub fn get_version_info() -> (&'static str, &'static str) {
    let version = env!("CARGO_PKG_VERSION");
    let git_hash = env!("GIT_HASH_SHORT");
    (version, git_hash)
}

pub fn get_version_info_str() -> String {
    let (version, git_hash) = get_version_info();
    format!("F1r3fly Node {} ({})", version, git_hash)
}
