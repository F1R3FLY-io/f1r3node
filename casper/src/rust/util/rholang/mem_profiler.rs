use std::sync::OnceLock;

const MEM_PROFILE_ENV: &str = "F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE";

pub fn mem_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(MEM_PROFILE_ENV) {
        Ok(v) => {
            let normalized = v.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true"
        }
        Err(_) => false,
    })
}

#[cfg(target_os = "linux")]
pub fn read_vm_rss_kb() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(not(target_os = "linux"))]
pub fn read_vm_rss_kb() -> Option<usize> {
    None
}
