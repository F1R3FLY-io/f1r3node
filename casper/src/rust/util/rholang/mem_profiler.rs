pub fn mem_profile_enabled() -> bool {
    false
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
