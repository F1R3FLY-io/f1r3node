use std::time::Duration;
use sysinfo::{CpuExt, System, SystemExt};
use tokio::time;
use crate::rust::diagnostics::SYSTEM_METRICS_SOURCE;

pub fn start_sigar_reporter(interval_duration: Duration) {
    tokio::spawn(async move {
        let mut sys = System::new_all();
        let mut interval = time::interval(interval_duration);
        loop {
            interval.tick().await;
            sys.refresh_cpu();
            sys.refresh_memory();

            let cpu_usage = sys.global_cpu_info().cpu_usage();
            let mem_usage = sys.used_memory() as f64 / sys.total_memory() as f64 * 100.0;

            metrics::gauge!("system_cpu_usage_percent", "source" => SYSTEM_METRICS_SOURCE).set(cpu_usage as f64);
            metrics::gauge!("system_memory_usage_percent", "source" => SYSTEM_METRICS_SOURCE).set(mem_usage);
        }
    });
}
