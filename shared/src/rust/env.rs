use std::str::FromStr;

pub fn var_parsed<T: FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.parse::<T>().ok()
}

pub fn var_or<T: FromStr + Copy>(name: &str, default: T) -> T {
    var_parsed(name).unwrap_or(default)
}

pub fn var_or_filtered<T: FromStr + Copy>(
    name: &str,
    default: T,
    predicate: impl Fn(&T) -> bool,
) -> T {
    var_parsed(name).filter(predicate).unwrap_or(default)
}

pub fn var_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}
