// See comm/src/main/scala/coop/rchain/comm/package.scala

use std::net::IpAddr;

use crate::rust::errors::CommError;

/// Validates an IP address using a predicate function
pub fn validate_inet_address<P>(host: &str, predicate: P) -> Result<bool, CommError>
where
    P: Fn(&IpAddr) -> bool,
{
    match host.parse::<IpAddr>() {
        Ok(addr) => Ok(predicate(&addr)),
        Err(_) => Ok(false), // Invalid address returns false, not error
    }
}

/// Checks if an IP address is valid (not any local address)
pub fn is_valid_inet_address(host: &str) -> Result<bool, CommError> {
    validate_inet_address(host, |addr| !is_any_local_address(addr))
}

/// Checks if an IP address is a valid public address (not private/local)
pub fn is_valid_public_inet_address(host: &str) -> Result<bool, CommError> {
    validate_inet_address(host, |addr| {
        !(is_any_local_address(addr)
            || is_link_local_address(addr)
            || is_loopback_address(addr)
            || is_multicast_address(addr)
            || is_site_local_address(addr))
    })
}

/// Helper function to check if address is any local address (0.0.0.0 or ::)
fn is_any_local_address(addr: &IpAddr) -> bool {
    match addr {
        IpAddr::V4(ipv4) => ipv4.is_unspecified(),
        IpAddr::V6(ipv6) => ipv6.is_unspecified(),
    }
}

/// Helper function to check if address is link local
fn is_link_local_address(addr: &IpAddr) -> bool {
    match addr {
        IpAddr::V4(ipv4) => {
            // Link-local IPv4: 169.254.0.0/16
            let octets = ipv4.octets();
            octets[0] == 169 && octets[1] == 254
        }
        IpAddr::V6(ipv6) => {
            // Link-local IPv6: fe80::/10
            let segments = ipv6.segments();
            (segments[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Helper function to check if address is loopback
fn is_loopback_address(addr: &IpAddr) -> bool {
    match addr {
        IpAddr::V4(ipv4) => ipv4.is_loopback(),
        IpAddr::V6(ipv6) => ipv6.is_loopback(),
    }
}

/// Helper function to check if address is multicast
fn is_multicast_address(addr: &IpAddr) -> bool {
    match addr {
        IpAddr::V4(ipv4) => ipv4.is_multicast(),
        IpAddr::V6(ipv6) => ipv6.is_multicast(),
    }
}

/// Helper function to check if address is site local (private)
fn is_site_local_address(addr: &IpAddr) -> bool {
    match addr {
        IpAddr::V4(ipv4) => {
            // Private IPv4 ranges:
            // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            ipv4.is_private()
        }
        IpAddr::V6(ipv6) => {
            // Site-local IPv6: fec0::/10 (deprecated but still checked)
            let segments = ipv6.segments();
            (segments[0] & 0xffc0) == 0xfec0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_inet_address() {
        // Valid public addresses
        assert!(is_valid_inet_address("8.8.8.8").unwrap());
        assert!(is_valid_inet_address("1.1.1.1").unwrap());
        assert!(is_valid_inet_address("2001:db8::1").unwrap());

        // Invalid addresses (any local)
        assert!(!is_valid_inet_address("0.0.0.0").unwrap());
        assert!(!is_valid_inet_address("::").unwrap());

        // Invalid format
        assert!(!is_valid_inet_address("invalid").unwrap());
        assert!(!is_valid_inet_address("256.256.256.256").unwrap());
    }

    #[test]
    fn test_is_valid_public_inet_address() {
        // Valid public addresses
        assert!(is_valid_public_inet_address("8.8.8.8").unwrap());
        assert!(is_valid_public_inet_address("1.1.1.1").unwrap());

        // Invalid - any local
        assert!(!is_valid_public_inet_address("0.0.0.0").unwrap());
        assert!(!is_valid_public_inet_address("::").unwrap());

        // Invalid - loopback
        assert!(!is_valid_public_inet_address("127.0.0.1").unwrap());
        assert!(!is_valid_public_inet_address("::1").unwrap());

        // Invalid - private/site local
        assert!(!is_valid_public_inet_address("192.168.1.1").unwrap());
        assert!(!is_valid_public_inet_address("10.0.0.1").unwrap());
        assert!(!is_valid_public_inet_address("172.16.0.1").unwrap());

        // Invalid - link local
        assert!(!is_valid_public_inet_address("169.254.1.1").unwrap());

        // Invalid - multicast
        assert!(!is_valid_public_inet_address("224.0.0.1").unwrap());

        // Invalid format
        assert!(!is_valid_public_inet_address("invalid").unwrap());
    }

    #[test]
    fn test_validate_inet_address_with_custom_predicate() {
        // Custom predicate: only allow loopback addresses
        let result = validate_inet_address("127.0.0.1", |addr| is_loopback_address(addr));
        assert!(result.unwrap());

        let result = validate_inet_address("8.8.8.8", |addr| is_loopback_address(addr));
        assert!(!result.unwrap());
    }

    #[test]
    fn test_helper_functions() {
        let localhost_v4: IpAddr = "127.0.0.1".parse().unwrap();
        let localhost_v6: IpAddr = "::1".parse().unwrap();
        let public_v4: IpAddr = "8.8.8.8".parse().unwrap();
        let private_v4: IpAddr = "192.168.1.1".parse().unwrap();
        let any_v4: IpAddr = "0.0.0.0".parse().unwrap();
        let link_local_v4: IpAddr = "169.254.1.1".parse().unwrap();
        let multicast_v4: IpAddr = "224.0.0.1".parse().unwrap();

        // Test loopback
        assert!(is_loopback_address(&localhost_v4));
        assert!(is_loopback_address(&localhost_v6));
        assert!(!is_loopback_address(&public_v4));

        // Test any local
        assert!(is_any_local_address(&any_v4));
        assert!(!is_any_local_address(&public_v4));

        // Test site local (private)
        assert!(is_site_local_address(&private_v4));
        assert!(!is_site_local_address(&public_v4));

        // Test link local
        assert!(is_link_local_address(&link_local_v4));
        assert!(!is_link_local_address(&public_v4));

        // Test multicast
        assert!(is_multicast_address(&multicast_v4));
        assert!(!is_multicast_address(&public_v4));
    }
}
