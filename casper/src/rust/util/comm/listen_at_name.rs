// See casper/src/main/scala/coop/rchain/casper/util/comm/ListenAtName.scala

use std::collections::HashMap;
use std::time::Duration;

use crate::rust::util::comm::ServiceResult;
use crate::rust::util::rholang::interpreter_util;
use models::rhoapi::g_unforgeable::UnfInstance;
use models::rhoapi::{GPrivate, GUnforgeable, Par};
use serde::Deserialize;
use tokio::time::sleep;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub enum Name {
    /// Public name with string content
    PubName(String),
    /// Private name with string content
    PrivName(String),
}

/// Trait for building Par from different container types
pub trait BuildPar<T> {
    fn build(&self, input: T) -> ServiceResult<Par>;
}

/// Implementation for single Name
impl BuildPar<Name> for Name {
    fn build(&self, _input: Name) -> ServiceResult<Par> {
        build_par_id(self.clone())
    }
}

/// Build Par from a single Name
pub fn build_par_id(name: Name) -> ServiceResult<Par> {
    match name {
        Name::PubName(content) => {
            // Use the existing mk_term function from interpreter_util
            let normalizer_env = HashMap::new(); // Empty normalizer environment
            interpreter_util::mk_term(&content, normalizer_env)
                .map_err(|e| vec![format!("Failed to parse public name '{}': {}", content, e)])
        }
        Name::PrivName(content) => {
            // Create a GPrivate Par from the content
            let g_private = Par::default().with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                    id: content.as_bytes().to_vec(),
                })),
            }]);
            Ok(g_private)
        }
    }
}

/// Apply a function until a break condition is met
async fn apply_until<F, Fut, T>(
    mut retrieve: F,
    break_cond: impl Fn(&T) -> bool,
) -> ServiceResult<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = ServiceResult<T>>,
{
    loop {
        sleep(Duration::from_secs(1)).await;
        let data = retrieve().await?;
        if break_cond(&data) {
            return Ok(data);
        }
    }
}

/// Listen at name until changes occur
///
/// This function monitors a name and waits until the number of results increases,
/// indicating that new data has been added.
/// Note: this
pub async fn listen_at_name_until_changes<F, Fut, T>(name: Name, mut request: F) -> Fut::Output
where
    F: FnMut(Par) -> Fut,
    Fut: std::future::Future<Output = ServiceResult<Vec<T>>>,
    T: std::fmt::Debug,
{
    println!("Listen at name: {:?}", name);
    println!("Start monitoring for changes");

    // Build Par from the name
    let par = build_par_id(name)?;

    // Get initial data
    let init_size = 1; // to be consistent with Scala version where ID.size is hardcoded to 1

    println!("Initial data size: {}", init_size);

    // Monitor until size increases
    let result = apply_until(|| request(par.clone()), |data| data.len() > init_size).await?;

    println!("Detected changes:");
    let new_data = result.len() - init_size;
    if new_data > 0 {
        println!("New items count: {}", new_data);
        // Print the new items (last new_data items)
        println!("New item: {:?}", result.iter().skip(init_size));
    }

    Ok(result)
}

/// Listen at multiple names until changes occur
pub async fn listen_at_names_until_changes<F, Fut, T>(names: Vec<Name>, request: F) -> Fut::Output
where
    F: Fn(Vec<Par>) -> Fut,
    Fut: std::future::Future<Output = ServiceResult<Vec<T>>>,
    T: std::fmt::Debug,
{
    println!("Listen at names: {:?}", names);
    println!("Start monitoring for changes");

    // Build Pars from the names
    let pars: ServiceResult<Vec<Par>> = names
        .iter()
        .map(|name| build_par_id(name.clone()))
        .collect();
    let pars = pars?;

    // Get initial data
    let init = request(pars.clone()).await?;
    let init_size = init.len();

    println!("Initial data size: {}", init_size);

    // Monitor until size increases
    let result = apply_until(|| request(pars.clone()), |data| data.len() > init_size).await?;

    println!("Detected changes:");
    let new_data = result.len() - init_size;
    if new_data > 0 {
        println!("New items count: {}", new_data);
        // Print the new items (last new_data items)
        println!("New item: {:?}", result.iter().skip(init_size));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;

    #[test]
    fn test_build_par_pub_name() {
        let name = Name::PubName("0".to_string());
        let result = build_par_id(name);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_par_priv_name() {
        let name = Name::PrivName("private_test".to_string());
        let result = build_par_id(name);
        assert!(result.is_ok());

        let par = result.unwrap();
        assert!(!par.unforgeables.is_empty());
        assert!(matches!(
            par.unforgeables[0].unf_instance,
            Some(models::rhoapi::g_unforgeable::UnfInstance::GPrivateBody(_))
        ));
    }

    #[test]
    fn test_build_par_invalid_pub_name() {
        let name = Name::PubName("invalid rholang syntax {".to_string());
        let result = build_par_id(name);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_listen_at_name_until_changes() {
        let name = Name::PubName("0".to_string());

        // Mock request function that returns increasing data
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let request = move |_par: Par| {
            let call_count = Arc::clone(&call_count_clone);
            async move {
                let count = call_count.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(vec![0; count])
            }
        };

        // This should complete quickly since we're just testing the logic
        let result = listen_at_name_until_changes(name, request).await;
        assert!(result.is_ok());
    }
}
