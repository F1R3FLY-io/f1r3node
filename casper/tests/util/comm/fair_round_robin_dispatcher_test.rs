// See casper/src/test/scala/coop/rchain/casper/util/comm/FairRoundRobinDispatcherSpec.scala

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::Semaphore;

use casper::rust::util::comm::fair_round_robin_dispatcher::{
    Dispatch, DispatcherConfig, FairRoundRobinDispatcher,
};

/// Helper macro to create VecDeque from array
macro_rules! queue {
    ($($item:expr),* $(,)?) => {
        {
            #[allow(unused_mut)]
            let mut q = VecDeque::new();
            $(q.push_back($item);)*
            q
        }
    };
}

/// Helper macro to create HashMap
macro_rules! map {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            #[allow(unused_mut)]
            let mut m = HashMap::new();
            $(m.insert($key, $value);)*
            m
        }
    };
}

/// Test fixture equivalent to Scala's TestEnv class.
struct TestFixture {
    dispatcher: FairRoundRobinDispatcher<String, i32>,
    handled: Arc<Mutex<Vec<(String, i32)>>>,
}

impl TestFixture {
    /// Create a new test fixture.
    fn new(
        max_source_queue_size: usize,
        give_up_after_skipped: usize,
        drop_source_after_retries: usize,
    ) -> Self {
        Self::with_filter(
            max_source_queue_size,
            give_up_after_skipped,
            drop_source_after_retries,
            |_| Box::pin(async { Dispatch::Handle }),
        )
    }

    /// Create a new test fixture with a custom filter function.
    fn with_filter<F, Fut>(
        max_source_queue_size: usize,
        give_up_after_skipped: usize,
        drop_source_after_retries: usize,
        filter: F,
    ) -> Self
    where
        F: Fn(&i32) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Dispatch> + Send + 'static,
    {
        let handled = Arc::new(Mutex::new(Vec::new()));
        let handled_clone = Arc::clone(&handled);

        let handle = move |source: String, message: i32| {
            let handled = Arc::clone(&handled_clone);
            Box::pin(async move {
                handled.lock().unwrap().push((source, message));
            }) as Pin<Box<dyn Future<Output = ()> + Send>>
        };

        let config = DispatcherConfig::new(
            max_source_queue_size,
            give_up_after_skipped,
            drop_source_after_retries,
        );

        let lock = Arc::new(Semaphore::new(1));

        let dispatcher = FairRoundRobinDispatcher::new(filter, handle, config, lock);

        Self {
            dispatcher,
            handled,
        }
    }

    /// Get the dispatcher reference.
    fn dispatcher(&self) -> &FairRoundRobinDispatcher<String, i32> {
        &self.dispatcher
    }

    /// Get the list of handled messages.
    fn get_handled(&self) -> Vec<(String, i32)> {
        self.handled.lock().unwrap().clone()
    }

    /// Get the internal state for validation.
    fn get_state(
        &self,
    ) -> (
        VecDeque<String>,
        HashMap<String, VecDeque<i32>>,
        HashMap<String, usize>,
        usize,
    ) {
        self.dispatcher.get_test_state().unwrap()
    }

    /// Validate test execution with expected state.
    async fn validate<F, Fut, T>(
        &self,
        block: F,
        expected_queue: Option<VecDeque<String>>,
        expected_messages: Option<HashMap<String, VecDeque<i32>>>,
        expected_retries: Option<HashMap<String, usize>>,
        expected_skipped: Option<usize>,
        expected_handled: Option<Vec<(String, i32)>>,
    ) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let result = block().await;

        let (queue, messages, retries, skipped) = self.get_state();
        let handled = self.get_handled();

        if let Some(expected) = expected_queue {
            assert_eq!(queue, expected, "Queue mismatch");
        }

        if let Some(expected) = expected_messages {
            assert_eq!(messages, expected, "Messages mismatch");
        }

        if let Some(expected) = expected_retries {
            assert_eq!(retries, expected, "Retries mismatch");
        }

        if let Some(expected) = expected_skipped {
            assert_eq!(skipped, expected, "Skipped mismatch");
        }

        if let Some(expected) = expected_handled {
            assert_eq!(handled, expected, "Handled mismatch");
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod ensure_source_exists {
        use super::*;

        #[tokio::test]
        async fn add_a_new_source() {
            let fixture = TestFixture::new(1, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![]]),
                    Some(map!["A".to_string() => 0]),
                    None,
                    None,
                )
                .await;
        }

        #[tokio::test]
        async fn add_new_sources_to_the_head_of_the_queue() {
            let fixture = TestFixture::new(1, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["C".to_string(), "B".to_string(), "A".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    None,
                    None,
                )
                .await;
        }

        #[tokio::test]
        async fn not_add_a_source_if_it_exists() {
            let fixture = TestFixture::new(1, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![]]),
                    Some(map!["A".to_string() => 0]),
                    None,
                    None,
                )
                .await;
        }
    }

    mod is_duplicate {
        use super::*;

        #[tokio::test]
        async fn be_true_if_the_message_is_already_enqueued() {
            let fixture = TestFixture::new(10, 0, 0);

            let result: bool = fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .is_duplicate(&"A".to_string(), &1)
                            .await
                            .unwrap()
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![1]]),
                    Some(map!["A".to_string() => 0]),
                    None,
                    None,
                )
                .await;

            assert_eq!(result, true);
        }

        #[tokio::test]
        async fn be_false_if_the_message_is_not_enqueued_yet() {
            let fixture = TestFixture::new(10, 0, 0);

            let result: bool = fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .is_duplicate(&"A".to_string(), &2)
                            .await
                            .unwrap()
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![1]]),
                    Some(map!["A".to_string() => 0]),
                    None,
                    None,
                )
                .await;

            assert_eq!(result, false);
        }
    }

    mod enqueue_message {
        use super::*;

        #[tokio::test]
        async fn enqueue_the_message() {
            let fixture = TestFixture::new(10, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .set_retries("A".to_string(), 2)
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("A".to_string(), 1)
                            .await
                            .unwrap();
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![1]]),
                    Some(map!["A".to_string() => 0]),
                    None,
                    None,
                )
                .await;
        }

        #[tokio::test]
        async fn drop_the_message_if_the_queue_reached_max_size() {
            let fixture = TestFixture::new(1, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .set_retries("A".to_string(), 2)
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("A".to_string(), 2)
                            .await
                            .unwrap();
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![1]]),
                    Some(map!["A".to_string() => 2]),
                    None,
                    None,
                )
                .await;
        }
    }

    mod handle_message {
        use super::*;

        #[tokio::test]
        async fn handle_the_message() {
            let fixture = TestFixture::new(10, 0, 0);

            let result: bool = fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .handle_message(&"A".to_string())
                            .await
                            .unwrap()
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![1]]),
                    Some(map!["A".to_string() => 0]),
                    None,
                    Some(vec![("A".to_string(), 1)]),
                )
                .await;

            assert_eq!(result, true);
        }

        #[tokio::test]
        async fn not_handle_any_messages_if_the_current_source_queue_is_empty() {
            let fixture = TestFixture::new(10, 0, 0);

            let result: bool = fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .handle_message(&"B".to_string())
                            .await
                            .unwrap()
                    },
                    Some(queue!["B".to_string(), "A".to_string()]),
                    Some(map![
                        "A".to_string() => queue![1],
                        "B".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0
                    ]),
                    None,
                    Some(vec![]),
                )
                .await;

            assert_eq!(result, false);
        }
    }

    mod rotate {
        use super::*;

        #[tokio::test]
        async fn rotate_the_sources() {
            let fixture = TestFixture::new(1, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture.dispatcher().rotate().await.unwrap();
                    },
                    Some(queue!["B".to_string(), "A".to_string(), "C".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    None,
                    None,
                )
                .await;
        }

        #[tokio::test]
        async fn rotate_single_source_queue() {
            let fixture = TestFixture::new(1, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture.dispatcher().rotate().await.unwrap();
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![]]),
                    Some(map!["A".to_string() => 0]),
                    None,
                    None,
                )
                .await;
        }
    }

    mod drop_source {
        use super::*;

        #[tokio::test]
        async fn drop_the_source() {
            let fixture = TestFixture::new(1, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .drop_source(&"A".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["B".to_string()]),
                    Some(map!["B".to_string() => queue![]]),
                    Some(map!["B".to_string() => 0]),
                    None,
                    None,
                )
                .await;
        }
    }

    mod give_up {
        use super::*;

        #[tokio::test]
        async fn reset_skipped_increment_retries_and_rotate() {
            let fixture = TestFixture::new(1, 0, 1);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture.dispatcher().set_skipped(2).unwrap();
                        fixture
                            .dispatcher()
                            .give_up(&"A".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["B".to_string(), "A".to_string(), "C".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 1,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    None,
                    None,
                )
                .await;
        }

        #[tokio::test]
        async fn reset_skipped_and_drop_source() {
            let fixture = TestFixture::new(1, 0, 1);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .set_retries("A".to_string(), 2)
                            .unwrap();
                        fixture.dispatcher().set_skipped(2).unwrap();
                        fixture
                            .dispatcher()
                            .give_up(&"A".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["C".to_string(), "B".to_string()]),
                    Some(map![
                        "B".to_string() => queue![],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    None,
                    None,
                )
                .await;
        }
    }

    mod success {
        use super::*;

        #[tokio::test]
        async fn remove_message_from_queue_reset_skipped_and_rotate() {
            let fixture = TestFixture::new(1, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("C".to_string(), 3)
                            .await
                            .unwrap();
                        fixture.dispatcher().set_skipped(2).unwrap();
                        fixture
                            .dispatcher()
                            .success(&"A".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["B".to_string(), "A".to_string(), "C".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![2],
                        "C".to_string() => queue![3]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    Some(0),
                    None,
                )
                .await;
        }
    }

    mod failure {
        use super::*;

        #[tokio::test]
        async fn increment_skipped_if_skipped_less_than_give_up_after_skipped() {
            let fixture = TestFixture::new(1, 2, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("C".to_string(), 3)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .failure(&"A".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["C".to_string(), "B".to_string(), "A".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![2],
                        "C".to_string() => queue![3]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    Some(1),
                    None,
                )
                .await;
        }

        #[tokio::test]
        async fn give_up_if_skipped_greater_than_or_equal_to_give_up_after_skipped() {
            let fixture = TestFixture::new(1, 1, 1);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .enqueue_message("C".to_string(), 3)
                            .await
                            .unwrap();
                        fixture.dispatcher().set_skipped(1).unwrap();
                        fixture
                            .dispatcher()
                            .failure(&"A".to_string())
                            .await
                            .unwrap();
                    },
                    Some(queue!["B".to_string(), "A".to_string(), "C".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![2],
                        "C".to_string() => queue![3]
                    ]),
                    Some(map![
                        "A".to_string() => 1,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    Some(0),
                    None,
                )
                .await;
        }
    }

    mod dispatch_integration {
        use super::*;

        #[tokio::test]
        async fn discovers_new_source() {
            let fixture = TestFixture::new(10, 10, 10);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 1)
                            .await
                            .unwrap();
                    },
                    Some(queue!["A".to_string()]),
                    Some(map!["A".to_string() => queue![]]),
                    Some(map!["A".to_string() => 0]),
                    None,
                    Some(vec![("A".to_string(), 1)]),
                )
                .await;
        }

        #[tokio::test]
        async fn dispatch_round_robin() {
            let fixture = TestFixture::new(10, 10, 10);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("C".to_string(), 3)
                            .await
                            .unwrap();
                    },
                    Some(queue!["C".to_string(), "B".to_string(), "A".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    None,
                    Some(vec![
                        ("C".to_string(), 3),
                        ("B".to_string(), 2),
                        ("A".to_string(), 1),
                    ]),
                )
                .await;
        }

        #[tokio::test]
        async fn enqueue_when_waiting_for_message_from_other_sources() {
            let fixture = TestFixture::new(10, 10, 10);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 3)
                            .await
                            .unwrap();
                    },
                    Some(queue!["C".to_string(), "B".to_string(), "A".to_string()]),
                    Some(map![
                        "A".to_string() => queue![3],
                        "B".to_string() => queue![1, 2],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    Some(3),
                    None,
                )
                .await;
        }

        #[tokio::test]
        async fn enqueue_when_waiting_for_message_from_other_sources_and_drop_if_full() {
            let fixture = TestFixture::new(2, 10, 10);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 3)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 4)
                            .await
                            .unwrap();
                    },
                    Some(queue!["C".to_string(), "B".to_string(), "A".to_string()]),
                    Some(map![
                        "A".to_string() => queue![3],
                        "B".to_string() => queue![1, 2],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    Some(4),
                    None,
                )
                .await;
        }

        #[tokio::test]
        async fn dispatch_as_many_messages_from_buffer_as_possible() {
            let fixture = TestFixture::new(10, 10, 10);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 3)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("C".to_string(), 4)
                            .await
                            .unwrap();
                    },
                    Some(queue!["C".to_string(), "B".to_string(), "A".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![2],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0
                    ]),
                    None,
                    Some(vec![
                        ("C".to_string(), 4),
                        ("B".to_string(), 1),
                        ("A".to_string(), 3),
                    ]),
                )
                .await;
        }

        #[tokio::test]
        async fn give_up_after_skipped_messages_and_continue_with_other_sources() {
            let fixture = TestFixture::new(10, 3, 10);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 3)
                            .await
                            .unwrap();
                    },
                    Some(queue!["C".to_string(), "B".to_string(), "A".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![2],
                        "C".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 1
                    ]),
                    None,
                    Some(vec![("B".to_string(), 1), ("A".to_string(), 3)]),
                )
                .await;
        }

        #[tokio::test]
        async fn drop_source_after_retries_and_continue_with_other_sources() {
            let fixture = TestFixture::new(10, 3, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"A".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"B".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .ensure_source_exists(&"C".to_string())
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 3)
                            .await
                            .unwrap();
                    },
                    Some(queue!["A".to_string(), "B".to_string()]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![]
                    ]),
                    Some(map!["A".to_string() => 0, "B".to_string() => 0]),
                    None,
                    Some(vec![
                        ("B".to_string(), 1),
                        ("A".to_string(), 3),
                        ("B".to_string(), 2),
                    ]),
                )
                .await;
        }

        #[tokio::test]
        async fn be_effectively_disabled_with_give_up_after_skipped_0_and_drop_source_after_retries_0(
        ) {
            let fixture = TestFixture::new(10, 0, 0);

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("C".to_string(), 3)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("D".to_string(), 4)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("C".to_string(), 5)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 6)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 7)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("D".to_string(), 8)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("C".to_string(), 9)
                            .await
                            .unwrap();
                    },
                    Some(queue![
                        "B".to_string(),
                        "A".to_string(),
                        "D".to_string(),
                        "C".to_string()
                    ]),
                    Some(map![
                        "A".to_string() => queue![],
                        "B".to_string() => queue![],
                        "C".to_string() => queue![],
                        "D".to_string() => queue![]
                    ]),
                    Some(map![
                        "A".to_string() => 0,
                        "B".to_string() => 0,
                        "C".to_string() => 0,
                        "D".to_string() => 0
                    ]),
                    None,
                    Some(vec![
                        ("A".to_string(), 1),
                        ("B".to_string(), 2),
                        ("C".to_string(), 3),
                        ("D".to_string(), 4),
                        ("C".to_string(), 5),
                        ("B".to_string(), 6),
                        ("A".to_string(), 7),
                        ("D".to_string(), 8),
                        ("C".to_string(), 9),
                    ]),
                )
                .await;
        }

        #[tokio::test]
        async fn pass_the_message_on_filter_pass() {
            let fixture =
                TestFixture::with_filter(10, 0, 0, |_| Box::pin(async { Dispatch::Pass }));

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("C".to_string(), 3)
                            .await
                            .unwrap();
                    },
                    None,
                    None,
                    None,
                    None,
                    Some(vec![
                        ("A".to_string(), 1),
                        ("B".to_string(), 2),
                        ("C".to_string(), 3),
                    ]),
                )
                .await;
        }

        #[tokio::test]
        async fn drop_the_message_on_filter_drop() {
            let fixture =
                TestFixture::with_filter(10, 0, 0, |_| Box::pin(async { Dispatch::Drop }));

            fixture
                .validate(
                    || async {
                        fixture
                            .dispatcher()
                            .dispatch("A".to_string(), 1)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("B".to_string(), 2)
                            .await
                            .unwrap();
                        fixture
                            .dispatcher()
                            .dispatch("C".to_string(), 3)
                            .await
                            .unwrap();
                    },
                    None,
                    None,
                    None,
                    None,
                    Some(vec![]),
                )
                .await;
        }
    }
}
