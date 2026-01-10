// See casper/src/test/scala/coop/rchain/casper/helper/TestResultCollector.scala

use std::{collections::HashMap, sync::Mutex};

use models::{
    rhoapi::{expr::ExprInstance, ListParWithRandom, Par},
    rust::{rholang::implicits::single_expr, utils::new_gbool_par},
};
use rholang::rust::interpreter::{
    contract_call::ContractCall,
    rho_type::{RhoBoolean, RhoNumber, RhoString},
    system_processes::ProcessContext,
};

struct IsAssert;

impl IsAssert {
    pub fn unapply(p: Vec<Par>) -> Option<(String, i64, Par, String, Par)> {
        match p.as_slice() {
            [test_name_par, attempt_par, assertion_par, clue_par, ack_channel_par] => {
                if let (Some(test_name), Some(attempt), Some(clue)) = (
                    RhoString::unapply(test_name_par),
                    RhoNumber::unapply(attempt_par),
                    RhoString::unapply(clue_par),
                ) {
                    Some((
                        test_name,
                        attempt,
                        assertion_par.clone(),
                        clue,
                        ack_channel_par.clone(),
                    ))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

struct IsComparison;

impl IsComparison {
    pub fn unapply(p: Par) -> Option<(Par, String, Par)> {
        if let Some(expr) = single_expr(&p) {
            match expr.expr_instance.unwrap() {
                ExprInstance::ETupleBody(etuple) => match etuple.ps.as_slice() {
                    [expected_par, operator_par, actual_par] => {
                        if let Some(operator) = RhoString::unapply(operator_par) {
                            Some((expected_par.clone(), operator, actual_par.clone()))
                        } else {
                            None
                        }
                    }
                    _ => None,
                },

                _ => None,
            }
        } else {
            None
        }
    }
}

struct IsSetFinished;

impl IsSetFinished {
    pub fn unapply(p: Vec<Par>) -> Option<bool> {
        match p.as_slice() {
            [has_finished_par] => {
                if let Some(has_finished) = RhoBoolean::unapply(has_finished_par) {
                    Some(has_finished)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum RhoTestAssertion {
    RhoAssertTrue {
        test_name: String,
        is_success: bool,
        clue: String,
    },

    RhoAssertEquals {
        test_name: String,
        expected: Par,
        actual: Par,
        clue: String,
    },

    RhoAssertNotEquals {
        test_name: String,
        unexpected: Par,
        actual: Par,
        clue: String,
    },
}

impl RhoTestAssertion {
    pub fn test_name(&self) -> &str {
        match self {
            RhoTestAssertion::RhoAssertTrue { test_name, .. } => test_name,
            RhoTestAssertion::RhoAssertEquals { test_name, .. } => test_name,
            RhoTestAssertion::RhoAssertNotEquals { test_name, .. } => test_name,
        }
    }

    pub fn clue(&self) -> &str {
        match self {
            RhoTestAssertion::RhoAssertTrue { clue, .. } => clue,
            RhoTestAssertion::RhoAssertEquals { clue, .. } => clue,
            RhoTestAssertion::RhoAssertNotEquals { clue, .. } => clue,
        }
    }

    pub fn is_success(&self) -> bool {
        match self {
            RhoTestAssertion::RhoAssertTrue { is_success, .. } => *is_success,
            RhoTestAssertion::RhoAssertEquals {
                expected, actual, ..
            } => actual == expected,
            RhoTestAssertion::RhoAssertNotEquals {
                unexpected, actual, ..
            } => actual != unexpected,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TestResult {
    pub assertions: HashMap<String, HashMap<i64, Vec<RhoTestAssertion>>>,
    pub has_finished: bool,
}

impl TestResult {
    pub fn add_assertion(&self, attempt: i64, assertion: RhoTestAssertion) -> Self {
        let current_attempt_assertions = self
            .assertions
            .get(assertion.test_name())
            .cloned()
            .unwrap_or_else(|| HashMap::new());

        let new_assertion = (attempt, {
            let mut new_assertions = current_attempt_assertions
                .get(&attempt)
                .cloned()
                .unwrap_or_else(|| Vec::new());

            new_assertions.insert(0, assertion.clone());
            new_assertions
        });

        let new_current_attempt_assertions = {
            let mut new_assertions_map = current_attempt_assertions.clone();
            new_assertions_map.insert(new_assertion.0, new_assertion.1.clone());
            new_assertions_map
        };

        Self {
            assertions: {
                let mut new_assertions = self.assertions.clone();
                new_assertions.insert(
                    assertion.test_name().to_string(),
                    new_current_attempt_assertions,
                );
                new_assertions
            },
            has_finished: self.has_finished,
        }
    }

    pub fn set_finished(&self, has_finished: bool) -> Self {
        Self {
            assertions: self.assertions.clone(),
            has_finished,
        }
    }
}

pub struct TestResultCollector {
    result: Mutex<TestResult>,
}

impl TestResultCollector {
    pub fn new() -> Self {
        Self {
            result: Mutex::new(TestResult {
                assertions: HashMap::new(),
                has_finished: false,
            }),
        }
    }

    pub fn get_result(&self) -> TestResult {
        self.result.try_lock().unwrap().clone()
    }

    pub fn update(&self, test_result: TestResult) {
        self.result.lock().unwrap().clone_from(&test_result);
    }

    pub async fn handle_message(
        &self,
        ctx: ProcessContext,
        message: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> () {
        let is_contract_call = ContractCall {
            space: ctx.space.clone(),
            dispatcher: ctx.dispatcher.clone(),
        };

        // println!("\nhit handle_message");

        if let Some((produce, _, _, assert_par)) = is_contract_call.unapply(message) {
            if let Some((test_name, attempt, assertion, clue, ack_channel)) =
                IsAssert::unapply(assert_par.clone())
            {
                if let Some((expected_or_unexpected, equals_or_not_equals_str, actual)) =
                    IsComparison::unapply(assertion.clone())
                {
                    if equals_or_not_equals_str == "==" {
                        let assertion = RhoTestAssertion::RhoAssertEquals {
                            test_name,
                            expected: expected_or_unexpected,
                            actual,
                            clue,
                        };
                        println!("\nassertion: {:?}", assertion);

                        let curr_test_result = self.get_result();
                        let new_test_result =
                            curr_test_result.add_assertion(attempt, assertion.clone());
                        self.update(new_test_result);

                        if let Err(e) = produce(
                            &vec![new_gbool_par(assertion.is_success(), Vec::new(), false)],
                            &ack_channel.clone(),
                        )
                        .await
                        {
                            eprintln!("Error producing result: {:?}", e);
                        }
                    } else if equals_or_not_equals_str == "!=" {
                        let assertion = RhoTestAssertion::RhoAssertNotEquals {
                            test_name,
                            unexpected: expected_or_unexpected,
                            actual,
                            clue,
                        };
                        println!("\nassertion: {:?}", assertion);

                        let curr_test_result = self.get_result();
                        let new_test_result =
                            curr_test_result.add_assertion(attempt, assertion.clone());
                        self.update(new_test_result);

                        if let Err(e) = produce(
                            &vec![new_gbool_par(assertion.is_success(), Vec::new(), false)],
                            &ack_channel.clone(),
                        )
                        .await
                        {
                            eprintln!("Error producing result: {:?}", e);
                        }
                    } else {
                        println!("\nreturning Unit");
                        ()
                    }
                } else if let Some(condition) = RhoBoolean::unapply(&assertion) {
                    println!("\ncondition: {:?}", condition);

                    let curr_test_result = self.get_result();
                    let new_test_result = curr_test_result.add_assertion(
                        attempt,
                        RhoTestAssertion::RhoAssertTrue {
                            test_name,
                            is_success: condition,
                            clue,
                        },
                    );
                    self.update(new_test_result);

                    if let Err(e) = produce(
                        &vec![new_gbool_par(condition, Vec::new(), false)],
                        &ack_channel.clone(),
                    )
                    .await
                    {
                        eprintln!("Error producing result: {:?}", e);
                    }
                } else {
                    println!("\nfailed to evaluate assertion: {:?}", assertion);

                    let curr_test_result = self.get_result();
                    let new_test_result = curr_test_result.add_assertion(
                        attempt,
                        RhoTestAssertion::RhoAssertTrue {
                            test_name,
                            is_success: false,
                            clue: format!("Failed to evaluate assertion: {:?}", assertion),
                        },
                    );
                    self.update(new_test_result);

                    if let Err(e) = produce(
                        &vec![new_gbool_par(false, Vec::new(), false)],
                        &ack_channel.clone(),
                    )
                    .await
                    {
                        eprintln!("Error producing result: {:?}", e);
                    }
                }
            } else if let Some(has_finished) = IsSetFinished::unapply(assert_par) {
                println!("\nhas_finished: {}", has_finished);

                let curr_test_result = self.get_result();
                let new_test_result = curr_test_result.set_finished(has_finished);
                self.update(new_test_result);
            } else {
                println!("\nreturning Unit");
                ()
            }
        } else {
            panic!("SystemProcesses: is_contract_call failed");
        }
    }
}
