use models::rhoapi::{
    tagged_continuation::TaggedCont, BindPattern, ListParWithRandom, PCost, Par, ParWithRandom,
    TaggedContinuation,
};
use rspace_plus_plus::rspace::hashing::blake2b256_hash;
use shared::rust::ByteString;
use std::borrow::Cow;
use std::ops::{Add, Mul, Sub};

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/accounting/Costs.scala
#[derive(Debug, Clone, PartialEq, Default, Eq, Hash)]
pub struct Cost {
    pub value: i64,
    pub operation: Cow<'static, str>,
}

impl Sub for Cost {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Cost {
            value: self.value - other.value,
            operation: Cow::Borrowed("subtraction"),
        }
    }
}

impl Add for Cost {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Cost {
            value: self.value + other.value,
            operation: Cow::Borrowed("addition"),
        }
    }
}

impl Mul for Cost {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Cost {
            value: self.value * other.value,
            operation: Cow::Borrowed("multiplication"),
        }
    }
}

impl Cost {
    pub fn create<S>(value: i64, operation: S) -> Cost
    where
        S: Into<Cow<'static, str>>,
    {
        Cost {
            value,
            operation: operation.into(),
        }
    }

    pub fn create_from_cost(cost: Cost) -> Cost {
        Cost {
            value: cost.value,
            operation: cost.operation,
        }
    }

    // See rholang/src/main/scala/coop/rchain/rholang/interpreter/accounting/Chargeable.scala
    pub fn create_from_generic<A: prost::Message, S>(term: A, operation: S) -> Cost
    where
        S: Into<Cow<'static, str>>,
    {
        Cost {
            value: term.encoded_len() as i64,
            operation: operation.into(),
        }
    }

    pub fn unsafe_max() -> Self {
        Cost::create(i64::MAX, "unsafe_max creation")
    }

    // TODO: Fix to remove conversion to u64
    pub fn to_proto(cost: Cost) -> PCost {
        PCost {
            cost: cost.value as u64,
        }
    }
}

pub fn sum_cost() -> Cost {
    Cost::create(3, "sum")
}

pub fn subtraction_cost() -> Cost {
    Cost::create(3, "subtraction")
}

pub fn subtraction_cost_with_value(value: i64) -> Cost {
    Cost::create(value, "subtraction")
}

pub fn equality_check_cost<T: prost::Message, P: prost::Message>(x: &T, y: &P) -> Cost {
    let size_x = x.encoded_len();
    let size_y = y.encoded_len();
    let min_size = std::cmp::min(size_x, size_y);

    Cost {
        value: min_size as i64,
        operation: Cow::Borrowed("equality check"),
    }
}

pub fn boolean_and_cost() -> Cost {
    Cost::create(2, "boolean and")
}

pub fn boolean_or_cost() -> Cost {
    Cost::create(2, "boolean or")
}

pub fn comparison_cost() -> Cost {
    Cost::create(3, "comparison")
}

pub fn multiplication_cost() -> Cost {
    Cost::create(9, "multiplication")
}

pub fn division_cost() -> Cost {
    Cost::create(9, "division")
}

pub fn modulo_cost() -> Cost {
    Cost::create(9, "modulo")
}

// operations on collections
// source: https://docs.scala-lang.org/overviews/collections/performance-characteristics.html
pub fn lookup_cost() -> Cost {
    Cost::create(3, "lookup")
}

pub fn remove_cost() -> Cost {
    Cost::create(3, "remove")
}

pub fn add_cost() -> Cost {
    Cost::create(3, "addition")
}

// decoding to bytes is linear with respect to the length of the string
pub fn hex_to_bytes_cost(str: &String) -> Cost {
    Cost::create(str.len() as i64, "hex to bytes")
}

// encoding to hex is linear with respect to the length of the byte array
pub fn bytes_to_hex_cost(bytes: &Vec<u8>) -> Cost {
    Cost::create(bytes.len() as i64, "bytes to hex")
}

// Both Set#remove and Map#remove have complexity of eC
pub fn diff_cost(num_elements: i64) -> Cost {
    Cost::create(
        remove_cost().value * num_elements,
        format!("{} elements diff cost", num_elements),
    )
}

// Both Set#add and Map#add have complexity of eC
pub fn union_cost(num_elements: i64) -> Cost {
    Cost::create(
        add_cost().value * num_elements,
        format!("{} union cost", num_elements),
    )
}

// GByteArray uses ByteString internally which in turn are implemented using
// data structure called Rope for which append operation is O(logN)
pub fn byte_array_append_cost(left: ByteString) -> Cost {
    if left.is_empty() {
        Cost::create(0, "byte array append")
    } else {
        let size = left.len() as f64;
        Cost::create(size.log(10.0) as i64, "byte array append")
    }
}

// According to scala doc Vector#append is eC so it's n*eC.
pub fn list_append_cost(right: Vec<Par>) -> Cost {
    Cost::create(right.len() as i64, "list append")
}

// String append creates a char[] of size n + m and then copies all elements to it.
pub fn string_append_cost(n: i64, m: i64) -> Cost {
    Cost::create(n + m, "string append")
}

// To interpolate we traverse whole base string and for each placeholder
// we look for matching key in the interpolation map
pub fn interpolate_cost(str_length: i64, map_size: i64) -> Cost {
    Cost::create(str_length * map_size, "interpolate")
}

// serializing any Par into a Array[Byte]:
// + allocates byte array of the same size as `serializedSize`
// + then it copies all elements of the Par
pub fn to_byte_array_cost(message: &impl prost::Message) -> Cost {
    Cost::create(message.encoded_len() as i64, "to byte array")
}

pub fn size_method_cost(size: i64) -> Cost {
    Cost::create(size, "size")
}

// slice(from, to) needs to drop `from` elements and then append `to - from` elements
// we charge proportionally to `to` and fail if the method call is incorrect, for example
// if underlying string is shorter then the `to` value.
pub fn slice_cost(to: i64) -> Cost {
    Cost::create(to, "slice")
}

pub fn take_cost(to: i64) -> Cost {
    Cost::create(to, "take")
}

pub fn to_list_cost(size: i64) -> Cost {
    Cost::create(size, "to_list")
}

pub fn parsing_cost(term: &str) -> Cost {
    Cost::create(term.as_bytes().len() as i64, "parsing")
}

pub fn nth_method_call_cost() -> Cost {
    Cost::create(10, "nth method call")
}

pub fn keys_method_cost() -> Cost {
    Cost::create(10, "keys method")
}

pub fn length_method_cost() -> Cost {
    Cost::create(10, "length method")
}

pub fn method_call_cost() -> Cost {
    Cost::create(10, "method call")
}

pub fn op_call_cost() -> Cost {
    Cost::create(10, "op call")
}

pub fn var_eval_cost() -> Cost {
    Cost::create(10, "var eval")
}

pub fn send_eval_cost() -> Cost {
    Cost::create(11, "send eval")
}

pub fn receive_eval_cost() -> Cost {
    Cost::create(11, "receive eval")
}

pub fn channel_eval_cost() -> Cost {
    Cost::create(11, "channel eval")
}

// The idea is that evaluation of `new x1, x2, …, xn in { }` should be charged depending
// on the # of bindings and constant cost of evaluating `new … in  { … }` construct
pub fn new_binding_cost() -> Cost {
    Cost::create(2, "new binding")
}

pub fn new_eval_cost() -> Cost {
    Cost::create(10, "new eval")
}

pub fn new_bindings_cost(n: i64) -> Cost {
    Cost::create(
        (new_binding_cost().value * n) + new_eval_cost().value,
        "send eval",
    )
}

pub fn match_eval_cost() -> Cost {
    Cost::create(12, "match eval")
}

pub fn storage_cost_consume(
    channels: Vec<Par>,
    patterns: Vec<BindPattern>,
    continuation: TaggedContinuation,
) -> Cost {
    let body_cost = Some(continuation).and_then(|cont| {
        if let Some(TaggedCont::ParBody(ParWithRandom { body, .. })) = cont.tagged_cont {
            Some(storage_cost(&vec![body.unwrap()]))
        } else {
            None
        }
    });

    let total_cost = storage_cost(&channels).value
        + storage_cost(&patterns).value
        + body_cost.unwrap_or(Cost::create(0, "")).value;

    Cost::create(total_cost, "consume storage")
}

pub fn storage_cost_produce(channel: Par, data: ListParWithRandom) -> Cost {
    Cost::create(
        storage_cost(&vec![channel]).value + storage_cost(&data.pars).value,
        "produces storage",
    )
}

pub fn comm_event_storage_cost(channels_involved: i64) -> Cost {
    let consume_cost = event_storage_cost(channels_involved);
    let produce_costs = event_storage_cost(1).value * channels_involved;
    Cost::create(
        consume_cost.value + produce_costs,
        "comm event storage cost",
    )
}

pub fn event_storage_cost(channels_involved: i64) -> Cost {
    Cost::create(
        blake2b256_hash::LENGTH + channels_involved * blake2b256_hash::LENGTH,
        "event storage cost",
    )
}

fn storage_cost<A: prost::Message>(as_: &[A]) -> Cost {
    let total_size: usize = as_.iter().map(|a| a.encoded_len()).sum();
    Cost::create(total_size as i64, "storage cost")
}
