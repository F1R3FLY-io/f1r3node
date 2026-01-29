// See casper/src/main/scala/coop/rchain/casper/util/EventConverter.scala

use models::rust::casper::protocol::casper_message::CommEvent;
use models::rust::casper::protocol::casper_message::ConsumeEvent;
use models::rust::casper::protocol::casper_message::Event;
use models::rust::casper::protocol::casper_message::Peek;
use models::rust::casper::protocol::casper_message::ProduceEvent;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::trace::event::Consume;
use rspace_plus_plus::rspace::trace::event::Event as RspaceEvent;
use rspace_plus_plus::rspace::trace::event::IOEvent;
use rspace_plus_plus::rspace::trace::event::Produce;
use rspace_plus_plus::rspace::trace::event::COMM as RspaceComm;

pub fn to_casper_event(event: RspaceEvent) -> Event {
    match event {
        RspaceEvent::Comm(RspaceComm {
            consume,
            produces,
            peeks,
            times_repeated,
        }) => Event::Comm(CommEvent {
            consume: ConsumeEvent {
                channels_hashes: consume
                    .channel_hashes
                    .iter()
                    .map(|h| h.to_bytes_prost())
                    .collect(),
                hash: consume.hash.to_bytes_prost(),
                persistent: consume.persistent,
            },
            produces: produces
                .into_iter()
                .map(|p| ProduceEvent {
                    channels_hash: p.channel_hash.to_bytes_prost(),
                    hash: p.hash.to_bytes_prost(),
                    persistent: p.persistent,
                    times_repeated: *times_repeated.get(&p).unwrap_or(&0),
                    is_deterministic: p.is_deterministic,
                    output_value: p.output_value.clone().into_iter().map(|v| v.into()).collect(),
                    failed: p.failed,
                })
                .collect(),
            peeks: peeks.iter().map(|p| Peek { channel_index: *p }).collect(),
        }),

        RspaceEvent::IoEvent(ioevent) => match ioevent {
            IOEvent::Produce(produce) => Event::Produce(ProduceEvent {
                channels_hash: produce.channel_hash.to_bytes_prost(),
                hash: produce.hash.to_bytes_prost(),
                persistent: produce.persistent,
                times_repeated: 0,
                is_deterministic: produce.is_deterministic,
                output_value: produce.output_value.into_iter().map(|v| v.into()).collect(),
                failed: produce.failed,
            }),

            IOEvent::Consume(consume) => Event::Consume(ConsumeEvent {
                channels_hashes: consume
                    .channel_hashes
                    .iter()
                    .map(|h| h.to_bytes_prost())
                    .collect(),
                hash: consume.hash.to_bytes_prost(),
                persistent: consume.persistent,
            }),
        },
    }
}

pub fn to_rspace_event(event: &Event) -> RspaceEvent {
    match event {
        Event::Produce(produce_event) => RspaceEvent::IoEvent(IOEvent::Produce(Produce {
            channel_hash: Blake2b256Hash::from_bytes_prost(&produce_event.channels_hash),
            hash: Blake2b256Hash::from_bytes_prost(&produce_event.hash),
            persistent: produce_event.persistent,
            is_deterministic: produce_event.is_deterministic,
            output_value: produce_event
                .output_value
                .clone()
                .into_iter()
                .map(|v| v.into())
                .collect(),
            failed: produce_event.failed,
        })),

        Event::Consume(consume_event) => RspaceEvent::IoEvent(IOEvent::Consume(Consume {
            channel_hashes: consume_event
                .channels_hashes
                .iter()
                .map(|h| Blake2b256Hash::from_bytes_prost(h))
                .collect(),
            hash: Blake2b256Hash::from_bytes_prost(&consume_event.hash),
            persistent: consume_event.persistent,
        })),

        Event::Comm(comm_event) => {
            let rspace_consume = Consume {
                channel_hashes: comm_event
                    .consume
                    .channels_hashes
                    .iter()
                    .map(|h| Blake2b256Hash::from_bytes_prost(h))
                    .collect(),
                hash: Blake2b256Hash::from_bytes_prost(&comm_event.consume.hash),
                persistent: comm_event.consume.persistent,
            };

            let mut produces = Vec::new();
            let mut times_repeated = std::collections::BTreeMap::new();

            for produce in &comm_event.produces {
                let rspace_produce = Produce {
                    channel_hash: Blake2b256Hash::from_bytes_prost(&produce.channels_hash),
                    hash: Blake2b256Hash::from_bytes_prost(&produce.hash),
                    persistent: produce.persistent,
                    is_deterministic: produce.is_deterministic,
                    output_value: produce
                        .output_value
                        .clone()
                        .into_iter()
                        .map(|v| v.into())
                        .collect(),
                    failed: produce.failed,
                };
                times_repeated.insert(rspace_produce.clone(), produce.times_repeated);
                produces.push(rspace_produce);
            }

            produces.sort_by(|a, b| {
                a.channel_hash
                    .cmp(&b.channel_hash)
                    .then_with(|| a.hash.cmp(&b.hash))
                    .then_with(|| a.persistent.cmp(&b.persistent))
            });

            let peeks: std::collections::BTreeSet<_> =
                comm_event.peeks.iter().map(|p| p.channel_index).collect();

            RspaceEvent::Comm(RspaceComm {
                consume: rspace_consume,
                produces,
                peeks,
                times_repeated,
            })
        }
    }
}
