pub mod rust;

pub mod comm {
    include!(concat!(env!("OUT_DIR"), "/coop.rchain.comm.discovery.rs"));
}
