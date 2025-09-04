use casper::rust::api::machine_verifiable_dag::{MachineVerifiableDag, VerifiableEdge};
use casper::rust::TopoSort;
use crypto::rust::hash::blake2b256::Blake2b256;
use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;

#[test]
fn should_create_dag_for_simple_two_blocks_with_one_merge_block() {
    // given
    let genesis: BlockHash = Blake2b256::hash("genesis".as_bytes().to_vec()).into();
    let block1: BlockHash = Blake2b256::hash("block1".as_bytes().to_vec()).into();
    let block2: BlockHash = Blake2b256::hash("block2".as_bytes().to_vec()).into();
    let block3: BlockHash = Blake2b256::hash("block3".as_bytes().to_vec()).into();

    let toposort: TopoSort = vec![vec![block1.clone(), block2.clone()], vec![block3.clone()]];

    let fetch = |b: BlockHash| -> Vec<BlockHash> {
        match b {
            ref block_hash if *block_hash == block1 => vec![genesis.clone()],
            ref block_hash if *block_hash == block2 => vec![genesis.clone()],
            ref block_hash if *block_hash == block3 => vec![block1.clone(), block2.clone()],
            _ => vec![],
        }
    };

    // when
    let result: Vec<VerifiableEdge> = MachineVerifiableDag::apply(toposort, fetch);

    // then
    // Note: We use PrettyPrinter::build_string_bytes instead of .show() as Scala does
    // because BlockHash (prost::bytes::Bytes) doesn't implement Display trait in Rust
    assert_eq!(
        result[0],
        VerifiableEdge::new(
            PrettyPrinter::build_string_bytes(&block3),
            PrettyPrinter::build_string_bytes(&block1)
        )
    );
    assert_eq!(
        result[1],
        VerifiableEdge::new(
            PrettyPrinter::build_string_bytes(&block3),
            PrettyPrinter::build_string_bytes(&block2)
        )
    );
    assert_eq!(
        result[2],
        VerifiableEdge::new(
            PrettyPrinter::build_string_bytes(&block1),
            PrettyPrinter::build_string_bytes(&genesis)
        )
    );
    assert_eq!(
        result[3],
        VerifiableEdge::new(
            PrettyPrinter::build_string_bytes(&block2),
            PrettyPrinter::build_string_bytes(&genesis)
        )
    );
}
