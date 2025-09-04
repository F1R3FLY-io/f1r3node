use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use graphz::rust::graphz::{
    apply, subgraph, GraphArrowType, GraphRankDir, GraphSerializer, GraphShape, GraphStyle,
    GraphType, Graphz, GraphzError,
};
use itertools::Itertools;
use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ValidatorBlock {
    pub block_hash: String,
    pub parents: Vec<String>,
    pub justifications: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GraphConfig {
    pub show_justification_lines: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            show_justification_lines: false,
        }
    }
}

pub type ValidatorsBlocks = HashMap<i64, Vec<ValidatorBlock>>;

#[derive(Debug, Clone)]
pub struct DagInfo {
    pub validators: HashMap<String, ValidatorsBlocks>,
    pub timeseries: Vec<i64>,
}

impl DagInfo {
    pub fn empty() -> Self {
        Self {
            validators: HashMap::new(),
            timeseries: Vec::new(),
        }
    }
}

pub struct GraphzGenerator;

impl GraphzGenerator {
    pub async fn dag_as_cluster(
        topo_sort: Vec<Vec<BlockHash>>,
        last_finalized_block_hash: String,
        config: GraphConfig,
        ser: Arc<dyn GraphSerializer>,
        block_store: &mut KeyValueBlockStore,
    ) -> Result<Graphz, GraphGeneratorError> {
        let mut acc = DagInfo::empty();
        for block_hashes in topo_sort {
            acc = Self::accumulate_dag_info(acc, block_hashes, block_store).await?;
        }

        let mut timeseries = acc.timeseries;
        timeseries.reverse();

        let first_ts = timeseries.first().copied().unwrap_or(0);
        let validators = acc.validators.clone();
        let validators_list: Vec<(String, ValidatorsBlocks)> = validators
            .into_iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .collect();

        let g = Self::init_graph("dag", ser.clone()).await?;

        let all_ancestors: Vec<String> = validators_list
            .iter()
            .flat_map(|(_, blocks)| {
                blocks
                    .get(&first_ts)
                    .map(|validator_blocks| {
                        validator_blocks
                            .iter()
                            .flat_map(|b| b.parents.iter().cloned())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            .unique()
            .sorted()
            .collect();

        // draw ancestors first
        for ancestor in &all_ancestors {
            let style = Self::style_for(ancestor, &last_finalized_block_hash);
            g.node(ancestor, GraphShape::Box, style, None, None).await?;
        }

        // create invisible edges from ancestors to first node in each cluster for proper alignment
        let mut invisible_edge_pairs = Vec::new();
        for (id, blocks) in &validators_list {
            let nodes = Self::nodes_for_ts(id, first_ts, blocks, &last_finalized_block_hash);
            let node_names: Vec<String> = nodes.keys().cloned().collect();
            for ancestor in &all_ancestors {
                for node_name in &node_names {
                    invisible_edge_pairs.push((ancestor.clone(), node_name.clone()));
                }
            }
        }

        // Create edges sequentially
        for (ancestor, node_name) in invisible_edge_pairs {
            g.edge(&ancestor, &node_name, Some(GraphStyle::Invis), None, None)
                .await?;
        }

        // Draw clusters per validator
        for (id, blocks) in &validators_list {
            Self::validator_cluster(
                id,
                blocks,
                &timeseries,
                &last_finalized_block_hash,
                ser.clone(),
            )
            .await?;
        }

        // Draw parent dependencies
        let validator_blocks_list: Vec<ValidatorsBlocks> = validators_list
            .into_iter()
            .map(|(_, blocks)| blocks)
            .collect();
        Self::draw_parent_dependencies(&g, &validator_blocks_list).await?;

        // Draw justification dotted lines
        if config.show_justification_lines {
            Self::draw_justification_dotted_lines(&g, &acc.validators).await?;
        }

        g.close().await?;

        Ok(g)
    }

    pub async fn accumulate_dag_info(
        mut acc: DagInfo,
        block_hashes: Vec<BlockHash>,
        block_store: &mut KeyValueBlockStore,
    ) -> Result<DagInfo, GraphGeneratorError> {
        let mut blocks = Vec::new();
        for block_hash in &block_hashes {
            let block = block_store.get_unsafe(block_hash);
            blocks.push(block);
        }

        let time_entry = blocks.first().unwrap().body.state.block_number;

        let validators: Vec<HashMap<String, ValidatorsBlocks>> = blocks
            .into_iter()
            .map(|block| {
                let block_hash = PrettyPrinter::build_string_bytes(&block.block_hash);
                let block_sender_hash = PrettyPrinter::build_string_bytes(&block.sender);

                let parents: Vec<String> = block
                    .header
                    .parents_hash_list
                    .iter()
                    .map(|hash| PrettyPrinter::build_string_bytes(hash))
                    .collect();

                let justifications: Vec<String> = block
                    .justifications
                    .iter()
                    .map(|j| PrettyPrinter::build_string_bytes(&j.latest_block_hash))
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();

                let mut validator_blocks = HashMap::new();
                validator_blocks.insert(
                    time_entry,
                    vec![ValidatorBlock {
                        block_hash,
                        parents,
                        justifications,
                    }],
                );

                let mut block_map = HashMap::new();
                block_map.insert(block_sender_hash, validator_blocks);
                block_map
            })
            .collect();

        acc.timeseries.insert(0, time_entry);

        // Equivalent to acc.validators |+| Foldable[List].fold(validators)
        for (block_sender_hash, blocks_map) in validators.into_iter().flat_map(|m| m.into_iter()) {
            let acc_validator = acc
                .validators
                .entry(block_sender_hash)
                .or_insert_with(HashMap::new);
            for (ts, blocks) in blocks_map {
                acc_validator
                    .entry(ts)
                    .or_insert_with(Vec::new)
                    .extend(blocks);
            }
        }

        Ok(acc)
    }

    pub async fn init_graph(
        name: &str,
        ser: Arc<dyn GraphSerializer>,
    ) -> Result<Graphz, GraphGeneratorError> {
        let mut node_attrs = HashMap::new();
        node_attrs.insert("width".to_string(), "0".to_string());
        node_attrs.insert("height".to_string(), "0".to_string());
        node_attrs.insert("margin".to_string(), "0.03".to_string());
        node_attrs.insert("fontsize".to_string(), "8".to_string());

        let graph = apply(
            name.to_string(),
            GraphType::DiGraph,
            ser,
            false,
            None,
            None,
            Some("false".to_string()),
            None,
            Some(GraphRankDir::BT),
            None,
            None,
            node_attrs,
        )
        .await?;

        Ok(graph)
    }

    pub async fn draw_parent_dependencies(
        g: &Graphz,
        validators: &[ValidatorsBlocks],
    ) -> Result<(), GraphGeneratorError> {
        // Equivalent to validators.flatMap(_.values.toList.flatten)
        let all_validator_blocks: Vec<&ValidatorBlock> = validators
            .iter()
            .flat_map(|validator_blocks| validator_blocks.values())
            .flatten()
            .collect();

        // Equivalent to case ValidatorBlock(blockHash, parentsHashes, _) => parentsHashes.traverse(...)
        let edge_pairs: Vec<(&String, &String)> = all_validator_blocks
            .iter()
            .flat_map(|validator_block| {
                validator_block
                    .parents
                    .iter()
                    .map(move |parent_hash| (&validator_block.block_hash, parent_hash))
            })
            .collect();

        // Create edges (async operations must be done sequentially)
        for (block_hash, parent_hash) in edge_pairs {
            g.edge(block_hash, parent_hash, None, None, Some(false))
                .await?;
        }

        Ok(())
    }

    pub async fn draw_justification_dotted_lines(
        g: &Graphz,
        validators: &HashMap<String, ValidatorsBlocks>,
    ) -> Result<(), GraphGeneratorError> {
        // Equivalent to validators.values.toList.flatMap(_.values.toList.flatten)
        let all_validator_blocks: Vec<&ValidatorBlock> = validators
            .values()
            .flat_map(|validator_blocks| validator_blocks.values())
            .flatten()
            .collect();

        // Equivalent to case ValidatorBlock(blockHash, _, justifications) => justifications.traverse(...)
        let edge_pairs: Vec<(&String, &String)> = all_validator_blocks
            .iter()
            .flat_map(|validator_block| {
                validator_block
                    .justifications
                    .iter()
                    .map(move |justification| (&validator_block.block_hash, justification))
            })
            .collect();

        // Create edges (async operations must be done sequentially)
        for (block_hash, justification) in edge_pairs {
            g.edge(
                block_hash,
                justification,
                Some(GraphStyle::Dotted),
                Some(GraphArrowType::NoneArrow),
                Some(false),
            )
            .await?;
        }

        Ok(())
    }

    pub fn nodes_for_ts(
        validator_id: &str,
        ts: i64,
        blocks: &ValidatorsBlocks,
        last_finalized_block_hash: &str,
    ) -> HashMap<String, Option<GraphStyle>> {
        match blocks.get(&ts) {
            Some(ts_blocks) => ts_blocks
                .iter()
                .map(|validator_block| {
                    let style =
                        Self::style_for(&validator_block.block_hash, last_finalized_block_hash);
                    (validator_block.block_hash.clone(), style)
                })
                .collect(),
            None => {
                let invisible_node = format!("{}_{}", ts, validator_id);
                let mut map = HashMap::new();
                map.insert(invisible_node, Some(GraphStyle::Invis));
                map
            }
        }
    }

    async fn validator_cluster(
        id: &str,
        blocks: &ValidatorsBlocks,
        timeseries: &[i64],
        last_finalized_block_hash: &str,
        ser: Arc<dyn GraphSerializer>,
    ) -> Result<Graphz, GraphGeneratorError> {
        let cluster_name = format!("cluster_{}", id);
        let g = subgraph(
            cluster_name,
            GraphType::DiGraph,
            ser,
            Some(id.to_string()),
            None,
            None,
            None,
            None,
        )
        .await?;

        // Equivalent to nodes = timeseries.map(ts => nodesForTs(id, ts, blocks, lastFinalizedBlockHash))
        let nodes: Vec<HashMap<String, Option<GraphStyle>>> = timeseries
            .iter()
            .map(|&ts| Self::nodes_for_ts(id, ts, blocks, last_finalized_block_hash))
            .collect();

        // Equivalent to nodes.traverse(ns => ns.toList.traverse { case (name, style) => g.node(...) })
        for node_map in &nodes {
            for (name, style) in node_map {
                g.node(name, GraphShape::Box, *style, None, None).await?;
            }
        }

        // Equivalent to nodes.zip(nodes.drop(1)).traverse { case (n1s, n2s) => ... }
        let edge_pairs: Vec<(&String, &String)> = nodes
            .windows(2)
            .flat_map(|window| {
                if let [n1s, n2s] = window {
                    n1s.keys()
                        .flat_map(move |n1| n2s.keys().map(move |n2| (n1, n2)))
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                }
            })
            .collect();

        // Create invisible edges between consecutive timestamp nodes
        for (n1, n2) in edge_pairs {
            g.edge(n1, n2, Some(GraphStyle::Invis), None, None).await?;
        }

        g.close().await?;
        Ok(g)
    }

    pub fn style_for(block_hash: &str, last_finalized_block_hash: &str) -> Option<GraphStyle> {
        if block_hash == last_finalized_block_hash {
            Some(GraphStyle::Filled)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub enum GraphGeneratorError {
    GraphError(String),
    BlockStoreError(String),
    GraphzError(GraphzError),
}

impl std::fmt::Display for GraphGeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphGeneratorError::GraphError(msg) => write!(f, "Graph Error: {}", msg),
            GraphGeneratorError::BlockStoreError(msg) => write!(f, "Block Store Error: {}", msg),
            GraphGeneratorError::GraphzError(err) => write!(f, "Graphz Error: {}", err),
        }
    }
}

impl std::error::Error for GraphGeneratorError {}

impl From<GraphzError> for GraphGeneratorError {
    fn from(err: GraphzError) -> Self {
        GraphGeneratorError::GraphzError(err)
    }
}
