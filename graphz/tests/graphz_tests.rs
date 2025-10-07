use graphz::rust::graphz::{
    apply, default_shape, subgraph, GraphRank, GraphRankDir, GraphShape, GraphType,
    StringSerializer,
};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn simple_graph() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "G".to_string(),
        GraphType::Graph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "graph \"G\" {\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn simple_digraph() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "G".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "digraph \"G\" {\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn simple_graph_with_comment() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "G".to_string(),
        GraphType::Graph,
        ser.clone(),
        false,
        Some("this is comment".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "// this is comment\ngraph \"G\" {\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn graph_two_nodes_one_edge() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "G".to_string(),
        GraphType::Graph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();
    g.edge("Hello", "World", None, None, None).await.unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "graph \"G\" {\n  \"Hello\" -- \"World\"\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn digraph_two_nodes_one_edge() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "G".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();
    g.edge("Hello", "World", None, None, None).await.unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "digraph \"G\" {\n  \"Hello\" -> \"World\"\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn digraph_nodes_with_style() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "G".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();
    g.node("Hello", GraphShape::Box, None, None, None)
        .await
        .unwrap();
    g.node("World", GraphShape::DoubleCircle, None, None, None)
        .await
        .unwrap();
    g.edge("Hello", "World", None, None, None).await.unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "digraph \"G\" {\n  \"Hello\" [shape=box]\n  \"World\" [shape=doublecircle]\n  \"Hello\" -> \"World\"\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn digraph_with_simple_subgraphs() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "Process".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();

    g.node("0", default_shape(), None, None, None)
        .await
        .unwrap();

    // Process 1 subgraph
    let g1 = subgraph(
        "".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        None,
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();
    g1.node("A", default_shape(), None, None, None)
        .await
        .unwrap();
    g1.node("B", default_shape(), None, None, None)
        .await
        .unwrap();
    g1.node("C", default_shape(), None, None, None)
        .await
        .unwrap();
    g1.edge("A", "B", None, None, None).await.unwrap();
    g1.edge("B", "C", None, None, None).await.unwrap();
    g1.close().await.unwrap();

    g.edge("0", "A", None, None, None).await.unwrap();

    // Process 2 subgraph
    let g2 = subgraph(
        "".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        None,
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();
    g2.node("K", default_shape(), None, None, None)
        .await
        .unwrap();
    g2.node("L", default_shape(), None, None, None)
        .await
        .unwrap();
    g2.node("M", default_shape(), None, None, None)
        .await
        .unwrap();
    g2.edge("K", "L", None, None, None).await.unwrap();
    g2.edge("L", "M", None, None, None).await.unwrap();
    g2.close().await.unwrap();

    g.edge("0", "K", None, None, None).await.unwrap();
    g.node("1", default_shape(), None, None, None)
        .await
        .unwrap();
    g.edge("M", "1", None, None, None).await.unwrap();
    g.edge("C", "1", None, None, None).await.unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "digraph \"Process\" {\n  \"0\"\n  subgraph {\n    \"A\"\n    \"B\"\n    \"C\"\n    \"A\" -> \"B\"\n    \"B\" -> \"C\"\n  }\n  \"0\" -> \"A\"\n  subgraph {\n    \"K\"\n    \"L\"\n    \"M\"\n    \"K\" -> \"L\"\n    \"L\" -> \"M\"\n  }\n  \"0\" -> \"K\"\n  \"1\"\n  \"M\" -> \"1\"\n  \"C\" -> \"1\"\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn digraph_with_fancy_subgraphs() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "Process".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();

    g.node("0", default_shape(), None, None, None)
        .await
        .unwrap();

    // Process 1 subgraph
    let g1 = subgraph(
        "cluster_p1".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        Some("process #1".to_string()),
        None,
        None,
        None,
        Some("blue".to_string()),
    )
    .await
    .unwrap();
    g1.node("A", default_shape(), None, None, None)
        .await
        .unwrap();
    g1.node("B", default_shape(), None, None, None)
        .await
        .unwrap();
    g1.node("C", default_shape(), None, None, None)
        .await
        .unwrap();
    g1.edge("A", "B", None, None, None).await.unwrap();
    g1.edge("B", "C", None, None, None).await.unwrap();
    g1.close().await.unwrap();

    g.edge("0", "A", None, None, None).await.unwrap();

    // Process 2 subgraph
    let g2 = subgraph(
        "cluster_p2".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        Some("process #2".to_string()),
        None,
        None,
        None,
        Some("green".to_string()),
    )
    .await
    .unwrap();
    g2.node("K", default_shape(), None, None, None)
        .await
        .unwrap();
    g2.node("L", default_shape(), None, None, None)
        .await
        .unwrap();
    g2.node("M", default_shape(), None, None, None)
        .await
        .unwrap();
    g2.edge("K", "L", None, None, None).await.unwrap();
    g2.edge("L", "M", None, None, None).await.unwrap();
    g2.close().await.unwrap();

    g.edge("0", "K", None, None, None).await.unwrap();
    g.node("1", default_shape(), None, None, None)
        .await
        .unwrap();
    g.edge("M", "1", None, None, None).await.unwrap();
    g.edge("C", "1", None, None, None).await.unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "digraph \"Process\" {\n  \"0\"\n  subgraph \"cluster_p1\" {\n    label = \"process #1\"\n    color=blue\n    \"A\"\n    \"B\"\n    \"C\"\n    \"A\" -> \"B\"\n    \"B\" -> \"C\"\n  }\n  \"0\" -> \"A\"\n  subgraph \"cluster_p2\" {\n    label = \"process #2\"\n    color=green\n    \"K\"\n    \"L\"\n    \"M\"\n    \"K\" -> \"L\"\n    \"L\" -> \"M\"\n  }\n  \"0\" -> \"K\"\n  \"1\"\n  \"M\" -> \"1\"\n  \"C\" -> \"1\"\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn blockchain_simple() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "Blockchain".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        Some(GraphRankDir::BT),
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();

    // Level 1 subgraph
    let lvl1 = subgraph(
        "".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        None,
        Some(GraphRank::Same),
        None,
        None,
        None,
    )
    .await
    .unwrap();
    lvl1.node("1", default_shape(), None, None, None)
        .await
        .unwrap();
    lvl1.node("ddeecc", GraphShape::Box, None, None, None)
        .await
        .unwrap();
    lvl1.node("ffeeff", GraphShape::Box, None, None, None)
        .await
        .unwrap();
    lvl1.close().await.unwrap();

    // Edges from level 0 to level 1
    g.edge_tuple(("000000".to_string(), "ffeeff".to_string()))
        .await
        .unwrap();
    g.edge_tuple(("000000".to_string(), "ddeecc".to_string()))
        .await
        .unwrap();

    // Level 0 subgraph
    let lvl0 = subgraph(
        "".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        None,
        Some(GraphRank::Same),
        None,
        None,
        None,
    )
    .await
    .unwrap();
    lvl0.node("0", default_shape(), None, None, None)
        .await
        .unwrap();
    lvl0.node("000000", GraphShape::Box, None, None, None)
        .await
        .unwrap();
    lvl0.close().await.unwrap();

    // Timeline subgraph
    let timeline = subgraph(
        "timeline".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        None,
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();
    timeline
        .node("3", GraphShape::PlainText, None, None, None)
        .await
        .unwrap();
    timeline
        .node("2", GraphShape::PlainText, None, None, None)
        .await
        .unwrap();
    timeline
        .node("1", GraphShape::PlainText, None, None, None)
        .await
        .unwrap();
    timeline
        .node("0", GraphShape::PlainText, None, None, None)
        .await
        .unwrap();
    timeline
        .edge_tuple(("0".to_string(), "1".to_string()))
        .await
        .unwrap();
    timeline
        .edge_tuple(("1".to_string(), "2".to_string()))
        .await
        .unwrap();
    timeline
        .edge_tuple(("2".to_string(), "3".to_string()))
        .await
        .unwrap();
    timeline.close().await.unwrap();

    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "digraph \"Blockchain\" {\n  rankdir=BT\n  subgraph {\n    rank=same\n    \"1\"\n    \"ddeecc\" [shape=box]\n    \"ffeeff\" [shape=box]\n  }\n  \"000000\" -> \"ffeeff\"\n  \"000000\" -> \"ddeecc\"\n  subgraph {\n    rank=same\n    \"0\"\n    \"000000\" [shape=box]\n  }\n  subgraph \"timeline\" {\n    \"3\" [shape=plaintext]\n    \"2\" [shape=plaintext]\n    \"1\" [shape=plaintext]\n    \"0\" [shape=plaintext]\n    \"0\" -> \"1\"\n    \"1\" -> \"2\"\n    \"2\" -> \"3\"\n  }\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn process_example() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "G".to_string(),
        GraphType::Graph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();

    // Add all edges
    g.edge("run", "intr", None, None, None).await.unwrap();
    g.edge("intr", "runbl", None, None, None).await.unwrap();
    g.edge("runbl", "run", None, None, None).await.unwrap();
    g.edge("run", "kernel", None, None, None).await.unwrap();
    g.edge("kernel", "zombie", None, None, None).await.unwrap();
    g.edge("kernel", "sleep", None, None, None).await.unwrap();
    g.edge("kernel", "runmem", None, None, None).await.unwrap();
    g.edge("sleep", "swap", None, None, None).await.unwrap();
    g.edge("swap", "runswap", None, None, None).await.unwrap();
    g.edge("runswap", "new", None, None, None).await.unwrap();
    g.edge("runswap", "runmem", None, None, None).await.unwrap();
    g.edge("new", "runmem", None, None, None).await.unwrap();
    g.edge("sleep", "runmem", None, None, None).await.unwrap();
    g.close().await.unwrap();

    let result = ser.get_content().await;
    let expected = "graph \"G\" {\n  \"run\" -- \"intr\"\n  \"intr\" -- \"runbl\"\n  \"runbl\" -- \"run\"\n  \"run\" -- \"kernel\"\n  \"kernel\" -- \"zombie\"\n  \"kernel\" -- \"sleep\"\n  \"kernel\" -- \"runmem\"\n  \"sleep\" -- \"swap\"\n  \"swap\" -- \"runswap\"\n  \"runswap\" -- \"new\"\n  \"runswap\" -- \"runmem\"\n  \"new\" -- \"runmem\"\n  \"sleep\" -- \"runmem\"\n}";
    assert_eq!(result, expected);
}

#[tokio::test]
async fn huge_graph() {
    let ser = Arc::new(StringSerializer::new());
    let g = apply(
        "G".to_string(),
        GraphType::DiGraph,
        ser.clone(),
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        HashMap::new(),
    )
    .await
    .unwrap();

    // Add 1000 edges
    for i in 1..=1000 {
        g.edge_tuple((format!("e{}", i), format!("e{}", i + 1)))
            .await
            .unwrap();
    }

    g.close().await.unwrap();

    // Just check that we didn't crash - don't check the exact content
    let result = ser.get_content().await;
    assert!(result.starts_with("digraph \"G\" {"));
    assert!(result.ends_with("}"));
}
