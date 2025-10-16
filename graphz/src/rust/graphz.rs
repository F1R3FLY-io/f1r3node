use async_trait::async_trait;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

#[async_trait]
pub trait GraphSerializer: Send + Sync {
    async fn push(&self, str: &str, suffix: &str) -> Result<(), GraphzError>;
}

pub struct StringSerializer {
    buffer: Arc<Mutex<String>>,
}

impl StringSerializer {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(String::new())),
        }
    }

    pub async fn get_content(&self) -> String {
        self.buffer.lock().await.clone()
    }
}

impl Default for StringSerializer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GraphSerializer for StringSerializer {
    async fn push(&self, str: &str, suffix: &str) -> Result<(), GraphzError> {
        let mut buffer = self.buffer.lock().await;
        buffer.push_str(str);
        buffer.push_str(suffix);
        Ok(())
    }
}

pub struct ListSerializer {
    lines: Arc<std::sync::Mutex<Vec<String>>>,
    sender: Option<tokio::sync::oneshot::Sender<Vec<String>>>,
}

impl ListSerializer {
    pub fn new(sender: tokio::sync::oneshot::Sender<Vec<String>>) -> Self {
        Self {
            lines: Arc::new(std::sync::Mutex::new(Vec::new())),
            sender: Some(sender),
        }
    }

    pub async fn get_lines(&self) -> Vec<String> {
        self.lines.lock().expect("Failed to lock mutex").clone() // it only could fail if the thread holding the mutex is panicked,
    }
}

impl Drop for ListSerializer {
    fn drop(&mut self) {
        let lines = self.lines.lock().expect("Failed to lock mutex");
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(lines.clone());
        }
    }
}

#[async_trait]
impl GraphSerializer for ListSerializer {
    async fn push(&self, str: &str, suffix: &str) -> Result<(), GraphzError> {
        let mut lines = self.lines.lock().expect("Failed to lock mutex");
        lines.push(format!("{}{}", str, suffix));
        Ok(())
    }
}

pub struct FileSerializer {
    file: Arc<Mutex<File>>,
}

impl FileSerializer {
    pub async fn new(file: File) -> Self {
        Self {
            file: Arc::new(Mutex::new(file)),
        }
    }
}

#[async_trait]
impl GraphSerializer for FileSerializer {
    async fn push(&self, str: &str, suffix: &str) -> Result<(), GraphzError> {
        let mut file = self.file.lock().await;
        let content = format!("{}{}", str, suffix);
        file.write_all(content.as_bytes())
            .await
            .map_err(|e| GraphzError::IoError(e.to_string()))?;
        file.flush()
            .await
            .map_err(|e| GraphzError::IoError(e.to_string()))?;
        Ok(())
    }
}

// Sealed traits equivalent - all enum definitions first (like Scala sealed traits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    Graph,
    DiGraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphShape {
    Circle,
    DoubleCircle,
    Box,
    PlainText,
    Msquare,
    Record,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphRank {
    Same,
    Min,
    Source,
    Max,
    Sink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphRankDir {
    TB,
    BT,
    LR,
    RL,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphStyle {
    Solid,
    Bold,
    Filled,
    Invis,
    Dotted,
    Dashed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphArrowType {
    NormalArrow,
    InvArrow,
    NoneArrow,
}

// Display implementations (equivalent to Scala's Show instances in object Graphz)

// Note: We don't use a small_to_string function like Scala's smallToString[A]
// because:
// 1. It would depend on Debug implementation which can change
// 2. It would require runtime string allocation + processing
// 3. Explicit match provides compile-time safety and zero-cost abstractions
// 4. Rust doesn't have Scala's type-level abstractions for Show[A]

impl Display for GraphShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            GraphShape::Circle => "circle",
            GraphShape::DoubleCircle => "doublecircle",
            GraphShape::Box => "box",
            GraphShape::PlainText => "plaintext",
            GraphShape::Msquare => "Msquare",
            GraphShape::Record => "record",
        };
        write!(f, "{}", s)
    }
}

impl Display for GraphStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            GraphStyle::Solid => "solid",
            GraphStyle::Bold => "bold",
            GraphStyle::Filled => "filled",
            GraphStyle::Invis => "invis",
            GraphStyle::Dotted => "dotted",
            GraphStyle::Dashed => "dashed",
        };
        write!(f, "{}", s)
    }
}

impl Display for GraphRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            GraphRank::Same => "same",
            GraphRank::Min => "min",
            GraphRank::Source => "source",
            GraphRank::Max => "max",
            GraphRank::Sink => "sink",
        };
        write!(f, "{}", s)
    }
}

impl Display for GraphRankDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Display for GraphArrowType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            GraphArrowType::NormalArrow => "normal",
            GraphArrowType::InvArrow => "inv",
            GraphArrowType::NoneArrow => "none",
        };
        write!(f, "{}", s)
    }
}

// Public functions corresponding to Scala's object Graphz
const TAB: &str = "  ";

pub fn default_shape() -> GraphShape {
    GraphShape::Circle
}

pub async fn apply(
    name: String,
    gtype: GraphType,
    ser: Arc<dyn GraphSerializer>,
    subgraph: bool,
    comment: Option<String>,
    label: Option<String>,
    splines: Option<String>,
    rank: Option<GraphRank>,
    rankdir: Option<GraphRankDir>,
    style: Option<String>,
    color: Option<String>,
    node: HashMap<String, String>,
) -> Result<Graphz, GraphzError> {
    let insert = |opt_str: Option<String>, transform: fn(&str) -> String| {
        let ser = ser.clone();
        async move {
            let indent = if subgraph {
                format!("{}{}", TAB, TAB)
            } else {
                TAB.to_string()
            };

            if let Some(s) = opt_str {
                ser.push(&format!("{}{}", indent, transform(&s)), "\n")
                    .await?;
            }
            Ok::<(), GraphzError>(())
        }
    };

    if let Some(c) = &comment {
        ser.push(&format!("// {}", c), "\n").await?;
    }

    let t = if subgraph {
        format!("{}{}", TAB, TAB)
    } else {
        TAB.to_string()
    };

    let header = head(gtype, subgraph, &name);
    ser.push(&header, "\n").await?;

    insert(label, |l| format!("label = {}", quote(l))).await?;
    insert(style, |s| format!("style={}", s)).await?;
    insert(color, |s| format!("color={}", s)).await?;
    insert(rank.map(|r| r.to_string()), |r| format!("rank={}", r)).await?;
    insert(rankdir.map(|r| r.to_string()), |r| format!("rankdir={}", r)).await?;
    insert(attr_mk_str(&node), |n| format!("node {}", n)).await?;
    insert(splines, |s| format!("splines={}", s)).await?;

    Ok(Graphz {
        graph_type: gtype,
        tab: t,
        serializer: ser,
    })
}

pub async fn subgraph(
    name: String,
    gtype: GraphType,
    ser: Arc<dyn GraphSerializer>,
    label: Option<String>,
    rank: Option<GraphRank>,
    rankdir: Option<GraphRankDir>,
    style: Option<String>,
    color: Option<String>,
) -> Result<Graphz, GraphzError> {
    apply(
        name,
        gtype,
        ser,
        true,
        None,
        label,
        None,
        rank,
        rankdir,
        style,
        color,
        HashMap::new(),
    )
    .await
}

fn head(graph_type: GraphType, subgraph: bool, name: &str) -> String {
    let prefix = match (graph_type, subgraph) {
        (_, true) => format!("{}subgraph", TAB),
        (GraphType::Graph, _) => "graph".to_string(),
        (GraphType::DiGraph, _) => "digraph".to_string(),
    };

    if name.is_empty() {
        format!("{} {{", prefix)
    } else {
        format!("{} \"{}\" {{", prefix, name)
    }
}

fn quote(s: &str) -> String {
    if s.starts_with('"') {
        s.to_string()
    } else {
        format!("\"{}\"", s)
    }
}

fn attr_mk_str(attrs: &HashMap<String, String>) -> Option<String> {
    if attrs.is_empty() {
        None
    } else {
        let attr_pairs: Vec<String> = attrs.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        Some(format!("[{}]", attr_pairs.join(" ")))
    }
}

pub struct Graphz {
    graph_type: GraphType,
    tab: String,
    serializer: Arc<dyn GraphSerializer>,
}

impl Graphz {
    pub async fn edge_tuple(&self, edg: (String, String)) -> Result<(), GraphzError> {
        self.edge(&edg.0, &edg.1, None, None, None).await
    }

    pub async fn edge(
        &self,
        src: &str,
        dst: &str,
        style: Option<GraphStyle>,
        arrow_head: Option<GraphArrowType>,
        constraint: Option<bool>,
    ) -> Result<(), GraphzError> {
        let mut attrs = HashMap::new();
        if let Some(s) = style {
            attrs.insert("style".to_string(), s.to_string());
        }

        if let Some(ah) = arrow_head {
            attrs.insert("arrowhead".to_string(), ah.to_string());
        }

        if let Some(c) = constraint {
            attrs.insert("constraint".to_string(), c.to_string());
        }

        let quoted_src = quote(src);
        let quoted_dst = quote(dst);
        let attr_str = attr_mk_str(&attrs)
            .map(|a| format!(" {}", a))
            .unwrap_or_default();

        let formatted_edge = self.edge_mk_str(&quoted_src, &quoted_dst, &attr_str);
        self.serializer.push(&formatted_edge, "\n").await
    }

    pub async fn node(
        &self,
        name: &str,
        shape: GraphShape,
        style: Option<GraphStyle>,
        color: Option<String>,
        label: Option<String>,
    ) -> Result<(), GraphzError> {
        let mut attrs = HashMap::new();

        if shape != default_shape() {
            attrs.insert("shape".to_string(), shape.to_string());
        }

        if let Some(s) = style {
            attrs.insert("style".to_string(), s.to_string());
        }

        if let Some(c) = color {
            attrs.insert("color".to_string(), c);
        }

        if let Some(l) = label {
            attrs.insert("label".to_string(), l);
        }

        let quoted_name = quote(name);
        let attr_str = attr_mk_str(&attrs)
            .map(|a| format!(" {}", a))
            .unwrap_or_default();

        let node_str = format!("{}{}{}", self.tab, quoted_name, attr_str);
        self.serializer.push(&node_str, "\n").await
    }

    pub async fn close(&self) -> Result<(), GraphzError> {
        let content = &self.tab[TAB.len()..];

        // Equivalent to Scala: val suffix = if (content.isEmpty) "" else "\n"
        let suffix = if content.is_empty() { "" } else { "\n" };

        self.serializer
            .push(&format!("{}}}", content), suffix)
            .await
    }

    fn edge_mk_str(&self, src: &str, dst: &str, attr_str: &str) -> String {
        match self.graph_type {
            GraphType::Graph => format!("{}{} -- {}{}", self.tab, src, dst, attr_str),
            GraphType::DiGraph => format!("{}{} -> {}{}", self.tab, src, dst, attr_str),
        }
    }
}

// Error type for graphz operations
#[derive(Debug, Clone)]
pub enum GraphzError {
    IoError(String),
    SerializationError(String),
}

impl Display for GraphzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphzError::IoError(msg) => write!(f, "IO Error: {}", msg),
            GraphzError::SerializationError(msg) => write!(f, "Serialization Error: {}", msg),
        }
    }
}

impl std::error::Error for GraphzError {}
