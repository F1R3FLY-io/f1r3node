//! Similarity Pattern Extraction Helpers
//!
//! This module provides functions for extracting similarity-related parameters
//! from Rholang Par values and EFunction modifiers. These functions are used
//! by GenericRSpace and the VectorDB subsystem to process similarity queries.
//!
//! # Extraction Functions
//!
//! ## Pattern Modifiers
//! - [`extract_modifiers_from_efunctions`]: Parse `sim()` and `rank()` EFunction calls
//! - [`ExtractedModifiers`]: Consolidated result of modifier extraction
//!
//! ## Par-based Extraction
//! - [`extract_embedding_from_par`]: Extract embedding vectors from Par values
//! - [`extract_number_from_par`]: Extract integer values from Par
//! - [`extract_threshold_from_par`]: Extract similarity thresholds
//! - [`extract_top_k_from_par`]: Extract top-K ranking limits
//! - [`extract_metric_from_par`]: Extract similarity metric specification
//! - [`extract_rank_function_from_par`]: Extract ranking function names
//!
//! ## VectorDB Data Extraction
//! - [`extract_embedding_from_map`]: Extract embeddings from map-structured data
//! - [`extract_channel_id_from_par`]: Extract GPrivate channel IDs
//!
//! # Similarity Computation
//!
//! - [`compute_cosine_similarity`]: Cosine similarity (normalized dot product)
//! - [`compute_dot_product`]: Raw dot product
//! - [`compute_euclidean_similarity`]: Euclidean distance-based similarity
//! - [`compute_manhattan_similarity`]: Manhattan (L1) distance-based similarity
//! - [`compute_hamming_similarity`]: Hamming distance for boolean vectors
//! - [`compute_jaccard_similarity`]: Jaccard similarity for set-like vectors

use models::rhoapi::{EFunction, Expr, Par, expr::ExprInstance, g_unforgeable::UnfInstance};
use models::rust::par_map_type_mapper::ParMapTypeMapper;

use super::collections::{EmbeddingType, SimilarityMetric};
use super::errors::SpaceError;

// =============================================================================
// Par-based Extraction Helpers
// =============================================================================

/// Extract an embedding vector from a Par.
///
/// The embedding is expected to be in one of these forms:
/// - GString with comma-delimited floats (e.g., `"0.8,0.2,0.5"`) - used as-is
/// - EList of GInt values (e.g., `[80, 20, 50]`) - scaled 0-100, converted to 0.0-1.0
/// - EList of expressions that evaluate to numbers
///
/// # Arguments
/// - `par`: The Par containing the embedding
///
/// # Returns
/// - `Ok(Vec<f32>)`: The extracted embedding vector
/// - `Err(SpaceError)`: If the Par doesn't contain a valid embedding
pub fn extract_embedding_from_par(par: &Option<Par>) -> Result<Vec<f32>, SpaceError> {
    let par = par.as_ref().ok_or_else(|| SpaceError::SimilarityMatchError {
        reason: "Similarity query embedding is missing".to_string(),
    })?;

    // Try float string format first: "0.8,0.2,0.5"
    for expr in &par.exprs {
        if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
            let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
            let mut embedding = Vec::with_capacity(parts.len());
            for part in parts {
                if let Ok(f) = part.parse::<f32>() {
                    embedding.push(f);
                } else {
                    return Err(SpaceError::SimilarityMatchError {
                        reason: format!("Invalid float in embedding string: '{}'", part),
                    });
                }
            }
            if !embedding.is_empty() {
                return Ok(embedding);
            }
        }
    }

    // Fall back to EList with integers: [80, 20, 50]
    for expr in &par.exprs {
        if let Some(ExprInstance::EListBody(list)) = &expr.expr_instance {
            let mut embedding = Vec::with_capacity(list.ps.len());
            for elem_par in &list.ps {
                // Each element should be a Par containing a GInt
                let value = extract_number_from_par(elem_par)?;
                // Scale from 0-100 to 0.0-1.0
                embedding.push(value as f32 / 100.0);
            }
            return Ok(embedding);
        }
    }

    Err(SpaceError::SimilarityMatchError {
        reason: "Similarity query embedding must be a list of numbers or comma-delimited float string".to_string(),
    })
}

/// Extract a single number from a Par.
///
/// # Arguments
/// - `par`: The Par containing the number
///
/// # Returns
/// - `Ok(i64)`: The extracted number
/// - `Err(SpaceError)`: If the Par doesn't contain a valid number
pub fn extract_number_from_par(par: &Par) -> Result<i64, SpaceError> {
    for expr in &par.exprs {
        if let Some(ExprInstance::GInt(n)) = &expr.expr_instance {
            return Ok(*n);
        }
    }

    Err(SpaceError::SimilarityMatchError {
        reason: "Expected a number in similarity pattern".to_string(),
    })
}

/// Extract a threshold from a Par.
///
/// The threshold can be specified as:
/// - GString with float (e.g., `"0.5"`) - used as-is (must be 0.0-1.0)
/// - GInt (e.g., `50`) - scaled from 0-100 to 0.0-1.0
///
/// # Arguments
/// - `par`: The Par containing the threshold
///
/// # Returns
/// - `Ok(f32)`: The extracted threshold (0.0 to 1.0)
/// - `Err(SpaceError)`: If the Par doesn't contain a valid threshold
pub fn extract_threshold_from_par(par: &Option<Par>) -> Result<f32, SpaceError> {
    let par = par.as_ref().ok_or_else(|| SpaceError::SimilarityMatchError {
        reason: "Explicit threshold value is missing".to_string(),
    })?;

    // Try float string format first: "0.5"
    for expr in &par.exprs {
        if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
            if let Ok(f) = s.parse::<f32>() {
                if f < 0.0 || f > 1.0 {
                    return Err(SpaceError::SimilarityMatchError {
                        reason: format!("Threshold string must be 0.0-1.0, got {}", f),
                    });
                }
                return Ok(f);
            }
        }
    }

    // Fall back to integer: 50 -> 0.5
    let value = extract_number_from_par(par)?;

    // Validate range and convert 0-100 to 0.0-1.0
    if value < 0 || value > 100 {
        return Err(SpaceError::SimilarityMatchError {
            reason: format!("Threshold must be 0-100, got {}", value),
        });
    }

    Ok(value as f32 / 100.0)
}

/// Extract a top-K value from a Par.
///
/// The K value must be a positive integer indicating how many results to return.
///
/// # Arguments
/// - `par`: The Par containing the K value
///
/// # Returns
/// - `Ok(usize)`: The extracted K value (must be >= 1)
/// - `Err(SpaceError)`: If the Par doesn't contain a valid K value
pub fn extract_top_k_from_par(par: &Option<Par>) -> Result<usize, SpaceError> {
    let par = par.as_ref().ok_or_else(|| SpaceError::SimilarityMatchError {
        reason: "Top-K value is missing".to_string(),
    })?;

    let value = extract_number_from_par(par)?;

    if value < 1 {
        return Err(SpaceError::SimilarityMatchError {
            reason: format!("Top-K value must be >= 1, got {}", value),
        });
    }

    Ok(value as usize)
}

/// Extract a similarity metric from a Par.
///
/// The metric can be specified as a string from the supported set:
/// - `"cos"` or `"cosine"` -> Cosine similarity
/// - `"dot"` or `"dotproduct"` -> Dot product
/// - `"euc"` or `"euclidean"` or `"l2"` -> Euclidean distance-based similarity
/// - `"manhattan"` or `"l1"` -> Manhattan distance-based similarity
/// - `"hamming"` -> Hamming distance for boolean vectors
/// - `"jaccard"` -> Jaccard similarity for boolean vectors
///
/// # Arguments
/// - `par`: The Par containing the metric string
///
/// # Returns
/// - `Ok(SimilarityMetric)`: The extracted metric
/// - `Err(SpaceError)`: If the Par doesn't contain a valid metric string
pub fn extract_metric_from_par(par: &Option<Par>) -> Result<SimilarityMetric, SpaceError> {
    let par = par.as_ref().ok_or_else(|| SpaceError::SimilarityMatchError {
        reason: "Similarity metric value is missing".to_string(),
    })?;

    // Extract string from Par
    for expr in &par.exprs {
        if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
            // Parse metric string (case-insensitive)
            return match s.to_lowercase().as_str() {
                "cos" | "cosine" => Ok(SimilarityMetric::Cosine),
                "dot" | "dotproduct" | "dot_product" => Ok(SimilarityMetric::DotProduct),
                "euc" | "euclidean" | "l2" => Ok(SimilarityMetric::Euclidean),
                "manhattan" | "l1" => Ok(SimilarityMetric::Manhattan),
                "hamming" => Ok(SimilarityMetric::Hamming),
                "jaccard" => Ok(SimilarityMetric::Jaccard),
                _ => Err(SpaceError::SimilarityMatchError {
                    reason: format!(
                        "Unknown similarity metric '{}'. Supported: cos, dot, euc, manhattan, hamming, jaccard",
                        s
                    ),
                }),
            };
        }
    }

    Err(SpaceError::SimilarityMatchError {
        reason: "Similarity metric must be a string (e.g., \"cos\", \"dot\", \"euc\")".to_string(),
    })
}

/// Extract a rank function from a Par.
///
/// Currently only `"topk"` is supported.
///
/// # Arguments
/// - `par`: The Par containing the rank function string
///
/// # Returns
/// - `Ok(String)`: The extracted rank function name
/// - `Err(SpaceError)`: If the Par doesn't contain a valid rank function
pub fn extract_rank_function_from_par(par: &Option<Par>) -> Result<String, SpaceError> {
    let par = par.as_ref().ok_or_else(|| SpaceError::SimilarityMatchError {
        reason: "Rank function value is missing".to_string(),
    })?;

    // Extract string from Par
    for expr in &par.exprs {
        if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
            // Validate rank function
            return match s.to_lowercase().as_str() {
                "topk" | "top_k" => Ok("topk".to_string()),
                _ => Err(SpaceError::SimilarityMatchError {
                    reason: format!(
                        "Unknown rank function '{}'. Supported: topk",
                        s
                    ),
                }),
            };
        }
    }

    Err(SpaceError::SimilarityMatchError {
        reason: "Rank function must be a string (e.g., \"topk\")".to_string(),
    })
}

// =============================================================================
// Pattern Modifier Extraction from EFunction
// =============================================================================

/// Extracted pattern modifiers from EFunction calls.
///
/// This struct consolidates the modifier information extracted from `sim` and `rank`
/// EFunction calls into a unified representation for space operations.
#[derive(Debug, Clone)]
pub struct ExtractedModifiers {
    /// Query embedding vector (from first argument of sim or rank)
    pub query_embedding: Option<Vec<f32>>,
    /// Similarity metric (from sim function)
    pub metric: Option<SimilarityMetric>,
    /// Similarity threshold (from sim params)
    pub threshold: Option<f32>,
    /// Top-K limit (from rank params)
    pub top_k: Option<usize>,
    /// Rank function name (for future extensibility)
    pub rank_function: Option<String>,
}

impl Default for ExtractedModifiers {
    fn default() -> Self {
        Self {
            query_embedding: None,
            metric: None,
            threshold: None,
            top_k: None,
            rank_function: None,
        }
    }
}

/// Extract modifier information from a list of EFunction calls.
///
/// EFunction format:
/// - `sim(query, metric, threshold, ...)` -> extracts embedding, metric, threshold
/// - `rank(query, function, params...)` -> extracts embedding, rank function, top_k
///
/// # Arguments
/// - `modifiers`: List of EFunction pattern modifiers
///
/// # Returns
/// - `Ok(ExtractedModifiers)`: The extracted modifier values
/// - `Err(SpaceError)`: If extraction fails
pub fn extract_modifiers_from_efunctions(modifiers: &[EFunction]) -> Result<ExtractedModifiers, SpaceError> {
    let mut result = ExtractedModifiers::default();

    for efunction in modifiers {
        match efunction.function_name.as_str() {
            "sim" => {
                // sim(query, [metric, [threshold, ...]])
                // Arguments: [0]=query, [1]=metric (optional), [2]=threshold (optional)
                if efunction.arguments.is_empty() {
                    return Err(SpaceError::SimilarityMatchError {
                        reason: "sim modifier requires at least a query argument".to_string(),
                    });
                }

                // Extract query embedding from first argument
                result.query_embedding = Some(extract_embedding_from_par(&Some(efunction.arguments[0].clone()))?);

                // Extract metric from second argument if present
                if efunction.arguments.len() > 1 {
                    result.metric = Some(extract_metric_from_par(&Some(efunction.arguments[1].clone()))?);
                }

                // Extract threshold from third argument if present
                if efunction.arguments.len() > 2 {
                    result.threshold = Some(extract_threshold_from_par(&Some(efunction.arguments[2].clone()))?);
                }
            }
            "rank" => {
                // rank(query, function, [params...])
                // Arguments: [0]=query, [1]=function, [2]=k (for topk)
                if efunction.arguments.len() < 2 {
                    return Err(SpaceError::SimilarityMatchError {
                        reason: "rank modifier requires query and function arguments".to_string(),
                    });
                }

                // If we haven't extracted the query yet, use rank's query
                if result.query_embedding.is_none() {
                    result.query_embedding = Some(extract_embedding_from_par(&Some(efunction.arguments[0].clone()))?);
                }

                // Extract rank function from second argument
                result.rank_function = Some(extract_rank_function_from_par(&Some(efunction.arguments[1].clone()))?);

                // Extract top-K from third argument if present
                if efunction.arguments.len() > 2 {
                    result.top_k = Some(extract_top_k_from_par(&Some(efunction.arguments[2].clone()))?);
                } else {
                    // Default to 1 for topk without explicit K
                    result.top_k = Some(1);
                }
            }
            other => {
                tracing::warn!("Unknown pattern modifier function: {}", other);
                // Skip unknown modifiers for forward compatibility
            }
        }
    }

    Ok(result)
}

/// Extract the GPrivate ID from a channel Par.
///
/// Channels in Rholang are represented as Par values containing a GUnforgeable
/// with a GPrivate body. This function extracts the unique ID bytes from such
/// a channel representation.
///
/// # Arguments
/// - `channel`: The Par representing a channel (should contain a GPrivate)
///
/// # Returns
/// - `Some(Vec<u8>)`: The GPrivate ID bytes if present
/// - `None`: If the Par doesn't contain a GPrivate channel
pub fn extract_channel_id_from_par(channel: &Par) -> Option<Vec<u8>> {
    for unf in &channel.unforgeables {
        if let Some(UnfInstance::GPrivateBody(g_private)) = &unf.unf_instance {
            return Some(g_private.id.clone());
        }
    }
    None
}

// =============================================================================
// Map-based Embedding Extraction for VectorDB
// =============================================================================

/// Extract an embedding from a Par that contains a map with an "embedding" key.
///
/// This function is used during produce to extract embeddings from data sent to
/// VectorDB-backed channels. The data format follows industry standards (Pinecone,
/// Qdrant, Weaviate) where records are stored as dictionaries with explicit fields.
///
/// # Expected Rholang Data Format
///
/// ```rholang
/// {"id": 0, "title": "Document Title", "embedding": [90, 5, 10, 20]}
/// ```
///
/// # Arguments
///
/// - `par`: The Par containing the map with embedding data
/// - `embedding_type`: The expected embedding format (Boolean, Integer, Float)
/// - `dimensions`: The expected number of embedding dimensions
///
/// # Returns
///
/// - `Ok(Vec<f32>)`: The extracted embedding vector as floats
/// - `Err(SpaceError)`: If extraction fails (wrong format, missing key, etc.)
///
/// # Embedding Type Handling
///
/// - **Boolean**: `[0, 1, 1, 0]` → `[0.0, 1.0, 1.0, 0.0]`
/// - **Integer**: `[90, 5, 10, 20]` → `[0.9, 0.05, 0.1, 0.2]` (scaled from 0-100)
/// - **Float**: `"0.9,0.05,0.1,0.2"` → parsed to `Vec<f32>`
pub fn extract_embedding_from_map(
    par: &Par,
    embedding_type: EmbeddingType,
    dimensions: usize,
) -> Result<Vec<f32>, SpaceError> {
    // Create a key Par for "embedding" lookup
    let embedding_key = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::GString("embedding".to_string())),
    }]);

    // Look for an EMap in the Par's expressions
    for expr in &par.exprs {
        if let Some(ExprInstance::EMapBody(emap)) = &expr.expr_instance {
            // Convert to ParMap for key-value access
            let par_map = ParMapTypeMapper::emap_to_par_map(emap.clone());

            // Look up the "embedding" key
            let embedding_par = par_map.ps.get_or_else(embedding_key.clone(), Par::default());

            // Check if we got a valid result (not default)
            if embedding_par == Par::default() {
                return Err(SpaceError::EmbeddingExtractionError {
                    description: "VectorDB data must contain an 'embedding' key. \
                             Expected format: {\"id\": ..., \"title\": ..., \"embedding\": [...]}".to_string(),
                });
            }

            // Extract the embedding based on type
            return extract_embedding_by_type(&embedding_par, embedding_type, dimensions);
        }
    }

    Err(SpaceError::EmbeddingExtractionError {
        description: "VectorDB data must be a map with an 'embedding' key. \
                 Expected format: {\"id\": ..., \"title\": ..., \"embedding\": [...]}".to_string(),
    })
}

/// Extract embedding based on the configured embedding type.
fn extract_embedding_by_type(
    par: &Par,
    embedding_type: EmbeddingType,
    dimensions: usize,
) -> Result<Vec<f32>, SpaceError> {
    match embedding_type {
        EmbeddingType::Boolean => extract_boolean_embedding(par, dimensions),
        EmbeddingType::Integer => extract_integer_embedding(par, dimensions),
        EmbeddingType::Float => extract_float_string_embedding(par, dimensions),
    }
}

/// Extract a boolean embedding: [0, 1, 1, 0] → [0.0, 1.0, 1.0, 0.0]
fn extract_boolean_embedding(par: &Par, dimensions: usize) -> Result<Vec<f32>, SpaceError> {
    // Look for an EList in the Par
    for expr in &par.exprs {
        if let Some(ExprInstance::EListBody(list)) = &expr.expr_instance {
            if list.ps.len() != dimensions {
                return Err(SpaceError::DimensionMismatch {
                    expected: dimensions,
                    actual: list.ps.len(),
                });
            }

            let mut embedding = Vec::with_capacity(dimensions);
            for elem_par in &list.ps {
                let value = extract_number_from_par(elem_par)?;
                if value != 0 && value != 1 {
                    return Err(SpaceError::EmbeddingExtractionError {
                        description: format!(
                            "Boolean embedding type requires values 0 or 1, found: {}",
                            value
                        ),
                    });
                }
                embedding.push(if value != 0 { 1.0 } else { 0.0 });
            }
            return Ok(embedding);
        }
    }

    Err(SpaceError::EmbeddingExtractionError {
        description: "Boolean embedding must be a list of 0/1 values".to_string(),
    })
}

/// Extract an integer embedding: [90, 5, 10, 20] → [0.9, 0.05, 0.1, 0.2]
fn extract_integer_embedding(par: &Par, dimensions: usize) -> Result<Vec<f32>, SpaceError> {
    // Look for an EList in the Par
    for expr in &par.exprs {
        if let Some(ExprInstance::EListBody(list)) = &expr.expr_instance {
            if list.ps.len() != dimensions {
                return Err(SpaceError::DimensionMismatch {
                    expected: dimensions,
                    actual: list.ps.len(),
                });
            }

            let mut embedding = Vec::with_capacity(dimensions);
            for elem_par in &list.ps {
                let value = extract_number_from_par(elem_par)?;
                if value < 0 || value > 100 {
                    return Err(SpaceError::EmbeddingExtractionError {
                        description: format!(
                            "Integer embedding type requires values in 0-100 range, found: {}",
                            value
                        ),
                    });
                }
                // Scale from 0-100 to 0.0-1.0
                embedding.push(value as f32 / 100.0);
            }
            return Ok(embedding);
        }
    }

    Err(SpaceError::EmbeddingExtractionError {
        description: "Integer embedding must be a list of integers (0-100 scale)".to_string(),
    })
}

/// Extract a float string embedding: "0.9,0.05,0.1,0.2" → [0.9, 0.05, 0.1, 0.2]
fn extract_float_string_embedding(par: &Par, dimensions: usize) -> Result<Vec<f32>, SpaceError> {
    // Look for a GString in the Par
    for expr in &par.exprs {
        if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() != dimensions {
                return Err(SpaceError::DimensionMismatch {
                    expected: dimensions,
                    actual: parts.len(),
                });
            }

            let mut embedding = Vec::with_capacity(dimensions);
            for (i, part) in parts.iter().enumerate() {
                let value = part.trim().parse::<f32>().map_err(|e| {
                    SpaceError::EmbeddingExtractionError {
                        description: format!(
                            "Failed to parse float at position {}: '{}' ({})",
                            i, part, e
                        ),
                    }
                })?;
                embedding.push(value);
            }
            return Ok(embedding);
        }
    }

    Err(SpaceError::EmbeddingExtractionError {
        description: "Float embedding must be a comma-separated string of floats (e.g., \"0.1,0.2,0.3\")".to_string(),
    })
}

// =============================================================================
// Similarity Computation Helpers
// =============================================================================

/// Compute cosine similarity between two embedding vectors.
///
/// Both vectors are L2-normalized before computing the dot product.
/// This matches the behavior of VectorDBDataCollection's similarity computation.
///
/// # Arguments
/// - `a`: First embedding vector
/// - `b`: Second embedding vector
///
/// # Returns
/// Cosine similarity score in range [-1.0, 1.0], or 0.0 if either vector is zero.
pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Compute L2 norms
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    // Compute dot product of normalized vectors
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (norm_a * norm_b)
}

/// Compute dot product of two vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Dot product score, or 0.0 if vectors have different lengths or are empty.
pub fn compute_dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute Euclidean distance-based similarity of two vectors.
///
/// Converts Euclidean distance to a similarity score using: 1 / (1 + distance)
/// This ensures similarity is in range [0, 1], where 1 means identical vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Euclidean similarity score in range [0.0, 1.0], or 0.0 if vectors have different lengths or are empty.
pub fn compute_euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Compute Euclidean distance
    let distance: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();

    // Convert distance to similarity: 1 / (1 + distance)
    1.0 / (1.0 + distance)
}

/// Compute Manhattan distance-based similarity of two vectors.
///
/// Converts Manhattan distance to a similarity score using: 1 / (1 + distance)
/// This ensures similarity is in range [0, 1], where 1 means identical vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Manhattan similarity score in range [0.0, 1.0], or 0.0 if vectors have different lengths or are empty.
pub fn compute_manhattan_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Compute Manhattan distance (L1 norm)
    let distance: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum();

    // Convert distance to similarity: 1 / (1 + distance)
    1.0 / (1.0 + distance)
}

/// Compute Hamming similarity of two vectors.
///
/// Hamming distance counts the number of positions where elements differ.
/// Similarity is computed as: 1 - (hamming_distance / length)
///
/// For float vectors, elements are considered different if their difference
/// exceeds a small epsilon (0.001).
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Hamming similarity score in range [0.0, 1.0], or 0.0 if vectors have different lengths or are empty.
pub fn compute_hamming_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    const EPSILON: f32 = 0.001;
    let different_count = a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (*x - *y).abs() > EPSILON)
        .count();

    1.0 - (different_count as f32 / a.len() as f32)
}

/// Compute Jaccard similarity of two vectors.
///
/// Jaccard similarity = |A ∩ B| / |A ∪ B|
/// For float vectors, we interpret non-zero elements as "present".
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Jaccard similarity score in range [0.0, 1.0], or 0.0 if both vectors are zero.
pub fn compute_jaccard_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    const EPSILON: f32 = 0.001;
    let mut intersection = 0;
    let mut union = 0;

    for (x, y) in a.iter().zip(b.iter()) {
        let a_present = x.abs() > EPSILON;
        let b_present = y.abs() > EPSILON;

        if a_present || b_present {
            union += 1;
            if a_present && b_present {
                intersection += 1;
            }
        }
    }

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Select and compute similarity using the specified metric.
///
/// This is a convenience function that dispatches to the appropriate
/// similarity computation based on the metric enum.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
/// * `metric` - The similarity metric to use
///
/// # Returns
/// Similarity score according to the specified metric.
pub fn compute_similarity(a: &[f32], b: &[f32], metric: &SimilarityMetric) -> f32 {
    match metric {
        SimilarityMetric::Cosine => compute_cosine_similarity(a, b),
        SimilarityMetric::DotProduct => compute_dot_product(a, b),
        SimilarityMetric::Euclidean => compute_euclidean_similarity(a, b),
        SimilarityMetric::Manhattan => compute_manhattan_similarity(a, b),
        SimilarityMetric::Hamming => compute_hamming_similarity(a, b),
        SimilarityMetric::Jaccard => compute_jaccard_similarity(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = compute_cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = compute_cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = compute_dot_product(&a, &b);
        assert!((dot - 32.0).abs() < 0.0001); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_euclidean_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = compute_euclidean_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_hamming_similarity() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 1.0, 0.0];
        let sim = compute_hamming_similarity(&a, &b);
        assert!((sim - 0.75).abs() < 0.0001); // 3/4 match
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];
        let sim = compute_jaccard_similarity(&a, &b);
        // Union: {0, 1, 2}, Intersection: {0}
        assert!((sim - (1.0 / 3.0)).abs() < 0.0001);
    }

    #[test]
    fn test_extracted_modifiers_default() {
        let mods = ExtractedModifiers::default();
        assert!(mods.query_embedding.is_none());
        assert!(mods.metric.is_none());
        assert!(mods.threshold.is_none());
        assert!(mods.top_k.is_none());
        assert!(mods.rank_function.is_none());
    }
}
