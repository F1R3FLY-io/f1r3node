use std::sync::Arc;
use tokio::sync::RwLock;

use super::engine::{noop, Engine};

/// EngineCell is a concurrency-safe mutable container for the current Engine instance.
///
/// This is the Rust equivalent of Scala's Cell[F, Engine[F]] (see EngineCell.scala).
/// It provides async operations that match the Scala F[_] monadic interface.
///
/// Usage:
///   let engine_cell = EngineCell::init().await?;
///   let engine = engine_cell.read().await?;  // Returns Arc<dyn Engine>
///   engine_cell.set(Arc::new(MyEngine::new(...))).await?;
///
/// This implementation provides 1:1 API compatibility with the Scala EngineCell.
/// Uses Arc internally to avoid expensive cloning on read operations.\
#[derive(Clone)]
pub struct EngineCell {
    inner: Arc<RwLock<Arc<dyn Engine>>>,
}

impl EngineCell {
    /// Initialize EngineCell with NoopEngine (equivalent to Cell.mvarCell[F, Engine[F]](Engine.noop))
    pub fn init() -> Self {
        let engine = Arc::new(noop());
        EngineCell {
            inner: Arc::new(RwLock::new(engine)),
        }
    }

    /// Read the current engine (equivalent to Cell.read: F[Engine[F]])
    /// This is the most frequently used method in the Scala codebase
    #[inline]
    pub async fn get(&self) -> Arc<dyn Engine> {
        Arc::clone(&*self.inner.read().await)
    }

    /// Set the engine to a new instance (equivalent to Cell.set(s: Engine[F]): F[Unit])
    #[inline]
    pub async fn set(&self, engine: Arc<dyn Engine>) {
        *self.inner.write().await = engine;
    }
}
