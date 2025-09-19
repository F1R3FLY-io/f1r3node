use std::sync::Arc;
use tokio::sync::RwLock;

use super::engine::{noop, Engine};
use crate::rust::errors::CasperError;

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
/// Uses Arc internally to avoid expensive cloning on read operations.
pub struct EngineCell {
    inner: Arc<RwLock<Arc<dyn Engine>>>,
}

impl EngineCell {
    /// Initialize EngineCell with NoopEngine (equivalent to Cell.mvarCell[F, Engine[F]](Engine.noop))
    /// Returns Future<Result<EngineCell, CasperError>> to match Scala's F[EngineCell[F]]
    pub async fn init() -> Result<Self, CasperError> {
        let engine = Arc::new(noop()?);
        Ok(EngineCell {
            inner: Arc::new(RwLock::new(engine)),
        })
    }

    /// Create EngineCell with NoopEngine synchronously (equivalent to Cell.unsafe[F, Engine[F]](Engine.noop))
    /// This is used where we don't want async initialization
    pub fn unsafe_init() -> Result<Self, CasperError> {
        let engine = Arc::new(noop()?);
        Ok(EngineCell {
            inner: Arc::new(RwLock::new(engine)),
        })
    }

    /// Read the current engine (equivalent to Cell.read: F[Engine[F]])
    /// This is the most frequently used method in the Scala codebase
    pub async fn read(&self) -> Result<Arc<dyn Engine>, CasperError> {
        // Return a cheap Arc clone, matching Scala's read behavior which doesn't hold locks
        // This is much more efficient than the previous clone_box() approach
        let guard = self.inner.read().await;
        Ok(Arc::clone(&*guard))
    }

    /// Convenience method to read engine as Box (for backwards compatibility if needed)
    pub async fn read_boxed(&self) -> Result<Box<dyn Engine>, CasperError> {
        let arc_engine = self.read().await?;
        Ok(Box::new(EngineArcAdapter { inner: arc_engine }))
    }

    /// Set the engine to a new instance (equivalent to Cell.set(s: Engine[F]): F[Unit])
    pub async fn set(&self, engine: Arc<dyn Engine>) -> Result<(), CasperError> {
        let mut guard = self.inner.write().await;
        *guard = engine;
        Ok(())
    }

    /// Convenience method to set engine from Box (for easier migration)
    pub async fn set_boxed(&self, engine: Box<dyn Engine>) -> Result<(), CasperError> {
        self.set(engine.into()).await
    }

    /// Modify the engine with a pure function (equivalent to Cell.modify(f: Engine[F] => Engine[F]): F[Unit])
    pub async fn modify<F>(&self, f: F) -> Result<(), CasperError>
    where
        F: FnOnce(Arc<dyn Engine>) -> Arc<dyn Engine> + Send,
    {
        let mut guard = self.inner.write().await;
        let current_engine = std::mem::replace(&mut *guard, Arc::new(noop()?));
        *guard = f(current_engine);
        Ok(())
    }

    /// Modify the engine with an async function (equivalent to Cell.flatModify(f: Engine[F] => F[Engine[F]]): F[Unit])
    pub async fn flat_modify<F, Fut>(&self, f: F) -> Result<(), CasperError>
    where
        F: FnOnce(Arc<dyn Engine>) -> Fut + Send,
        Fut: std::future::Future<Output = Result<Arc<dyn Engine>, CasperError>> + Send,
    {
        // Acquire write lock and hold it for the entire duration to prevent race conditions.
        // This matches Scala's MVar.take pattern where the value is exclusively held during
        // the async operation, preventing other threads from reading placeholder state or
        // causing lost updates from concurrent modifications.
        let mut guard = self.inner.write().await;
        let current_engine = std::mem::replace(&mut *guard, Arc::new(noop()?));

        // Execute the provided async function while holding the write lock.
        // This ensures no other thread can access the EngineCell during the operation.
        // Use cheap Arc::clone instead of expensive clone_box().
        let result = f(Arc::clone(&current_engine)).await;

        // Commit the new engine or restore the original on failure,
        // matching Scala's bracket-style resource handling.
        match result {
            Ok(new_engine) => {
                *guard = new_engine;
                Ok(())
            }
            Err(e) => {
                *guard = current_engine; // restore previous state
                Err(e)
            }
        }
        // Write lock is held until this point, ensuring atomic operation
    }

    /// Read with a transformation function (equivalent to Cell.reads[A](f: Engine[F] => A): F[A])
    pub async fn reads<A, F>(&self, f: F) -> Result<A, CasperError>
    where
        F: FnOnce(&dyn Engine) -> A + Send,
        A: Send,
    {
        let guard = self.inner.read().await;
        Ok(f(guard.as_ref()))
    }
}

impl Clone for EngineCell {
    fn clone(&self) -> Self {
        EngineCell {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct EngineArcAdapter {
    inner: std::sync::Arc<dyn Engine>,
}

#[async_trait::async_trait(?Send)]
impl Engine for EngineArcAdapter {
    async fn init(&self) -> Result<(), crate::rust::errors::CasperError> {
        self.inner.init().await
    }

    async fn handle(
        &self,
        peer: comm::rust::peer_node::PeerNode,
        msg: models::rust::casper::protocol::casper_message::CasperMessage,
    ) -> Result<(), crate::rust::errors::CasperError> {
        // We need &self here; delegate by temporarily cloning Arc and getting mutable reference via Arc::get_mut is not possible.
        // But Engine::handle now takes &self, so delegate directly.
        self.inner.handle(peer, msg).await
    }

    fn with_casper(&self) -> Option<&dyn crate::rust::casper::MultiParentCasper> {
        self.inner.with_casper()
    }

    fn clone_box(&self) -> Box<dyn Engine> {
        Box::new(EngineArcAdapter {
            inner: std::sync::Arc::clone(&self.inner),
        })
    }
}
