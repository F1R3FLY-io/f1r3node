// Storage management module for Rholang-RGB integration
//
// This module provides configurable storage options for RholangCodex instances,
// addressing the directory management and persistence requirements.

use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Storage configuration for RholangCodex instances
#[derive(Debug, Clone)]
pub enum StorageConfig {
    /// Persistent storage at the specified path
    /// Use for production RGB validation with state continuity
    Persistent(PathBuf),
    
    /// Ephemeral storage using temporary directory
    /// Use for one-shot validations or testing
    Ephemeral,
    
    /// Default persistent storage location
    /// Uses ~/.rholang-sonic or similar OS-appropriate directory
    Default,
}

pub struct StorageManager {
    storage_path: PathBuf,
    _temp_dir: Option<TempDir>, // Only used for ephemeral storage
}

impl StorageManager {
    pub fn new(config: StorageConfig) -> Result<Self, Box<dyn std::error::Error>> {
        match config {
            StorageConfig::Persistent(path) => {
                std::fs::create_dir_all(&path)?;
                Ok(Self {
                    storage_path: path,
                    _temp_dir: None,
                })
            }
            
            StorageConfig::Ephemeral => {
                let temp_dir = tempfile::tempdir()?;
                let storage_path = temp_dir.path().to_path_buf();
                Ok(Self {
                    storage_path,
                    _temp_dir: Some(temp_dir),
                })
            }
            
            StorageConfig::Default => {
                let default_path = Self::default_storage_path()?;
                std::fs::create_dir_all(&default_path)?;
                Ok(Self {
                    storage_path: default_path,
                    _temp_dir: None,
                })
            }
        }
    }
    
    pub fn storage_path(&self) -> &Path {
        &self.storage_path
    }
    
    pub fn is_first_time(&self) -> bool {
        !self.storage_path.join("rspace.lmdb").exists()
    }
    
    fn default_storage_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        // Use OS-appropriate directories
        let base_dir = if let Some(home) = dirs::home_dir() {
            home.join(".rholang-sonic")
        } else {
            // Fallback for systems without home directory
            PathBuf::from(".rholang-sonic")
        };
        
        Ok(base_dir)
    }
}

impl Default for StorageManager {
    fn default() -> Self {
        Self::new(StorageConfig::Default)
            .expect("Failed to create default storage manager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_persistent_storage() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().join("test_persistent");
        
        let manager = StorageManager::new(StorageConfig::Persistent(storage_path.clone())).unwrap();
        
        assert_eq!(manager.storage_path(), storage_path);
        assert!(storage_path.exists());
        assert!(manager.is_first_time()); // No LMDB file yet
    }
    
    #[test]
    fn test_ephemeral_storage() {
        let manager = StorageManager::new(StorageConfig::Ephemeral).unwrap();
        
        assert!(manager.storage_path().exists());
        assert!(manager.is_first_time());
        // When manager is dropped, temp directory should be cleaned up automatically
    }
    
    #[test]
    fn test_first_time_detection() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().join("test_first_time");
        
        let manager = StorageManager::new(StorageConfig::Persistent(storage_path.clone())).unwrap();
        assert!(manager.is_first_time());
        
        // Simulate LMDB file creation
        fs::create_dir_all(storage_path.join("rspace.lmdb")).unwrap();
        assert!(!manager.is_first_time());
    }
    
    #[test]
    fn test_ephemeral_cleanup() {
        let temp_path: std::path::PathBuf;
        
        // Create ephemeral storage and capture the path
        {
            let manager = StorageManager::new(StorageConfig::Ephemeral).unwrap();
            temp_path = manager.storage_path().to_path_buf();
            
            // Verify the directory exists while manager is alive
            assert!(temp_path.exists(), "Temporary directory should exist while StorageManager is alive");
            assert!(temp_path.is_dir(), "Temporary path should be a directory");
            
            // Create a test file to ensure the directory is really there
            let test_file = temp_path.join("test_cleanup.txt");
            fs::write(&test_file, b"test content").unwrap();
            assert!(test_file.exists(), "Test file should exist");
        } // manager goes out of scope here
        
        // Give the OS a moment to clean up (usually instantaneous but be safe)
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        // Verify the directory has been cleaned up
        assert!(!temp_path.exists(), 
            "Temporary directory should be cleaned up when StorageManager is dropped. Path: {}", 
            temp_path.display()
        );
    }
}
