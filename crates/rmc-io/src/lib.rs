//! Checkpoint and restart helpers.
//!
//! The formats are versioned envelopes around any serde-serializable payload. Simulation code
//! decides what belongs in the payload: typically state, RNG, kernel/update set, measurements, and
//! any user metadata needed to resume.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use rmc_core::RmcError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CHECKPOINT_VERSION: u32 = 1;

pub type Result<T> = std::result::Result<T, IoError>;
pub type ResultMap = BTreeMap<String, serde_json::Value>;
pub type EncodedResultMap = BTreeMap<String, Vec<u8>>;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("binary serialization error: {0}")]
    Binary(#[from] Box<bincode::ErrorKind>),
    #[error("unsupported checkpoint version {found}, expected {expected}")]
    UnsupportedVersion { expected: u32, found: u32 },
}

/// Versioned checkpoint envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint<T> {
    pub version: u32,
    pub payload: T,
}

/// In-memory `path -> JSON value` map with checkpoint round-tripping.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MapSink {
    results: ResultMap,
}

impl MapSink {
    pub fn new() -> Self {
        Self {
            results: BTreeMap::new(),
        }
    }

    pub fn results(&self) -> &ResultMap {
        &self.results
    }

    pub fn into_results(self) -> ResultMap {
        self.results
    }

    /// Insert a serializable value under `path`; duplicate paths are rejected.
    pub fn put<T: serde::Serialize>(&mut self, path: &str, value: &T) -> rmc_core::Result<()> {
        if self.results.contains_key(path) {
            return Err(RmcError::DuplicateResult(path.to_string()));
        }
        let value = serde_json::to_value(value)
            .map_err(|err| RmcError::Message(format!("result serialization failed: {err}")))?;
        self.results.insert(path.to_string(), value);
        Ok(())
    }

    pub fn into_checkpoint(self) -> Checkpoint<ResultMap> {
        Checkpoint::new(self.into_results())
    }

    pub fn from_checkpoint(checkpoint: Checkpoint<ResultMap>) -> Result<Self> {
        checkpoint.validate_version()?;
        Ok(Self {
            results: checkpoint.into_payload(),
        })
    }

    /// Encode result values as JSON bytes inside a checkpoint payload.
    ///
    /// This representation round-trips through binary checkpoint formats such as bincode, unlike
    /// `serde_json::Value` directly.
    pub fn to_encoded_checkpoint(&self) -> Result<Checkpoint<EncodedResultMap>> {
        let mut encoded = BTreeMap::new();
        for (path, value) in &self.results {
            encoded.insert(path.clone(), serde_json::to_vec(value)?);
        }
        Ok(Checkpoint::new(encoded))
    }

    pub fn from_encoded_checkpoint(checkpoint: Checkpoint<EncodedResultMap>) -> Result<Self> {
        checkpoint.validate_version()?;
        let mut results = BTreeMap::new();
        for (path, value) in checkpoint.into_payload() {
            results.insert(path, serde_json::from_slice(&value)?);
        }
        Ok(Self { results })
    }
}

impl From<ResultMap> for MapSink {
    fn from(results: ResultMap) -> Self {
        Self { results }
    }
}

impl<T> Checkpoint<T> {
    pub fn new(payload: T) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            payload,
        }
    }

    pub fn payload(&self) -> &T {
        &self.payload
    }

    pub fn payload_mut(&mut self) -> &mut T {
        &mut self.payload
    }

    pub fn into_payload(self) -> T {
        self.payload
    }

    pub fn validate_version(&self) -> Result<()> {
        validate_version(self.version)
    }
}

pub fn to_json_string<T: Serialize>(checkpoint: &Checkpoint<T>) -> Result<String> {
    Ok(serde_json::to_string_pretty(checkpoint)?)
}

pub fn to_json_vec<T: Serialize>(checkpoint: &Checkpoint<T>) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec_pretty(checkpoint)?)
}

pub fn from_json_str<T: DeserializeOwned>(json: &str) -> Result<Checkpoint<T>> {
    let checkpoint: Checkpoint<T> = serde_json::from_str(json)?;
    checkpoint.validate_version()?;
    Ok(checkpoint)
}

pub fn from_json_slice<T: DeserializeOwned>(json: &[u8]) -> Result<Checkpoint<T>> {
    let checkpoint: Checkpoint<T> = serde_json::from_slice(json)?;
    checkpoint.validate_version()?;
    Ok(checkpoint)
}

pub fn to_binary_vec<T: Serialize>(checkpoint: &Checkpoint<T>) -> Result<Vec<u8>> {
    Ok(bincode::serialize(checkpoint)?)
}

pub fn from_binary_slice<T: DeserializeOwned>(bytes: &[u8]) -> Result<Checkpoint<T>> {
    let checkpoint: Checkpoint<T> = bincode::deserialize(bytes)?;
    checkpoint.validate_version()?;
    Ok(checkpoint)
}

pub fn save_json<T: Serialize>(path: impl AsRef<Path>, checkpoint: &Checkpoint<T>) -> Result<()> {
    fs::write(path, to_json_vec(checkpoint)?)?;
    Ok(())
}

pub fn load_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Checkpoint<T>> {
    from_json_slice(&fs::read(path)?)
}

pub fn save_binary<T: Serialize>(path: impl AsRef<Path>, checkpoint: &Checkpoint<T>) -> Result<()> {
    fs::write(path, to_binary_vec(checkpoint)?)?;
    Ok(())
}

pub fn load_binary<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Checkpoint<T>> {
    from_binary_slice(&fs::read(path)?)
}

/// Save a checkpoint by writing a temporary sibling file and renaming it into place.
pub fn save_json_atomic<T: Serialize>(
    path: impl AsRef<Path>,
    checkpoint: &Checkpoint<T>,
) -> Result<()> {
    let path = path.as_ref();
    let tmp_path = temporary_path(path);

    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(&to_json_vec(checkpoint)?)?;
        file.sync_all()?;
    }

    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Save a binary checkpoint by writing a temporary sibling file and renaming it into place.
pub fn save_binary_atomic<T: Serialize>(
    path: impl AsRef<Path>,
    checkpoint: &Checkpoint<T>,
) -> Result<()> {
    let path = path.as_ref();
    let tmp_path = temporary_path(path);

    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(&to_binary_vec(checkpoint)?)?;
        file.sync_all()?;
    }

    fs::rename(&tmp_path, path)?;
    Ok(())
}

pub fn save_payload_json<T: Serialize>(path: impl AsRef<Path>, payload: &T) -> Result<()> {
    save_json(path, &Checkpoint::new(payload))
}

pub fn load_payload_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    Ok(load_json(path)?.into_payload())
}

pub fn save_payload_binary<T: Serialize>(path: impl AsRef<Path>, payload: &T) -> Result<()> {
    save_binary(path, &Checkpoint::new(payload))
}

pub fn load_payload_binary<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    Ok(load_binary(path)?.into_payload())
}

fn validate_version(version: u32) -> Result<()> {
    if version != CHECKPOINT_VERSION {
        return Err(IoError::UnsupportedVersion {
            expected: CHECKPOINT_VERSION,
            found: version,
        });
    }
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut tmp_path = path.to_path_buf();
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map_or_else(|| "tmp".to_string(), |extension| format!("{extension}.tmp"));
    tmp_path.set_extension(extension);
    tmp_path
}
