use thiserror::Error;

pub type Result<T> = std::result::Result<T, RmcError>;

#[derive(Debug, Error)]
pub enum RmcError {
    #[error("{0}")]
    Message(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("invalid state: {0}")]
    InvalidState(String),

    #[error("duplicate result path: {0}")]
    DuplicateResult(String),
}
