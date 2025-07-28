use thiserror::Error;

#[derive(Error, Debug)]
pub enum GoldbullError {
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Tensor operation error: {0}")]
    Tensor(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Inference error: {0}")]
    Inference(String),
}

pub type Result<T> = std::result::Result<T, GoldbullError>;