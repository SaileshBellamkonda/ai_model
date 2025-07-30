use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multimodal processing request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalRequest {
    /// List of input modalities
    pub inputs: Vec<ModalityInput>,
    /// Output modalities to generate
    pub output_modalities: Vec<ModalityType>,
    /// Processing options
    pub options: HashMap<String, String>,
}

impl Default for MultimodalRequest {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            output_modalities: vec![ModalityType::Text],
            options: HashMap::new(),
        }
    }
}

/// Input modality data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityInput {
    /// The modality type and data
    pub modality: InputModality,
    /// Metadata for this input
    pub metadata: HashMap<String, String>,
}

/// Input modality types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputModality {
    /// Text input
    Text { content: String },
    /// Image input
    Image { data: Vec<u8> },
    /// Audio input
    Audio { data: Vec<f32>, sample_rate: u32 },
}

/// Multimodal processing response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalResponse {
    /// Text output (optional)
    pub text_output: Option<String>,
    /// Image output (optional)  
    pub image_output: Option<Vec<u8>>,
    /// Audio output (optional)
    pub audio_output: Option<Vec<f32>>,
    /// Overall confidence score
    pub confidence: f64,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Types of modalities supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModalityType {
    /// Text modality
    Text,
    /// Vision modality
    Vision,
    /// Audio modality
    Audio,
}