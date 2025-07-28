use serde::{Deserialize, Serialize};

/// Available task types for the AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Generate text based on a prompt
    TextGeneration {
        max_tokens: usize,
        temperature: f32,
    },
    
    /// Complete code based on partial input
    CodeCompletion {
        language: String,
        context_lines: usize,
    },
    
    /// Answer questions with optional context
    QuestionAnswer {
        context: Option<String>,
    },
    
    /// Summarize text
    Summarization {
        max_length: usize,
        style: String, // "brief", "detailed", "bullet_points", etc.
    },
    
    /// Analyze visual content
    VisualAnalysis {
        image_data: Vec<u8>,
    },
}

/// Result of executing a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskResult {
    /// Text generation result
    TextGeneration {
        generated_text: String,
        tokens_generated: usize,
        finish_reason: String,
    },
    
    /// Code completion result
    CodeCompletion {
        completion: String,
        confidence_score: f32,
    },
    
    /// Question answering result
    QuestionAnswer {
        answer: String,
        confidence_score: f32,
        sources: Vec<String>,
    },
    
    /// Text summarization result
    Summarization {
        summary: String,
        compression_ratio: f32,
    },
    
    /// Visual analysis result
    VisualAnalysis {
        description: String,
        detected_objects: Vec<String>,
        confidence_scores: Vec<f32>,
    },
}

impl TaskResult {
    /// Get the number of output tokens if applicable
    pub fn output_tokens(&self) -> Option<usize> {
        match self {
            TaskResult::TextGeneration { tokens_generated, .. } => Some(*tokens_generated),
            TaskResult::CodeCompletion { completion, .. } => {
                // Rough token estimation (words * 1.3)
                Some((completion.split_whitespace().count() as f32 * 1.3) as usize)
            },
            TaskResult::QuestionAnswer { answer, .. } => {
                Some((answer.split_whitespace().count() as f32 * 1.3) as usize)
            },
            TaskResult::Summarization { summary, .. } => {
                Some((summary.split_whitespace().count() as f32 * 1.3) as usize)
            },
            TaskResult::VisualAnalysis { description, .. } => {
                Some((description.split_whitespace().count() as f32 * 1.3) as usize)
            },
        }
    }
    
    /// Get the main text output from the result
    pub fn text_output(&self) -> &str {
        match self {
            TaskResult::TextGeneration { generated_text, .. } => generated_text,
            TaskResult::CodeCompletion { completion, .. } => completion,
            TaskResult::QuestionAnswer { answer, .. } => answer,
            TaskResult::Summarization { summary, .. } => summary,
            TaskResult::VisualAnalysis { description, .. } => description,
        }
    }
}

/// Task execution context containing metadata and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub task_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub max_execution_time_ms: u64,
    pub priority: TaskPriority,
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for TaskContext {
    fn default() -> Self {
        Self {
            task_id: uuid::Uuid::new_v4().to_string(),
            user_id: None,
            session_id: None,
            max_execution_time_ms: 30000, // 30 seconds default
            priority: TaskPriority::Normal,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Priority levels for task execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Task execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed(TaskResult),
    Failed(String),
    Cancelled,
}