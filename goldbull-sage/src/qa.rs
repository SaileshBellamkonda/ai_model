use anyhow::Result;
use goldbull_core::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Question answering request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QARequest {
    /// The question to answer
    pub question: String,
    /// Optional context to search for the answer
    pub context: Option<String>,
    /// Type of question
    pub question_type: QuestionType,
    /// Maximum length of the answer
    pub max_answer_length: usize,
    /// Temperature for answer generation
    pub temperature: f64,
    /// Whether to use context-aware answering
    pub use_context: bool,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for QARequest {
    fn default() -> Self {
        Self {
            question: String::new(),
            context: None,
            question_type: QuestionType::Factual,
            max_answer_length: 100,
            temperature: 0.1,
            use_context: true,
            metadata: HashMap::new(),
        }
    }
}

/// Question answering response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAResponse {
    /// The generated answer
    pub answer: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Type of question that was answered
    pub question_type: QuestionType,
    /// Source information for the answer
    pub sources: Vec<AnswerSource>,
    /// Response metadata
    pub metadata: QAMetadata,
}

/// Types of questions supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuestionType {
    /// Factual questions requiring specific information
    Factual,
    /// Analytical questions requiring reasoning
    Analytical,
    /// Yes/No questions
    YesNo,
    /// Multiple choice questions
    MultipleChoice,
    /// Open-ended questions
    OpenEnded,
    /// Definition questions
    Definition,
    /// Procedural questions (how-to)
    Procedural,
}

/// Source information for answers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerSource {
    /// Source identifier
    pub id: String,
    /// Source title or name
    pub title: String,
    /// Relevant excerpt from source
    pub excerpt: String,
    /// Confidence in this source
    pub relevance_score: f64,
    /// URL or reference to full source
    pub url: Option<String>,
}

/// Metadata for QA responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAMetadata {
    /// Time taken to generate answer (ms)
    pub generation_time_ms: u64,
    /// Model version used
    pub model_version: String,
    /// Number of tokens in question
    pub question_tokens: usize,
    /// Number of tokens in context
    pub context_tokens: usize,
    /// Number of tokens in answer
    pub answer_tokens: usize,
    /// Additional processing information
    pub processing_info: HashMap<String, String>,
}

impl Default for QAMetadata {
    fn default() -> Self {
        Self {
            generation_time_ms: 0,
            model_version: "goldbull-sage-1.0".to_string(),
            question_tokens: 0,
            context_tokens: 0,
            answer_tokens: 0,
            processing_info: HashMap::new(),
        }
    }
}

/// Question analysis results
#[derive(Debug, Clone)]
pub struct QuestionAnalysis {
    /// Detected question type
    pub question_type: QuestionType,
    /// Keywords extracted from question
    pub keywords: Vec<String>,
    /// Expected answer format
    pub expected_format: AnswerFormat,
    /// Complexity score (0.0 - 1.0)
    pub complexity: f64,
}

/// Expected answer formats
#[derive(Debug, Clone, PartialEq)]
pub enum AnswerFormat {
    /// Short factual answer
    Short,
    /// Detailed explanation
    Detailed,
    /// Yes/No response
    Boolean,
    /// List of items
    List,
    /// Numerical answer
    Numerical,
    /// Date/Time answer
    DateTime,
}

/// Question classifier for determining question types
pub struct QuestionClassifier {
    /// Patterns for different question types
    patterns: HashMap<QuestionType, Vec<String>>,
}

impl QuestionClassifier {
    /// Create a new question classifier
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Factual question patterns
        patterns.insert(QuestionType::Factual, vec![
            "what is".to_string(),
            "who is".to_string(),
            "where is".to_string(),
            "when did".to_string(),
            "which".to_string(),
        ]);
        
        // Yes/No question patterns
        patterns.insert(QuestionType::YesNo, vec![
            "is".to_string(),
            "are".to_string(),
            "can".to_string(),
            "could".to_string(),
            "will".to_string(),
            "would".to_string(),
            "do".to_string(),
            "does".to_string(),
            "did".to_string(),
        ]);
        
        // Analytical question patterns
        patterns.insert(QuestionType::Analytical, vec![
            "why".to_string(),
            "how".to_string(),
            "explain".to_string(),
            "analyze".to_string(),
            "compare".to_string(),
            "evaluate".to_string(),
        ]);
        
        // Definition question patterns
        patterns.insert(QuestionType::Definition, vec![
            "define".to_string(),
            "what does".to_string(),
            "meaning of".to_string(),
            "definition".to_string(),
        ]);
        
        // Procedural question patterns
        patterns.insert(QuestionType::Procedural, vec![
            "how to".to_string(),
            "steps to".to_string(),
            "procedure".to_string(),
            "process".to_string(),
        ]);
        
        Self { patterns }
    }
    
    /// Classify a question to determine its type
    pub fn classify(&self, question: &str) -> QuestionAnalysis {
        let question_lower = question.to_lowercase();
        let mut scores = HashMap::new();
        
        // Score each question type based on pattern matches
        for (question_type, patterns) in &self.patterns {
            let mut score = 0.0;
            for pattern in patterns {
                if question_lower.contains(pattern) {
                    score += 1.0;
                }
            }
            scores.insert(*question_type, score);
        }
        
        // Find the highest scoring question type
        let question_type = scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(qt, _)| qt)
            .unwrap_or(QuestionType::OpenEnded);
        
        // Extract keywords (simple word extraction)
        let keywords: Vec<String> = question
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .filter(|word| !is_stop_word(word))
            .map(|word| word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty())
            .collect();
        
        // Determine expected answer format
        let expected_format = match question_type {
            QuestionType::YesNo => AnswerFormat::Boolean,
            QuestionType::Factual => AnswerFormat::Short,
            QuestionType::Analytical => AnswerFormat::Detailed,
            QuestionType::Definition => AnswerFormat::Detailed,
            QuestionType::Procedural => AnswerFormat::List,
            _ => AnswerFormat::Short,
        };
        
        // Calculate complexity based on question length and keywords
        let complexity = (question.len() as f64 / 100.0 + keywords.len() as f64 / 10.0).min(1.0);
        
        QuestionAnalysis {
            question_type,
            keywords,
            expected_format,
            complexity,
        }
    }
}

impl Default for QuestionClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a word is a stop word
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
    ];
    
    STOP_WORDS.contains(&word.to_lowercase().as_str())
}

/// Context processor for extracting relevant information
pub struct ContextProcessor {
    /// Maximum context length to consider
    max_context_length: usize,
}

impl ContextProcessor {
    /// Create a new context processor
    pub fn new(max_context_length: usize) -> Self {
        Self { max_context_length }
    }
    
    /// Process context to extract relevant passages for a question
    pub fn extract_relevant_passages(&self, question: &str, context: &str) -> Result<Vec<String>> {
        let question_analysis = QuestionClassifier::new().classify(question);
        let question_keywords = &question_analysis.keywords;
        
        // Split context into sentences
        let sentences: Vec<&str> = context
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        
        // Score sentences based on keyword overlap
        let mut scored_sentences: Vec<(f64, String)> = sentences
            .into_iter()
            .map(|sentence| {
                let sentence_lower = sentence.to_lowercase();
                let score = question_keywords
                    .iter()
                    .map(|keyword| if sentence_lower.contains(keyword) { 1.0 } else { 0.0 })
                    .sum::<f64>() / question_keywords.len().max(1) as f64;
                
                (score, sentence.to_string())
            })
            .collect();
        
        // Sort by relevance score
        scored_sentences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top scoring sentences up to max length
        let mut relevant_passages = Vec::new();
        let mut total_length = 0;
        
        for (score, sentence) in scored_sentences {
            if score > 0.0 && total_length + sentence.len() <= self.max_context_length {
                total_length += sentence.len();
                relevant_passages.push(sentence);
            }
        }
        
        Ok(relevant_passages)
    }
}

impl Default for ContextProcessor {
    fn default() -> Self {
        Self::new(2048) // Default to 2K characters
    }
}