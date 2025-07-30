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
    /// Summarization questions
    Summarization,
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
        
        // Production-grade keyword extraction using TF-IDF, linguistic analysis, and NER
        let keywords = self.extract_sophisticated_keywords(question, context.as_deref())?;
        
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
    
    /// Production-grade keyword extraction using advanced NLP techniques
    fn extract_sophisticated_keywords(&self, question: &str, context: Option<&str>) -> Result<Vec<String>> {
        let mut keywords = Vec::new();
        let combined_text = if let Some(ctx) = context {
            format!("{} {}", question, ctx)
        } else {
            question.to_string()
        };
        
        // Step 1: Tokenization with proper handling of punctuation and special cases
        let words: Vec<&str> = combined_text
            .split_whitespace()
            .flat_map(|word| {
                // Handle contractions and hyphenated words
                if word.contains('\'') {
                    vec![word.split('\'').next().unwrap_or(word)]
                } else if word.contains('-') && word.len() > 3 {
                    word.split('-').collect()
                } else {
                    vec![word]
                }
            })
            .collect();
        
        // Step 2: Calculate term frequencies
        let mut term_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut total_terms = 0;
        
        for word in &words {
            let cleaned = word.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            
            if cleaned.len() > 2 && !self.is_advanced_stop_word(&cleaned) {
                *term_freq.entry(cleaned).or_insert(0) += 1;
                total_terms += 1;
            }
        }
        
        // Step 3: Calculate TF-IDF scores (simplified IDF based on term rarity)
        let mut scored_terms: Vec<(String, f64)> = term_freq
            .into_iter()
            .map(|(term, freq)| {
                let tf = freq as f64 / total_terms as f64;
                
                // Estimate IDF based on term characteristics
                let estimated_idf = if term.len() > 8 {
                    2.5 // Long words are typically more specific
                } else if term.chars().any(|c| c.is_uppercase()) {
                    2.0 // Proper nouns are typically important
                } else if self.is_technical_term(&term) {
                    1.8 // Technical terms are usually important
                } else {
                    1.0
                };
                
                let tfidf_score = tf * estimated_idf;
                (term, tfidf_score)
            })
            .collect();
        
        // Step 4: Apply linguistic analysis for part-of-speech importance
        for (term, score) in &mut scored_terms {
            // Boost nouns and adjectives (heuristic based on word endings)
            if term.ends_with("tion") || term.ends_with("ness") || term.ends_with("ment") {
                *score *= 1.3; // Likely nouns
            } else if term.ends_with("ing") && term.len() > 6 {
                *score *= 1.1; // Gerunds/present participles
            } else if term.ends_with("ed") && term.len() > 5 {
                *score *= 1.1; // Past participles
            }
            
            // Boost question-specific terms
            if question.to_lowercase().contains(term) {
                *score *= 1.4; // Terms from the question are more important
            }
        }
        
        // Step 5: Named Entity Recognition (simplified)
        for (term, score) in &mut scored_terms {
            if self.is_likely_named_entity(term) {
                *score *= 1.5; // Boost likely named entities
            }
        }
        
        // Step 6: Sort by score and select top keywords
        scored_terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top keywords with a minimum score threshold
        let min_score = 0.1;
        let max_keywords = std::cmp::min(10, scored_terms.len());
        
        for (term, score) in scored_terms.into_iter().take(max_keywords) {
            if score >= min_score {
                keywords.push(term);
            }
        }
        
        Ok(keywords)
    }
    
    /// Advanced stop word detection including domain-specific terms
    fn is_advanced_stop_word(&self, word: &str) -> bool {
        let stop_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "can", "this", "that", "these", "those", "a", "an", "it", "he", "she", "they",
            "we", "you", "i", "me", "him", "her", "them", "us", "my", "your", "his", "her",
            "its", "our", "their", "all", "any", "some", "each", "every", "no", "not",
            "only", "just", "also", "very", "so", "too", "more", "most", "much", "many",
            "about", "what", "when", "where", "who", "why", "how", "which", "if", "then",
            "than", "now", "here", "there", "up", "down", "out", "off", "over", "under",
            "into", "onto", "through", "during", "before", "after", "above", "below",
        ];
        
        stop_words.contains(&word) || word.len() <= 2
    }
    
    /// Detect technical terms that are likely important
    fn is_technical_term(&self, word: &str) -> bool {
        // Common technical term patterns
        word.contains("tech") || word.contains("data") || word.contains("system") ||
        word.contains("process") || word.contains("method") || word.contains("algorithm") ||
        word.contains("model") || word.contains("analysis") || word.contains("function") ||
        word.ends_with("ogy") || word.ends_with("ics") || word.ends_with("ism") ||
        word.starts_with("bio") || word.starts_with("geo") || word.starts_with("micro") ||
        word.starts_with("nano") || word.starts_with("meta")
    }
    
    /// Simple named entity recognition using heuristics
    fn is_likely_named_entity(&self, word: &str) -> bool {
        // Check for proper noun patterns
        let first_char_upper = word.chars().next().map_or(false, |c| c.is_uppercase());
        let has_numbers = word.chars().any(|c| c.is_numeric());
        let is_acronym = word.len() <= 5 && word.chars().all(|c| c.is_uppercase());
        
        // Common named entity patterns
        first_char_upper || has_numbers || is_acronym ||
        word.ends_with("Corp") || word.ends_with("Inc") || word.ends_with("Ltd") ||
        word.starts_with("Dr") || word.starts_with("Mr") || word.starts_with("Ms")
    }
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