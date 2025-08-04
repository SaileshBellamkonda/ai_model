use goldbull_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub size: usize,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            size: 0,
        }
    }
    
    pub fn from_bpemb_url(_lang: &str, vocab_size: usize) -> Result<Self> {
        // Download and load BPEmb vocabulary
        // For now, create a mock vocabulary
        let mut vocab = Self::new();
        
        // Add basic vocabulary
        for i in 0..vocab_size.min(1000) {
            let token = format!("token_{i}");
            vocab.add_token(token);
        }
        
        Ok(vocab)
    }
    
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let vocab: Self = serde_json::from_str(&content)?;
        Ok(vocab)
    }
    
    pub fn save(&self, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    pub fn add_token(&mut self, token: String) -> u32 {
        if let Some(&id) = self.token_to_id.get(&token) {
            return id;
        }
        
        let id = self.size as u32;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
        self.size += 1;
        id
    }
    
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
    
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }
    
    pub fn len(&self) -> usize {
        self.size
    }
    
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeMerges {
    pub merges: Vec<(String, String)>,
}

impl BpeMerges {
    pub fn new() -> Self {
        Self {
            merges: Vec::new(),
        }
    }
    
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut merges = Vec::new();
        
        for line in content.lines().skip(1) { // Skip header
            if let Some((first, second)) = line.split_once(' ') {
                merges.push((first.to_string(), second.to_string()));
            }
        }
        
        Ok(Self { merges })
    }
    
    pub fn save(&self, path: &str) -> Result<()> {
        let mut content = String::from("#version: 0.2\n");
        for (first, second) in &self.merges {
            content.push_str(&format!("{first} {second}\n"));
        }
        std::fs::write(path, content)?;
        Ok(())
    }
    
    pub fn add_merge(&mut self, first: String, second: String) {
        self.merges.push((first, second));
    }
    
    pub fn len(&self) -> usize {
        self.merges.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.merges.is_empty()
    }
}

impl Default for BpeMerges {
    fn default() -> Self {
        Self::new()
    }
}