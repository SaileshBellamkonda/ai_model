# Goldbull AI Model Suite - Usage Guide

## Quick Start

### Installation

Add the Goldbull AI models to your `Cargo.toml`:

```toml
[dependencies]
goldbull-text = "0.1.0"
goldbull-code = "0.1.0"
goldbull-vision = "0.1.0"
goldbull-core = "0.1.0"
```

### Basic Usage

```rust
use goldbull_text::GoldbullText;
use goldbull_core::ModelConfig;
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    // Initialize the model
    let device = Device::Cpu;
    let config = ModelConfig::default();
    let model = GoldbullText::new(config, device)?;
    
    // Generate text
    let prompt = "The future of AI is";
    let generated = model.generate_text(prompt, 50)?;
    println!("Generated: {}", generated);
    
    Ok(())
}
```

## Model Components

### 1. Text Generation (goldbull-text)

#### Basic Text Generation
```rust
use goldbull_text::GoldbullText;

// Load a pre-trained model
let model = GoldbullText::from_pretrained("path/to/model")?;

// Generate text with different parameters
let short_text = model.generate_text("Once upon a time", 20)?;
let long_text = model.generate_text("Explain quantum computing", 200)?;

// Check memory usage
println!("Memory usage: {} bytes", model.get_memory_usage());
println!("Within limits: {}", model.is_within_memory_limit());
```

#### Advanced Generation Options
```rust
use goldbull_text::{GoldbullText, GenerationConfig};

let model = GoldbullText::from_pretrained("path/to/model")?;

// Custom generation configuration
let config = GenerationConfig {
    max_length: 100,
    temperature: 0.8,
    top_p: 0.9,
    repetition_penalty: 1.1,
    stop_tokens: vec![".", "!", "?"],
};

let generated = model.generate_with_config("Write a story about", &config)?;
```

#### Conversation Mode
```rust
use goldbull_text::GoldbullText;

let model = GoldbullText::from_pretrained("path/to/model")?;

// Maintain conversation context
let mut conversation = Vec::new();
conversation.push("User: What is machine learning?".to_string());

let response = model.generate_text(&conversation.join("\n"), 100)?;
conversation.push(format!("Assistant: {}", response));

// Continue conversation
conversation.push("User: Can you give me an example?".to_string());
let next_response = model.generate_text(&conversation.join("\n"), 100)?;
```

### 2. Code Generation (goldbull-code)

#### Code Completion
```rust
use goldbull_code::GoldbullCode;

let model = GoldbullCode::from_pretrained("path/to/model")?;

// Complete Python function
let code_context = r#"
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    "#;

let completion = model.complete_code(code_context, "python", 50)?;
println!("Completed code:\n{}", completion);
```

#### Multi-Language Support
```rust
use goldbull_code::{GoldbullCode, Language};

let model = GoldbullCode::from_pretrained("path/to/model")?;

// Rust code completion
let rust_code = "fn main() {\n    let x = ";
let rust_completion = model.complete_code(rust_code, "rust", 30)?;

// JavaScript code completion  
let js_code = "function calculateSum(arr) {\n    return arr.";
let js_completion = model.complete_code(js_code, "javascript", 30)?;

// Java code completion
let java_code = "public class Calculator {\n    public int add(int a, int b) {";
let java_completion = model.complete_code(java_code, "java", 50)?;
```

#### Code Analysis and Documentation
```rust
use goldbull_code::GoldbullCode;

let model = GoldbullCode::from_pretrained("path/to/model")?;

// Generate documentation
let function_code = r#"
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
"#;

let documentation = model.generate_documentation(function_code, "python")?;
println!("Generated docs:\n{}", documentation);

// Code explanation
let explanation = model.explain_code(function_code, "python")?;
println!("Explanation:\n{}", explanation);
```

### 3. Computer Vision (goldbull-vision)

#### Image Classification
```rust
use goldbull_vision::GoldbullVision;
use image::open;

let model = GoldbullVision::from_pretrained("path/to/model")?;

// Load and classify image
let image = open("path/to/image.jpg")?;
let predictions = model.classify_image(&image)?;

for (class, confidence) in predictions.iter().take(5) {
    println!("{}: {:.2}%", class, confidence * 100.0);
}
```

#### Object Detection
```rust
use goldbull_vision::{GoldbullVision, BoundingBox};

let model = GoldbullVision::from_pretrained("path/to/model")?;

let image = open("path/to/image.jpg")?;
let detections = model.detect_objects(&image)?;

for detection in detections {
    println!("Object: {} (confidence: {:.2}%)", 
             detection.class, detection.confidence * 100.0);
    println!("Location: ({}, {}) to ({}, {})",
             detection.bbox.x, detection.bbox.y,
             detection.bbox.x + detection.bbox.width,
             detection.bbox.y + detection.bbox.height);
}
```

#### Optical Character Recognition (OCR)
```rust
use goldbull_vision::GoldbullVision;

let model = GoldbullVision::from_pretrained("path/to/model")?;

let image = open("document.png")?;
let extracted_text = model.extract_text(&image)?;
println!("Extracted text:\n{}", extracted_text);

// OCR with bounding boxes
let text_regions = model.extract_text_with_regions(&image)?;
for region in text_regions {
    println!("Text: '{}' at ({}, {})", 
             region.text, region.bbox.x, region.bbox.y);
}
```

### 4. Question Answering (goldbull-sage)

#### Document-based Q&A
```rust
use goldbull_sage::GoldbullSage;

let model = GoldbullSage::from_pretrained("path/to/model")?;

// Load context document
let document = std::fs::read_to_string("knowledge_base.txt")?;

// Ask questions
let question = "What is the main topic of this document?";
let answer = model.answer_question(&question, &document)?;
println!("Answer: {}", answer.text);
println!("Confidence: {:.2}%", answer.confidence * 100.0);
```

#### Multi-document Q&A
```rust
use goldbull_sage::{GoldbullSage, DocumentSet};

let model = GoldbullSage::from_pretrained("path/to/model")?;

// Create document set
let mut doc_set = DocumentSet::new();
doc_set.add_document("doc1.txt", &std::fs::read_to_string("doc1.txt")?)?;
doc_set.add_document("doc2.txt", &std::fs::read_to_string("doc2.txt")?)?;
doc_set.add_document("doc3.txt", &std::fs::read_to_string("doc3.txt")?)?;

// Build search index
doc_set.build_index()?;

// Answer question across all documents
let question = "How do solar panels work?";
let answer = model.answer_from_documents(&question, &doc_set)?;

println!("Answer: {}", answer.text);
println!("Source: {}", answer.source_document);
println!("Relevance: {:.2}%", answer.relevance * 100.0);
```

### 5. Summarization (goldbull-brief)

#### Text Summarization
```rust
use goldbull_brief::GoldbullBrief;

let model = GoldbullBrief::from_pretrained("path/to/model")?;

// Summarize long text
let long_text = std::fs::read_to_string("article.txt")?;
let summary = model.summarize(&long_text, 100)?; // Max 100 words
println!("Summary:\n{}", summary);

// Extractive vs Abstractive summarization
let extractive = model.extractive_summary(&long_text, 3)?; // 3 sentences
let abstractive = model.abstractive_summary(&long_text, 50)?; // 50 words
```

#### Multi-document Summarization
```rust
use goldbull_brief::GoldbullBrief;

let model = GoldbullBrief::from_pretrained("path/to/model")?;

let documents = vec![
    std::fs::read_to_string("doc1.txt")?,
    std::fs::read_to_string("doc2.txt")?,
    std::fs::read_to_string("doc3.txt")?,
];

let combined_summary = model.summarize_documents(&documents, 200)?;
println!("Combined summary:\n{}", combined_summary);
```

### 6. Embeddings (goldbull-embedding)

#### Text Embeddings
```rust
use goldbull_embedding::GoldbullEmbedding;

let model = GoldbullEmbedding::from_pretrained("path/to/model")?;

// Generate embeddings
let text = "Machine learning is a subset of artificial intelligence.";
let embedding = model.encode_text(text)?;
println!("Embedding dimension: {}", embedding.len());

// Batch encoding
let texts = vec![
    "First document about AI",
    "Second document about ML", 
    "Third document about robotics"
];
let embeddings = model.encode_batch(&texts)?;
```

#### Similarity Search
```rust
use goldbull_embedding::{GoldbullEmbedding, EmbeddingIndex};

let model = GoldbullEmbedding::from_pretrained("path/to/model")?;

// Build searchable index
let mut index = EmbeddingIndex::new();
let documents = vec![
    "Document about machine learning algorithms",
    "Article on deep neural networks",
    "Paper on computer vision techniques",
    "Study on natural language processing"
];

for (id, doc) in documents.iter().enumerate() {
    let embedding = model.encode_text(doc)?;
    index.add_embedding(id, embedding);
}

// Search for similar documents
let query = "artificial intelligence and neural networks";
let query_embedding = model.encode_text(query)?;
let results = index.search(&query_embedding, 3)?; // Top 3 results

for (doc_id, similarity) in results {
    println!("Document {}: {} (similarity: {:.3})", 
             doc_id, documents[doc_id], similarity);
}
```

### 7. Multimodal AI (goldbull-multimodel)

#### Text and Image Processing
```rust
use goldbull_multimodel::GoldbullMultimodel;
use image::open;

let model = GoldbullMultimodel::from_pretrained("path/to/model")?;

// Image captioning
let image = open("photo.jpg")?;
let caption = model.generate_caption(&image)?;
println!("Caption: {}", caption);

// Visual question answering
let question = "What color is the car in the image?";
let answer = model.answer_visual_question(&image, question)?;
println!("Answer: {}", answer);

// Text-to-image search
let text_query = "sunset over mountains";
let image_candidates = vec![
    open("image1.jpg")?,
    open("image2.jpg")?,
    open("image3.jpg")?,
];
let best_match = model.find_best_image_match(text_query, &image_candidates)?;
println!("Best matching image index: {}", best_match);
```

## Command Line Interface

### Text Generation CLI
```bash
# Generate text from prompt
cargo run --bin goldbull-text -- generate "The future of AI" --max-length 100

# Interactive mode
cargo run --bin goldbull-text -- interactive

# Batch processing
cargo run --bin goldbull-text -- batch --input prompts.txt --output results.txt
```

### Code Completion CLI
```bash
# Complete code file
cargo run --bin goldbull-code -- complete --file main.py --language python

# Generate documentation
cargo run --bin goldbull-code -- document --file utils.rs --language rust

# Code analysis
cargo run --bin goldbull-code -- analyze --directory src/ --language rust
```

### Vision CLI
```bash
# Classify image
cargo run --bin goldbull-vision -- classify --image photo.jpg

# Object detection
cargo run --bin goldbull-vision -- detect --image scene.jpg --output detection.json

# OCR processing
cargo run --bin goldbull-vision -- ocr --image document.png --output text.txt
```

## REST API Usage

### Starting the API Server
```bash
cargo run --bin goldbull-api -- --port 8080 --models text,code,vision
```

### API Endpoints

#### Text Generation
```bash
curl -X POST http://localhost:8080/api/text/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The benefits of renewable energy",
    "max_length": 100,
    "temperature": 0.8
  }'
```

#### Code Completion
```bash
curl -X POST http://localhost:8080/api/code/complete \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def fibonacci(n):\n    if n <= 1:",
    "language": "python",
    "max_length": 50
  }'
```

#### Image Classification
```bash
curl -X POST http://localhost:8080/api/vision/classify \
  -F "image=@photo.jpg"
```

## Performance Optimization

### Memory Usage Monitoring
```rust
use goldbull_core::utils::MemoryMonitor;

let monitor = MemoryMonitor::new();
monitor.start_monitoring();

// Run your model operations
let model = GoldbullText::from_pretrained("path/to/model")?;
let result = model.generate_text("Hello", 50)?;

// Check memory statistics
let stats = monitor.get_statistics();
println!("Peak memory usage: {} MB", stats.peak_memory_mb);
println!("Average memory usage: {} MB", stats.average_memory_mb);
```

### CPU Optimization
```rust
use goldbull_core::utils::CpuOptimizer;

// Optimize for current hardware
let optimizer = CpuOptimizer::new();
optimizer.configure_for_hardware();

// Use optimized thread pool
let thread_count = optimizer.optimal_thread_count();
let model = GoldbullText::new_with_threads(config, device, thread_count)?;
```

### Batch Processing
```rust
use goldbull_text::GoldbullText;

let model = GoldbullText::from_pretrained("path/to/model")?;

// Process multiple prompts efficiently
let prompts = vec![
    "First prompt",
    "Second prompt", 
    "Third prompt"
];

let results = model.generate_batch(&prompts, 50)?;
for (prompt, result) in prompts.iter().zip(results.iter()) {
    println!("Prompt: {}\nResult: {}\n", prompt, result);
}
```

## Error Handling

### Common Error Patterns
```rust
use goldbull_core::{GoldbullError, Result};

fn handle_model_operations() -> Result<()> {
    match GoldbullText::from_pretrained("path/to/model") {
        Ok(model) => {
            match model.generate_text("Hello", 50) {
                Ok(text) => println!("Generated: {}", text),
                Err(GoldbullError::InsufficientMemory) => {
                    println!("Not enough memory for generation");
                }
                Err(GoldbullError::ModelNotFound) => {
                    println!("Model file not found");
                }
                Err(e) => println!("Other error: {}", e),
            }
        }
        Err(e) => println!("Failed to load model: {}", e),
    }
    Ok(())
}
```

### Resource Management
```rust
use goldbull_core::utils::ResourceManager;

// Set resource limits
let mut manager = ResourceManager::new();
manager.set_memory_limit(1_500_000_000); // 1.5 GB
manager.set_cpu_cores(4);
manager.set_timeout_seconds(30);

// Use with model operations
let model = GoldbullText::new_with_limits(config, device, manager)?;
```

## Integration Examples

### Web Service Integration
```rust
use warp::Filter;
use goldbull_text::GoldbullText;

#[tokio::main]
async fn main() {
    let model = Arc::new(GoldbullText::from_pretrained("path/to/model").unwrap());
    
    let generate = warp::path("generate")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || model.clone()))
        .and_then(handle_generate);
    
    warp::serve(generate)
        .run(([127, 0, 0, 1], 3030))
        .await;
}

async fn handle_generate(
    request: GenerateRequest,
    model: Arc<GoldbullText>
) -> Result<impl warp::Reply, warp::Rejection> {
    let result = model.generate_text(&request.prompt, request.max_length)
        .map_err(|_| warp::reject::custom(GenerationError))?;
    Ok(warp::reply::json(&GenerateResponse { text: result }))
}
```

### Database Integration
```rust
use sqlx::PgPool;
use goldbull_embedding::GoldbullEmbedding;

async fn store_document_with_embedding(
    pool: &PgPool,
    model: &GoldbullEmbedding,
    document: &str
) -> Result<i32> {
    let embedding = model.encode_text(document)?;
    let embedding_bytes: Vec<u8> = embedding.iter()
        .flat_map(|f| f.to_le_bytes().to_vec())
        .collect();
    
    let id = sqlx::query!(
        "INSERT INTO documents (content, embedding) VALUES ($1, $2) RETURNING id",
        document,
        embedding_bytes
    )
    .fetch_one(pool)
    .await?
    .id;
    
    Ok(id)
}
```

## Best Practices

### Model Lifecycle Management
1. **Load models once**: Reuse model instances across requests
2. **Monitor memory**: Regularly check memory usage and limits
3. **Graceful degradation**: Handle resource constraints appropriately
4. **Cleanup resources**: Properly dispose of models when done

### Performance Tips
1. **Batch operations**: Process multiple requests together when possible
2. **Use appropriate model sizes**: Choose model variants based on requirements
3. **Cache results**: Store frequently requested outputs
4. **Profile regularly**: Monitor performance characteristics

### Security Considerations
1. **Input validation**: Sanitize all user inputs
2. **Output filtering**: Check generated content for appropriateness
3. **Resource limits**: Prevent resource exhaustion attacks
4. **Model integrity**: Verify model files before loading