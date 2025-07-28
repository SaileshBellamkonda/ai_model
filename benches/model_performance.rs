use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ai_model::{AIModel, ModelConfig, TaskType};

fn create_test_model() -> AIModel {
    let config = ModelConfig {
        max_memory_mb: 1024,
        hidden_size: 256,
        num_layers: 2,
        vocab_size: 8000,
        ..Default::default()
    };
    
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(AIModel::new(config))
        .unwrap()
}

fn bench_text_generation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let model = create_test_model();
    
    c.bench_function("text_generation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let task = TaskType::TextGeneration {
                    max_tokens: black_box(20),
                    temperature: black_box(0.7),
                };
                
                model.execute_task(task, black_box("Hello world")).await
            })
        })
    });
}

fn bench_code_completion(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let model = create_test_model();
    
    c.bench_function("code_completion", |b| {
        b.iter(|| {
            rt.block_on(async {
                let task = TaskType::CodeCompletion {
                    language: black_box("python".to_string()),
                    context_lines: black_box(5),
                };
                
                model.execute_task(task, black_box("def hello():")).await
            })
        })
    });
}

fn bench_question_answering(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let model = create_test_model();
    
    c.bench_function("question_answering", |b| {
        b.iter(|| {
            rt.block_on(async {
                let task = TaskType::QuestionAnswer {
                    context: black_box(Some("The sky is blue.".to_string())),
                };
                
                model.execute_task(task, black_box("What color is the sky?")).await
            })
        })
    });
}

fn bench_summarization(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let model = create_test_model();
    
    c.bench_function("summarization", |b| {
        b.iter(|| {
            rt.block_on(async {
                let task = TaskType::Summarization {
                    max_length: black_box(30),
                    style: black_box("brief".to_string()),
                };
                
                let text = black_box("This is a long text that needs to be summarized into a shorter version.");
                model.execute_task(task, text).await
            })
        })
    });
}

fn bench_model_initialization(c: &mut Criterion) {
    c.bench_function("model_initialization", |b| {
        b.iter(|| {
            let config = ModelConfig {
                max_memory_mb: black_box(512),
                hidden_size: black_box(128),
                num_layers: black_box(2),
                vocab_size: black_box(4000),
                ..Default::default()
            };
            
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(AIModel::new(config))
        })
    });
}

criterion_group!(
    benches,
    bench_text_generation,
    bench_code_completion,
    bench_question_answering,
    bench_summarization,
    bench_model_initialization
);
criterion_main!(benches);