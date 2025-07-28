use ai_model::{AIModel, ModelConfig, TaskType};
use ai_model::tools::{CalculatorTool, HttpRequestTool, FileOperationsTool};

#[tokio::test]
async fn test_model_initialization() {
    let config = ModelConfig {
        max_memory_mb: 512, // Smaller for tests
        hidden_size: 128,
        num_layers: 2,
        vocab_size: 1000,
        ..Default::default()
    };
    
    let model = AIModel::new(config).await;
    assert!(model.is_ok());
    
    let model = model.unwrap();
    assert_eq!(model.get_config().max_memory_mb, 512);
    assert_eq!(model.get_config().hidden_size, 128);
}

#[tokio::test]
async fn test_text_generation() {
    let config = ModelConfig {
        max_memory_mb: 512,
        hidden_size: 64,
        num_layers: 1,
        vocab_size: 500,
        ..Default::default()
    };
    
    let model = AIModel::new(config).await.unwrap();
    
    let task = TaskType::TextGeneration {
        max_tokens: 10,
        temperature: 0.7,
    };
    
    let result = model.execute_task(task, "Hello").await;
    assert!(result.is_ok());
    
    let result = result.unwrap();
    assert!(!result.text_output().is_empty());
}

#[tokio::test]
async fn test_code_completion() {
    let config = ModelConfig {
        max_memory_mb: 512,
        hidden_size: 64,
        num_layers: 1,
        vocab_size: 500,
        ..Default::default()
    };
    
    let model = AIModel::new(config).await.unwrap();
    
    let task = TaskType::CodeCompletion {
        language: "python".to_string(),
        context_lines: 5,
    };
    
    let result = model.execute_task(task, "def hello():").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_question_answering() {
    let config = ModelConfig {
        max_memory_mb: 512,
        hidden_size: 64,
        num_layers: 1,
        vocab_size: 500,
        ..Default::default()
    };
    
    let model = AIModel::new(config).await.unwrap();
    
    let task = TaskType::QuestionAnswer {
        context: Some("The sky is blue.".to_string()),
    };
    
    let result = model.execute_task(task, "What color is the sky?").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_summarization() {
    let config = ModelConfig {
        max_memory_mb: 512,
        hidden_size: 64,
        num_layers: 1,
        vocab_size: 500,
        ..Default::default()
    };
    
    let model = AIModel::new(config).await.unwrap();
    
    let task = TaskType::Summarization {
        max_length: 20,
        style: "brief".to_string(),
    };
    
    let long_text = "This is a very long text that needs to be summarized. It contains multiple sentences and ideas.";
    let result = model.execute_task(task, long_text).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_visual_analysis() {
    let config = ModelConfig {
        max_memory_mb: 512,
        hidden_size: 64,
        num_layers: 1,
        vocab_size: 500,
        ..Default::default()
    };
    
    let model = AIModel::new(config).await.unwrap();
    
    let task = TaskType::VisualAnalysis {
        image_data: vec![1, 2, 3, 4], // Dummy image data
    };
    
    let result = model.execute_task(task, "A beautiful landscape").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_calculator_tool() {
    let config = ModelConfig {
        max_memory_mb: 512,
        hidden_size: 64,
        num_layers: 1,
        vocab_size: 500,
        ..Default::default()
    };
    
    let model = AIModel::new(config).await.unwrap();
    
    // Register calculator tool
    model.register_tool(
        "calculator".to_string(),
        "Test calculator".to_string(),
        Box::new(CalculatorTool),
    ).await.unwrap();
    
    // Test simple addition
    let args = serde_json::json!({"expression": "5 + 3"});
    let result = model.call_tool("calculator", args).await;
    assert!(result.is_ok());
    
    let result_value = result.unwrap();
    assert_eq!(result_value["result"], 8.0);
}

#[tokio::test]
async fn test_memory_limits() {
    let config = ModelConfig {
        max_memory_mb: 1, // Very low limit to test memory constraints
        hidden_size: 64,
        num_layers: 1,
        vocab_size: 500,
        ..Default::default()
    };
    
    // This should still work as our model is lightweight
    let model = AIModel::new(config).await;
    assert!(model.is_ok());
}

#[tokio::test]
async fn test_performance_metrics() {
    let config = ModelConfig {
        max_memory_mb: 512,
        hidden_size: 64,
        num_layers: 1,
        vocab_size: 500,
        ..Default::default()
    };
    
    let model = AIModel::new(config).await.unwrap();
    
    let task = TaskType::TextGeneration {
        max_tokens: 5,
        temperature: 0.7,
    };
    
    let _result = model.execute_task(task, "Test").await.unwrap();
    
    let metrics = model.get_metrics().await;
    assert!(metrics.inference_time_ms > 0.0);
    assert!(metrics.memory_usage_mb >= 0.0);
    assert!(metrics.throughput_tokens_per_sec >= 0.0);
}

#[tokio::test]
async fn test_configuration_validation() {
    // Test various configurations
    let configs = vec![
        ModelConfig {
            max_memory_mb: 2048,
            hidden_size: 512,
            num_layers: 6,
            vocab_size: 32000,
            ..Default::default()
        },
        ModelConfig {
            max_memory_mb: 1024,
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 16000,
            ..Default::default()
        },
        ModelConfig {
            max_memory_mb: 512,
            hidden_size: 128,
            num_layers: 2,
            vocab_size: 8000,
            ..Default::default()
        },
    ];
    
    for config in configs {
        let model = AIModel::new(config).await;
        assert!(model.is_ok(), "Failed to create model with valid configuration");
    }
}