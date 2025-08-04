use clap::{Arg, Command, ArgMatches};
use goldbull_sage::training::{Trainer, TrainingConfig, QADataset, QASample, EvaluationMetrics};
use goldbull_sage::{new_qa_model, QuestionType};
use goldbull_tokenizer::BpeTokenizer;
use anyhow::Result;
use candle_core::Device;
use std::path::Path;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let matches = Command::new("goldbull-sage-train")
        .version("1.0.0")
        .author("Goldbull AI Team")
        .about("Training tool for Goldbull Sage Question Answering model")
        .subcommand(
            Command::new("sample")
                .about("Train on a sample subset of data")
                .arg(
                    Arg::new("data")
                        .short('d')
                        .long("data")
                        .value_name("PATH")
                        .help("Path to training data")
                        .required(true),
                )
                .arg(
                    Arg::new("epochs")
                        .short('e')
                        .long("epochs")
                        .value_name("NUM")
                        .help("Number of training epochs")
                        .default_value("3"),
                )
                .arg(
                    Arg::new("batch-size")
                        .short('b')
                        .long("batch-size")
                        .value_name("SIZE")
                        .help("Training batch size")
                        .default_value("4"),
                )
        )
        .subcommand(
            Command::new("full")
                .about("Train on full dataset")
                .arg(
                    Arg::new("data")
                        .short('d')
                        .long("data")
                        .value_name("PATH")
                        .help("Path to training data")
                        .required(true),
                )
                .arg(
                    Arg::new("epochs")
                        .short('e')
                        .long("epochs")
                        .value_name("NUM")
                        .help("Number of training epochs")
                        .default_value("10"),
                )
                .arg(
                    Arg::new("batch-size")
                        .short('b')
                        .long("batch-size")
                        .value_name("SIZE")
                        .help("Training batch size")
                        .default_value("8"),
                )
                .arg(
                    Arg::new("learning-rate")
                        .short('l')
                        .long("learning-rate")
                        .value_name("RATE")
                        .help("Learning rate")
                        .default_value("1e-4"),
                )
        )
        .subcommand(
            Command::new("evaluate")
                .about("Evaluate trained model")
                .arg(
                    Arg::new("model")
                        .short('m')
                        .long("model")
                        .value_name("PATH")
                        .help("Path to trained model")
                        .required(true),
                )
                .arg(
                    Arg::new("test-data")
                        .short('t')
                        .long("test-data")
                        .value_name("PATH")
                        .help("Path to test data")
                        .required(true),
                )
        )
        .arg(
            Arg::new("device")
                .short('D')
                .long("device")
                .value_name("DEVICE")
                .help("Device to use (cpu, gpu)")
                .default_value("cpu"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("DIR")
                .help("Output directory for models and logs")
                .default_value("./output"),
        )
        .get_matches();

    let device = parse_device(matches.get_one::<String>("device").unwrap())?;
    let output_dir = matches.get_one::<String>("output").unwrap();

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    match matches.subcommand() {
        Some(("sample", sub_matches)) => {
            train_sample(sub_matches, device, output_dir).await?;
        }
        Some(("full", sub_matches)) => {
            train_full(sub_matches, device, output_dir).await?;
        }
        Some(("evaluate", sub_matches)) => {
            evaluate_model(sub_matches, device).await?;
        }
        _ => {
            eprintln!("No subcommand provided. Use --help for usage information.");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn train_sample(matches: &ArgMatches, device: Device, output_dir: &str) -> Result<()> {
    tracing::info!("Starting sample training for Goldbull Sage");

    let data_path = matches.get_one::<String>("data").unwrap();
    let epochs: usize = matches.get_one::<String>("epochs").unwrap().parse()?;
    let batch_size: usize = matches.get_one::<String>("batch-size").unwrap().parse()?;

    // Create sample training configuration
    let config = TrainingConfig {
        epochs,
        batch_size,
        learning_rate: 1e-3, // Higher learning rate for sample training
        weight_decay: 0.01,
        gradient_clip_norm: 1.0,
        validation_split: 0.2,
        early_stopping_patience: 2,
        checkpoint_interval: 1,
        max_sequence_length: 256, // Shorter for sample training
    };

    // Load model and dataset
    let model = new_qa_model(device)?;
    let tokenizer = BpeTokenizer::from_pretrained()?;
    let dataset = create_sample_dataset(data_path, tokenizer)?;

    // Create trainer and train
    let mut trainer = Trainer::new(model, config, dataset);
    trainer.train().await?;

    tracing::info!("Sample training completed. Model saved to {}", output_dir);
    Ok(())
}

async fn train_full(matches: &ArgMatches, device: Device, output_dir: &str) -> Result<()> {
    tracing::info!("Starting full training for Goldbull Sage");

    let data_path = matches.get_one::<String>("data").unwrap();
    let epochs: usize = matches.get_one::<String>("epochs").unwrap().parse()?;
    let batch_size: usize = matches.get_one::<String>("batch-size").unwrap().parse()?;
    let learning_rate: f64 = matches.get_one::<String>("learning-rate").unwrap().parse()?;

    // Create full training configuration
    let config = TrainingConfig {
        epochs,
        batch_size,
        learning_rate,
        weight_decay: 0.01,
        gradient_clip_norm: 1.0,
        validation_split: 0.1,
        early_stopping_patience: 3,
        checkpoint_interval: 1,
        max_sequence_length: 512,
    };

    // Load model and dataset
    let model = new_qa_model(device)?;
    let tokenizer = BpeTokenizer::from_pretrained()?;
    let dataset = load_full_dataset(data_path, tokenizer)?;

    // Create trainer and train
    let mut trainer = Trainer::new(model, config, dataset);
    trainer.train().await?;

    tracing::info!("Full training completed. Model saved to {}", output_dir);
    Ok(())
}

async fn evaluate_model(matches: &ArgMatches, device: Device) -> Result<()> {
    tracing::info!("Evaluating Goldbull Sage model");

    let model_path = matches.get_one::<String>("model").unwrap();
    let test_data_path = matches.get_one::<String>("test-data").unwrap();

    // Load model (placeholder - would load from checkpoint)
    let model = new_qa_model(device)?;
    
    // Load test data
    let tokenizer = BpeTokenizer::from_pretrained()?;
    let test_samples = load_test_samples(test_data_path)?;

    // Create trainer for evaluation
    let dataset = QADataset::new(tokenizer);
    let config = TrainingConfig::default();
    let trainer = Trainer::new(model, config, dataset);

    // Evaluate
    let metrics = trainer.evaluate(&test_samples).await?;

    // Print results
    println!("Evaluation Results:");
    println!("==================");
    println!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!("Average Confidence: {:.2}%", metrics.avg_confidence * 100.0);
    println!("Total Samples: {}", metrics.total_samples);
    println!("Correct Answers: {}", metrics.correct_answers);

    Ok(())
}

fn create_sample_dataset(data_path: &str, tokenizer: BpeTokenizer) -> Result<QADataset> {
    let path = Path::new(data_path);
    
    if path.extension().and_then(|s| s.to_str()) == Some("json") {
        // Load from JSON
        let mut dataset = QADataset::from_json_file(path, tokenizer)?;
        
        // Take only first 100 samples for sample training
        dataset.take_samples(100);
        
        Ok(dataset)
    } else {
        // Try to load from mOSCAR format
        QADataset::from_moscar(path, tokenizer)
    }
}

fn load_full_dataset(data_path: &str, tokenizer: BpeTokenizer) -> Result<QADataset> {
    let path = Path::new(data_path);
    
    if path.extension().and_then(|s| s.to_str()) == Some("json") {
        QADataset::from_json_file(path, tokenizer)
    } else {
        QADataset::from_moscar(path, tokenizer)
    }
}

fn load_test_samples(test_data_path: &str) -> Result<Vec<QASample>> {
    let content = std::fs::read_to_string(test_data_path)?;
    let samples: Vec<QASample> = serde_json::from_str(&content)?;
    Ok(samples)
}

fn parse_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "gpu" | "cuda" => Ok(Device::new_cuda(0)?),
        _ => Err(anyhow::anyhow!("Unknown device: {}", device_str)),
    }
}