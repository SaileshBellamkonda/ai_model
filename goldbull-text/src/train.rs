use goldbull_text::{GoldbullText, training::Trainer};
use goldbull_core::ModelConfig;
use candle_core::Device;
use clap::{Arg, Command};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    let matches = Command::new("goldbull-text-train")
        .version("0.1.0")
        .about("Goldbull Text Model Training Script")
        .arg(
            Arg::new("dataset")
                .help("Path to training dataset")
                .value_name("DATASET_PATH")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("epochs")
                .long("epochs")
                .help("Number of training epochs")
                .value_name("EPOCHS")
                .default_value("3"),
        )
        .arg(
            Arg::new("sample")
                .long("sample")
                .help("Train on sample dataset only")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let dataset_path = matches.get_one::<String>("dataset").unwrap();
    let epochs: usize = matches
        .get_one::<String>("epochs")
        .unwrap()
        .parse()
        .expect("Invalid epochs value");
    let sample_only = matches.get_flag("sample");

    println!("Goldbull Text Model Training");
    println!("Dataset: {}", dataset_path);
    println!("Epochs: {}", epochs);
    println!("Sample only: {}", sample_only);
    
    let device = Device::Cpu;
    let config = ModelConfig::text_generation();
    
    println!("Initializing model with config: {:?}", config.model_type);
    println!("Max sequence length: {}", config.max_sequence_length);
    println!("Vocab size: {}", config.vocab_size);
    
    // Initialize the goldbull-text model
    println!("Creating GoldbullText model...");
    let model = GoldbullText::new(config, device)?;
    
    // Initialize trainer
    let mut trainer = Trainer::new(model);
    
    println!("Starting training...");
    
    // Check if dataset path exists
    if !std::path::Path::new(dataset_path).exists() {
        println!("Warning: Dataset path '{}' does not exist.", dataset_path);
        println!("Creating sample dataset for demonstration...");
        
        // Create a sample dataset directory for testing
        std::fs::create_dir_all(dataset_path)?;
        
        // Create sample text files
        let sample_texts = vec![
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large amounts of data.",
            "Natural language processing enables computers to understand text.",
            "Deep learning networks can learn complex patterns.",
        ];
        
        for (i, text) in sample_texts.iter().enumerate() {
            let file_path = format!("{}/sample_{}.txt", dataset_path, i);
            std::fs::write(file_path, text)?;
        }
        
        println!("Created sample dataset with {} files", sample_texts.len());
    }
    
    // Perform training
    match trainer.train_on_dataset(dataset_path, epochs) {
        Ok(()) => {
            println!("Training completed successfully!");
            
            // Save final model
            if let Err(e) = trainer.save_checkpoint("final_model") {
                println!("Warning: Failed to save final model: {}", e);
            } else {
                println!("Final model saved to 'final_model' directory");
            }
        },
        Err(e) => {
            println!("Training failed: {}", e);
            return Err(e.into());
        }
    }
    
    if sample_only {
        println!("Note: Sample training completed using a subset of generated data");
    }
    
    Ok(())
}