use goldbull_text::GoldbullText;
use goldbull_core::ModelConfig;
use candle_core::Device;
use clap::{Arg, Command};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    let matches = Command::new("goldbull-text-cli")
        .version("0.1.0")
        .about("Goldbull Text Generation Model CLI")
        .arg(
            Arg::new("prompt")
                .help("Text prompt for generation")
                .value_name("PROMPT")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("max-length")
                .long("max-length")
                .help("Maximum length of generated text")
                .value_name("LENGTH")
                .default_value("100"),
        )
        .get_matches();

    let prompt = matches.get_one::<String>("prompt").unwrap();
    let max_length: usize = matches
        .get_one::<String>("max-length")
        .unwrap()
        .parse()
        .expect("Invalid max-length value");

    println!("Initializing Goldbull Text Generation Model...");
    
    let device = Device::Cpu;
    let config = ModelConfig::text_generation();
    
    let model = GoldbullText::new(config, device)?;
    
    println!("Model memory usage: {} bytes", model.get_memory_usage());
    println!("Within memory limit: {}", model.is_within_memory_limit());
    
    println!("\nGenerating text for prompt: \"{}\"", prompt);
    println!("Max length: {} tokens\n", max_length);
    
    let generated_text = model.generate_text(prompt, max_length)?;
    
    println!("Generated text:");
    println!("{}", generated_text);
    
    Ok(())
}