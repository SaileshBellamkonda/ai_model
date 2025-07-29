use clap::{Arg, Command};
use goldbull_sage::{new_qa_model, answer_question, QARequest, QuestionType};
use anyhow::Result;
use candle_core::Device;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let matches = Command::new("goldbull-sage-cli")
        .version("1.0.0")
        .author("Goldbull AI Team")
        .about("Goldbull Sage - Question Answering AI Model CLI")
        .arg(
            Arg::new("question")
                .short('q')
                .long("question")
                .value_name("QUESTION")
                .help("The question to answer")
                .required(true),
        )
        .arg(
            Arg::new("context")
                .short('c')
                .long("context")
                .value_name("CONTEXT")
                .help("Context to use for answering the question"),
        )
        .arg(
            Arg::new("type")
                .short('t')
                .long("type")
                .value_name("TYPE")
                .help("Question type (factual, analytical, yesno, definition, procedural)")
                .default_value("factual"),
        )
        .arg(
            Arg::new("max-length")
                .short('m')
                .long("max-length")
                .value_name("LENGTH")
                .help("Maximum answer length")
                .default_value("100"),
        )
        .arg(
            Arg::new("temperature")
                .long("temperature")
                .value_name("TEMP")
                .help("Generation temperature (0.0-1.0)")
                .default_value("0.1"),
        )
        .arg(
            Arg::new("device")
                .short('d')
                .long("device")
                .value_name("DEVICE")
                .help("Device to use (cpu, gpu)")
                .default_value("cpu"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FORMAT")
                .help("Output format (text, json)")
                .default_value("text"),
        )
        .get_matches();

    // Parse arguments
    let question = matches.get_one::<String>("question").unwrap();
    let context = matches.get_one::<String>("context").map(|s| s.clone());
    let question_type = parse_question_type(matches.get_one::<String>("type").unwrap())?;
    let max_length: usize = matches.get_one::<String>("max-length").unwrap().parse()?;
    let temperature: f64 = matches.get_one::<String>("temperature").unwrap().parse()?;
    let device = parse_device(matches.get_one::<String>("device").unwrap())?;
    let output_format = matches.get_one::<String>("output").unwrap();

    // Initialize model
    tracing::info!("Loading Goldbull Sage model...");
    let model = new_qa_model(device)?;

    // Create request
    let request = QARequest {
        question: question.clone(),
        context,
        question_type,
        max_answer_length: max_length,
        temperature,
        use_context: true,
        metadata: std::collections::HashMap::new(),
    };

    // Generate answer
    tracing::info!("Answering question: {}", question);
    let start_time = std::time::Instant::now();
    let response = answer_question(&model, request).await?;
    let generation_time = start_time.elapsed();

    // Output result
    match output_format.as_str() {
        "json" => {
            let output_json = serde_json::json!({
                "question": question,
                "answer": response.answer,
                "confidence": response.confidence,
                "question_type": response.question_type,
                "sources": response.sources,
                "generation_time_ms": generation_time.as_millis(),
                "metadata": response.metadata
            });
            println!("{}", serde_json::to_string_pretty(&output_json)?);
        }
        _ => {
            println!("Question: {}", question);
            println!("Answer: {}", response.answer);
            println!("Confidence: {:.2}%", response.confidence * 100.0);
            println!("Type: {:?}", response.question_type);
            println!("Generation time: {}ms", generation_time.as_millis());
            
            if !response.sources.is_empty() {
                println!("\nSources:");
                for (i, source) in response.sources.iter().enumerate() {
                    println!("  {}. {} (relevance: {:.2})", i + 1, source.title, source.relevance_score);
                }
            }
        }
    }

    Ok(())
}

fn parse_question_type(type_str: &str) -> Result<QuestionType> {
    match type_str.to_lowercase().as_str() {
        "factual" => Ok(QuestionType::Factual),
        "analytical" => Ok(QuestionType::Analytical),
        "yesno" | "yes-no" => Ok(QuestionType::YesNo),
        "definition" => Ok(QuestionType::Definition),
        "procedural" => Ok(QuestionType::Procedural),
        "multiple-choice" | "multiplechoice" => Ok(QuestionType::MultipleChoice),
        "open-ended" | "openended" => Ok(QuestionType::OpenEnded),
        _ => Err(anyhow::anyhow!("Unknown question type: {}", type_str)),
    }
}

fn parse_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "gpu" | "cuda" => Ok(Device::new_cuda(0)?),
        _ => Err(anyhow::anyhow!("Unknown device: {}", device_str)),
    }
}