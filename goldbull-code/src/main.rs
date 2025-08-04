use anyhow::Result;
use clap::{Parser, Subcommand};
use goldbull_code::{
    GoldbullCode, CodeGenerator, 
    completion::{CompletionRequest, CompletionEngine},
    generation::{GenerationRequest, GenerationConfig, GenerationContext, CompletionMode, StylePreferences},
    syntax::{LanguageType, SyntaxAnalyzer},
};
use goldbull_core::ModelConfig;
use candle_core::Device;
use std::{fs, io::{self, Write}, time::Instant};
use tracing::{info, error};
use tracing_subscriber;

/// Goldbull Code Completion CLI
/// Production-ready code completion and generation tool
#[derive(Parser)]
#[command(
    name = "goldbull-code-cli",
    about = "Goldbull Code Completion - AI-powered code completion and generation",
    version = "1.0.0"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Model weights file path
    #[arg(long, global = true)]
    model_path: Option<String>,
    
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
    
    /// Output format (text, json)
    #[arg(long, global = true, default_value = "text")]
    format: String,
    
    /// Device to use (cpu, gpu)
    #[arg(long, global = true, default_value = "cpu")]
    device: String,
    
    /// Maximum memory usage in MB
    #[arg(long, global = true)]
    max_memory: Option<usize>,
}

#[derive(Subcommand)]
enum Commands {
    /// Complete code based on context
    Complete {
        /// Code prefix to complete
        #[arg(short, long)]
        prefix: String,
        
        /// Optional code suffix
        #[arg(short, long)]
        suffix: Option<String>,
        
        /// Programming language
        #[arg(short, long, default_value = "auto")]
        language: String,
        
        /// Maximum tokens to generate
        #[arg(long, default_value = "100")]
        max_tokens: usize,
        
        /// Sampling temperature
        #[arg(long, default_value = "0.2")]
        temperature: f64,
        
        /// Top-p sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f64,
        
        /// Top-k sampling limit
        #[arg(long, default_value = "50")]
        top_k: usize,
        
        /// Include documentation
        #[arg(long)]
        include_docs: bool,
        
        /// Context files (comma-separated paths)
        #[arg(long)]
        context_files: Option<String>,
    },
    
    /// Generate code from prompt
    Generate {
        /// Code generation prompt
        #[arg(short, long)]
        prompt: String,
        
        /// Programming language
        #[arg(short, long, default_value = "auto")]
        language: String,
        
        /// Generation mode (line, block, function, type, module)
        #[arg(short, long, default_value = "block")]
        mode: String,
        
        /// Maximum tokens to generate
        #[arg(long, default_value = "200")]
        max_tokens: usize,
        
        /// Sampling temperature
        #[arg(long, default_value = "0.3")]
        temperature: f64,
        
        /// Include documentation
        #[arg(long)]
        include_docs: bool,
        
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// Analyze code syntax and features
    Analyze {
        /// File to analyze
        #[arg(short, long)]
        file: String,
        
        /// Programming language (auto-detect if not specified)
        #[arg(short, long)]
        language: Option<String>,
        
        /// Show detailed analysis
        #[arg(short, long)]
        detailed: bool,
        
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// Interactive code completion session
    Interactive {
        /// Programming language
        #[arg(short, long, default_value = "auto")]
        language: String,
        
        /// Enable syntax highlighting
        #[arg(long)]
        syntax_highlighting: bool,
    },
    
    /// Benchmark model performance
    Benchmark {
        /// Number of completion requests
        #[arg(short, long, default_value = "100")]
        requests: usize,
        
        /// Benchmark dataset file
        #[arg(short, long)]
        dataset: Option<String>,
        
        /// Save results to file
        #[arg(short, long)]
        save_results: Option<String>,
    },
    
    /// Show model information
    Info {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("goldbull_code={}", log_level))
        .init();
    
    info!("Starting Goldbull Code Completion CLI");
    
    // Check device availability
    let device = Device::Cpu;
    info!("Using CPU for inference");
    
    match &cli.command {
        Commands::Complete { 
            prefix, suffix, language, max_tokens, temperature, top_p, top_k, include_docs, context_files 
        } => {
            let model = load_model(&cli, device).await?;
            complete_code(
                &model,
                prefix.clone(),
                suffix.clone(),
                language.clone(),
                *max_tokens,
                *temperature,
                *top_p,
                *top_k,
                *include_docs,
                context_files.clone(),
                &cli.format,
            ).await?;
        }
        
        Commands::Generate { 
            prompt, language, mode, max_tokens, temperature, include_docs, output 
        } => {
            let model = load_model(&cli, device).await?;
            generate_code(
                &model,
                prompt.clone(),
                language.clone(),
                mode.clone(),
                *max_tokens,
                *temperature,
                *include_docs,
                output.clone(),
                &cli.format,
            ).await?;
        }
        
        Commands::Analyze { file, language, detailed, output } => {
            analyze_code(
                file.clone(),
                language.clone(),
                *detailed,
                output.clone(),
                &cli.format,
            ).await?;
        }
        
        Commands::Interactive { language, syntax_highlighting } => {
            let model = load_model(&cli, device).await?;
            interactive_session(&model, language.clone(), *syntax_highlighting).await?;
        }
        
        Commands::Benchmark { requests, dataset, save_results } => {
            let model = load_model(&cli, device).await?;
            benchmark_model(&model, *requests, dataset.clone(), save_results.clone()).await?;
        }
        
        Commands::Info { detailed } => {
            let model = load_model(&cli, device).await?;
            show_model_info(&model, *detailed, &cli.format)?;
        }
    }
    
    Ok(())
}

/// Load the code completion model
async fn load_model(cli: &Cli, device: Device) -> Result<GoldbullCode> {
    let start_time = Instant::now();
    
    let model = if let Some(model_path) = &cli.model_path {
        info!("Loading model from: {}", model_path);
        let config = ModelConfig::code_completion();
        GoldbullCode::from_weights(model_path, config, device)?
    } else {
        info!("Creating new model with default configuration");
        let config = ModelConfig::code_completion();
        GoldbullCode::new(config, device)?
    };
    
    let load_time = start_time.elapsed();
    info!("Model loaded in {:.2}s", load_time.as_secs_f64());
    
    Ok(model)
}

/// Complete code based on context
async fn complete_code(
    _model: &GoldbullCode,
    prefix: String,
    suffix: Option<String>,
    language: String,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    top_k: usize,
    include_docs: bool,
    context_files: Option<String>,
    format: &str,
) -> Result<()> {
    let start_time = Instant::now();
    
    // Parse language
    let lang = parse_language(&language, &prefix)?;
    
    // Load context files if provided
    let context_file_list = if let Some(files) = context_files {
        load_context_files(&files)?
    } else {
        Vec::new()
    };
    
    // Create completion request
    let request = CompletionRequest {
        prefix,
        suffix,
        language: lang,
        max_tokens,
        temperature,
        top_p,
        top_k,
        context_files: context_file_list,
        include_docs,
        ..Default::default()
    };
    
    // Generate completion
    let mut engine = CompletionEngine::new()?;
    let response = engine.complete(request).await?;
    
    let completion_time = start_time.elapsed();
    
    // Output result
    match format {
        "json" => {
            let output = serde_json::json!({
                "completion": response.completion,
                "confidence": response.confidence,
                "alternatives": response.alternatives,
                "reasoning": response.reasoning,
                "quality_metrics": response.quality_metrics,
                "suggestions": response.suggestions,
                "completion_time_ms": response.completion_time_ms,
                "total_time_ms": completion_time.as_millis()
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        _ => {
            println!("Code Completion:");
            println!("================");
            println!("{}", response.completion);
            println!();
            println!("Confidence: {:.2}", response.confidence);
            println!("Time: {}ms", completion_time.as_millis());
            
            if !response.suggestions.is_empty() {
                println!("\nSuggestions:");
                for suggestion in &response.suggestions {
                    println!("  • {}", suggestion);
                }
            }
            
            if !response.alternatives.is_empty() {
                println!("\nAlternatives:");
                for (i, alt) in response.alternatives.iter().enumerate() {
                    println!("  {}. {} (confidence: {:.2})", i + 1, alt.description, alt.confidence);
                }
            }
        }
    }
    
    Ok(())
}

/// Generate code from prompt
async fn generate_code(
    model: &GoldbullCode,
    prompt: String,
    language: String,
    mode: String,
    max_tokens: usize,
    temperature: f64,
    include_docs: bool,
    output: Option<String>,
    format: &str,
) -> Result<()> {
    let start_time = Instant::now();
    
    // Parse language and mode
    let lang = parse_language(&language, &prompt)?;
    let completion_mode = parse_completion_mode(&mode)?;
    
    // Create generation request
    let config = GenerationConfig {
        max_tokens,
        temperature,
        include_documentation: include_docs,
        ..Default::default()
    };
    
    let context = GenerationContext {
        file_name: output.clone(),
        style_preferences: StylePreferences::default(),
        ..Default::default()
    };
    
    let request = GenerationRequest {
        prompt,
        language: lang,
        config,
        context,
        completion_mode,
    };
    
    // Generate code
    let mut generator = CodeGenerator::new(&model)?;
    let response = generator.generate(request).await?;
    
    let generation_time = start_time.elapsed();
    
    // Save to file if specified
    if let Some(output_path) = &output {
        fs::write(output_path, &response.code)?;
        info!("Generated code saved to: {}", output_path);
    }
    
    // Output result
    match format {
        "json" => {
            let output_json = serde_json::json!({
                "code": response.code,
                "confidence": response.confidence,
                "quality_metrics": response.quality_metrics,
                "syntax_validation": response.syntax_validation,
                "alternatives": response.alternatives,
                "metadata": response.metadata,
                "total_time_ms": generation_time.as_millis()
            });
            println!("{}", serde_json::to_string_pretty(&output_json)?);
        }
        _ => {
            println!("Generated Code:");
            println!("===============");
            println!("{}", response.code);
            println!();
            println!("Confidence: {:.2}", response.confidence);
            println!("Time: {}ms", generation_time.as_millis());
            println!("Quality Score: {:.2}", response.quality_metrics.overall_score);
            
            if !response.syntax_validation.is_valid {
                println!("\nSyntax Issues:");
                for error in &response.syntax_validation.errors {
                    println!("  • Line {}: {}", error.line, error.message);
                }
            }
        }
    }
    
    Ok(())
}

/// Analyze code syntax and features
async fn analyze_code(
    file: String,
    language: Option<String>,
    detailed: bool,
    output: Option<String>,
    format: &str,
) -> Result<()> {
    let start_time = Instant::now();
    
    // Read code file
    let code = fs::read_to_string(&file)?;
    
    // Detect or parse language
    let lang = if let Some(lang_str) = language {
        parse_language(&lang_str, &code)?
    } else {
        detect_language_from_file(&file)?
    };
    
    // Analyze code
    let mut analyzer = SyntaxAnalyzer::new(lang)?;
    let features = analyzer.analyze(&code)?;
    
    let analysis_time = start_time.elapsed();
    
    // Prepare output
    let output_data = if detailed {
        serde_json::to_value(&features)?
    } else {
        serde_json::json!({
            "language": features.language,
            "functions_count": features.functions.len(),
            "variables_count": features.variables.len(),
            "complexity": features.complexity,
            "quality_score": features.formatting,
            "analysis_time_ms": analysis_time.as_millis()
        })
    };
    
    // Save to file if specified
    if let Some(output_path) = &output {
        let output_str = serde_json::to_string_pretty(&output_data)?;
        fs::write(output_path, output_str)?;
        info!("Analysis saved to: {}", output_path);
    }
    
    // Output result
    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&output_data)?);
        }
        _ => {
            println!("Code Analysis:");
            println!("==============");
            println!("Language: {}", features.language);
            println!("Functions: {}", features.functions.len());
            println!("Variables: {}", features.variables.len());
            println!("Complexity: {}", features.complexity.cyclomatic_complexity);
            println!("Lines of Code: {}", features.complexity.lines_of_code);
            println!("Analysis Time: {}ms", analysis_time.as_millis());
            
            if detailed {
                println!("\nDetailed Analysis:");
                for func in &features.functions {
                    println!("  Function: {} (parameters: {})", func.name, func.parameters.len());
                }
                
                for import in &features.imports {
                    println!("  Import: {}", import.module);
                }
            }
        }
    }
    
    Ok(())
}

/// Interactive code completion session
async fn interactive_session(
    _model: &GoldbullCode,
    language: String,
    _syntax_highlighting: bool,
) -> Result<()> {
    println!("Goldbull Code Interactive Session");
    println!("=================================");
    println!("Language: {}", language);
    println!("Type 'exit' to quit, 'help' for commands");
    println!();
    
    let lang = parse_language(&language, "")?;
    let mut engine = CompletionEngine::new()?;
    let mut context = String::new();
    
    loop {
        print!("goldbull> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" {
            break;
        }
        
        if input == "help" {
            println!("Commands:");
            println!("  exit         - Exit the session");
            println!("  help         - Show this help");
            println!("  clear        - Clear context");
            println!("  context      - Show current context");
            println!("  <code>       - Add code and get completion");
            continue;
        }
        
        if input == "clear" {
            context.clear();
            println!("Context cleared");
            continue;
        }
        
        if input == "context" {
            println!("Current context:");
            println!("{}", context);
            continue;
        }
        
        if input.is_empty() {
            continue;
        }
        
        // Add input to context
        context.push_str(input);
        context.push('\n');
        
        // Generate completion
        let request = CompletionRequest {
            prefix: context.clone(),
            language: lang,
            max_tokens: 50,
            temperature: 0.2,
            ..Default::default()
        };
        
        match engine.complete(request).await {
            Ok(response) => {
                println!("Completion: {}", response.completion);
                
                // Ask if user wants to accept
                print!("Accept completion? (y/n): ");
                io::stdout().flush()?;
                
                let mut accept = String::new();
                io::stdin().read_line(&mut accept)?;
                
                if accept.trim().to_lowercase() == "y" {
                    context.push_str(&response.completion);
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
    
    println!("Session ended");
    Ok(())
}

/// Benchmark model performance
async fn benchmark_model(
    _model: &GoldbullCode,
    requests: usize,
    _dataset: Option<String>,
    save_results: Option<String>,
) -> Result<()> {
    println!("Benchmarking model performance...");
    
    let mut engine = CompletionEngine::new()?;
    let mut total_time = 0u128;
    let mut successful_completions = 0;
    
    let test_prompts = vec![
        "fn fibonacci(n: u32) -> u32 {",
        "def factorial(n):",
        "function isPrime(num) {",
        "class Calculator {",
        "import ",
    ];
    
    let start_time = Instant::now();
    
    for i in 0..requests {
        let prompt = test_prompts[i % test_prompts.len()];
        let request_start = Instant::now();
        
        let request = CompletionRequest {
            prefix: prompt.to_string(),
            language: LanguageType::Unknown,
            max_tokens: 50,
            temperature: 0.2,
            ..Default::default()
        };
        
        match engine.complete(request).await {
            Ok(_) => {
                successful_completions += 1;
                total_time += request_start.elapsed().as_millis();
            }
            Err(_) => {
                // Skip failed completions
            }
        }
        
        if (i + 1) % 10 == 0 {
            println!("Completed {}/{} requests", i + 1, requests);
        }
    }
    
    let total_benchmark_time = start_time.elapsed();
    let avg_time = if successful_completions > 0 {
        total_time / successful_completions as u128
    } else {
        0
    };
    
    let results = serde_json::json!({
        "total_requests": requests,
        "successful_completions": successful_completions,
        "success_rate": successful_completions as f64 / requests as f64,
        "average_completion_time_ms": avg_time,
        "total_time_ms": total_benchmark_time.as_millis(),
        "requests_per_second": successful_completions as f64 / total_benchmark_time.as_secs_f64()
    });
    
    // Save results if specified
    if let Some(output_path) = save_results {
        let results_str = serde_json::to_string_pretty(&results)?;
        fs::write(output_path, results_str)?;
    }
    
    // Display results
    println!("\nBenchmark Results:");
    println!("==================");
    println!("Total Requests: {}", requests);
    println!("Successful: {}", successful_completions);
    println!("Success Rate: {:.1}%", successful_completions as f64 / requests as f64 * 100.0);
    println!("Average Time: {}ms", avg_time);
    println!("Requests/sec: {:.1}", successful_completions as f64 / total_benchmark_time.as_secs_f64());
    
    Ok(())
}

/// Show model information
fn show_model_info(model: &GoldbullCode, detailed: bool, format: &str) -> Result<()> {
    let metadata = model.generate_metadata();
    
    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&metadata)?);
        }
        _ => {
            println!("Model Information:");
            println!("==================");
            println!("Version: {}", metadata.version);
            println!("Parameters: {}", metadata.num_parameters);
            println!("Memory Footprint: {} MB", metadata.memory_footprint / 1024 / 1024);
            println!("Supported Languages: {}", metadata.supported_languages.join(", "));
            
            if detailed {
                println!("\nCapabilities:");
                for capability in &metadata.capabilities {
                    println!("  • {}", capability);
                }
                
                println!("\nTraining Info:");
                println!("  Epochs: {}", metadata.training_info.epochs);
                println!("  Final Loss: {:.4}", metadata.training_info.final_loss);
                println!("  Dataset Size: {}", metadata.training_info.dataset_size);
            }
        }
    }
    
    Ok(())
}

/// Parse language string to LanguageType
fn parse_language(language: &str, code: &str) -> Result<LanguageType> {
    match language.to_lowercase().as_str() {
        "auto" => detect_language_from_code(code),
        "rust" | "rs" => Ok(LanguageType::Rust),
        "python" | "py" => Ok(LanguageType::Python),
        "javascript" | "js" => Ok(LanguageType::JavaScript),
        "typescript" | "ts" => Ok(LanguageType::TypeScript),
        "java" => Ok(LanguageType::Java),
        "cpp" | "c++" => Ok(LanguageType::Cpp),
        "c" => Ok(LanguageType::C),
        "go" => Ok(LanguageType::Go),
        _ => Ok(LanguageType::Unknown),
    }
}

/// Detect language from code content
fn detect_language_from_code(code: &str) -> Result<LanguageType> {
    // Simple heuristics for language detection
    if code.contains("fn ") || code.contains("let ") || code.contains("use ") {
        Ok(LanguageType::Rust)
    } else if code.contains("def ") || code.contains("import ") {
        Ok(LanguageType::Python)
    } else if code.contains("function ") || code.contains("var ") || code.contains("const ") {
        Ok(LanguageType::JavaScript)
    } else if code.contains("public class ") || code.contains("import java") {
        Ok(LanguageType::Java)
    } else {
        Ok(LanguageType::Unknown)
    }
}

/// Detect language from file extension
fn detect_language_from_file(file_path: &str) -> Result<LanguageType> {
    let path = std::path::Path::new(file_path);
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        match extension {
            "rs" => Ok(LanguageType::Rust),
            "py" => Ok(LanguageType::Python),
            "js" => Ok(LanguageType::JavaScript),
            "ts" => Ok(LanguageType::TypeScript),
            "java" => Ok(LanguageType::Java),
            "cpp" | "cc" | "cxx" => Ok(LanguageType::Cpp),
            "c" => Ok(LanguageType::C),
            "go" => Ok(LanguageType::Go),
            _ => Ok(LanguageType::Unknown),
        }
    } else {
        Ok(LanguageType::Unknown)
    }
}

/// Parse completion mode string
fn parse_completion_mode(mode: &str) -> Result<CompletionMode> {
    match mode.to_lowercase().as_str() {
        "line" => Ok(CompletionMode::Line),
        "block" => Ok(CompletionMode::Block),
        "function" => Ok(CompletionMode::Function),
        "type" => Ok(CompletionMode::Type),
        "module" => Ok(CompletionMode::Module),
        _ => Err(anyhow::anyhow!("Invalid completion mode: {}", mode)),
    }
}

/// Load context files for completion
fn load_context_files(files: &str) -> Result<Vec<goldbull_code::completion::ContextFile>> {
    let file_paths: Vec<&str> = files.split(',').collect();
    let mut context_files = Vec::new();
    
    for file_path in file_paths {
        let file_path = file_path.trim();
        if let Ok(content) = fs::read_to_string(file_path) {
            let language = detect_language_from_file(file_path)?;
            context_files.push(goldbull_code::completion::ContextFile {
                path: file_path.to_string(),
                content,
                language,
                relevance: 1.0,
            });
        }
    }
    
    Ok(context_files)
}