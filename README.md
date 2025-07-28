# Goldbull AI Model Suite

A family of efficient, lightweight AI models designed for CPU execution with <2GB memory footprint.

## Overview

The Goldbull AI Model Suite consists of specialized models for different AI tasks:

- **goldbull-text**: NLP Text Generation
- **goldbull-code**: Code Completion
- **goldbull-sage**: Question Answering
- **goldbull-brief**: Summarization
- **goldbull-vision**: Computer Vision
- **goldbull-multimodel**: Multimodal AI
- **goldbull-embedding**: Embedding Generator

## Key Features

- ✅ **CPU-Optimized**: Designed for efficient CPU execution
- ✅ **Memory Efficient**: <2GB memory footprint
- ✅ **Real-time Inference**: Optimized for fast CPU inference
- ✅ **32K Sequence Length**: Support for long sequences
- ✅ **TikToken Tokenizer**: Unicode-aware BPE tokenization
- ✅ **Multilingual**: 1M vocabulary from BPEmb
- ✅ **Function Calling**: External tool and API integration
- ✅ **Multiple Inference**: llama.cpp and ONNX support

## Architecture

### Core Components

- **goldbull-core**: Base model architecture and tensor operations
- **goldbull-tokenizer**: TikToken-style BPE tokenizer with Unicode support

### Specialized Models

Each model is optimized for its specific task while maintaining the core efficiency characteristics.

## Quick Start

```bash
# Build the entire workspace
cargo build --workspace

# Check all crates
cargo check --workspace

# Run tests
cargo test --workspace
```

## Model Specifications

| Component | Status | Memory | Features |
|-----------|---------|---------|----------|
| Core Library | ✅ | <500MB | Base architecture, tensor ops |
| Tokenizer | ✅ | <100MB | BPE, Unicode, 1M vocab |
| Text Model | 🔄 | <2GB | Text generation |
| Code Model | 🔄 | <2GB | Code completion |
| Sage Model | 🔄 | <2GB | Question answering |
| Brief Model | 🔄 | <2GB | Summarization |
| Vision Model | 🔄 | <2GB | Computer vision |
| Multimodel | 🔄 | <2GB | Multimodal AI |
| Embedding | 🔄 | <2GB | Embeddings |

## Development

This project is implemented in Rust for maximum performance and safety. The codebase follows a modular architecture with clear separation of concerns.

### Project Structure

```
goldbull-ai/
├── goldbull-core/          # Base model architecture
├── goldbull-tokenizer/     # TikToken-style tokenizer
├── goldbull-text/          # Text generation model
├── goldbull-code/          # Code completion model
├── goldbull-sage/          # Question answering model
├── goldbull-brief/         # Summarization model
├── goldbull-vision/        # Computer vision model
├── goldbull-multimodel/    # Multimodal model
├── goldbull-embedding/     # Embedding model
├── goldbull-training/      # Training utilities
└── goldbull-inference/     # Inference engines
```

## License

MIT License - see LICENSE file for details.