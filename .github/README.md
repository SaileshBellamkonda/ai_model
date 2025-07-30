# GitHub Workflows

This directory contains GitHub Actions workflows for the Goldbull AI Model Suite.

## Workflows

### CI Build and Test (`ci.yml`)
- **Trigger**: Push to main branch and pull requests
- **Purpose**: Continuous integration builds and tests
- **Platforms**: 
  - Linux (x86_64, ARM64)
  - Windows (x86_64, ARM64)
- **Features**:
  - Cross-compilation for multiple architectures
  - Caching for faster builds
  - Separate test job with linting and security audit
  - Graceful handling of compilation errors in some packages

### Release Build (`release.yml`)
- **Trigger**: Release creation and version tags (v*)
- **Purpose**: Create release artifacts
- **Platforms**: Same as CI workflow
- **Features**:
  - Optimized release builds
  - Automatic asset upload to GitHub releases
  - Creates compressed archives (tar.gz for Unix, zip for Windows)
  - Includes documentation files in release packages

### Nightly Build (`nightly.yml`)
- **Trigger**: Daily schedule (2 AM UTC) and manual dispatch
- **Purpose**: Monitor build health over time
- **Platforms**: Linux and Windows (x86_64 only)
- **Features**:
  - Simplified build process
  - Short-term artifact retention (7 days)

## Cross-Compilation Support

The workflows support cross-compilation for:
- **Linux**: x86_64 and ARM64 (aarch64)
- **Windows**: x86_64 and ARM64

ARM64 Linux builds use the `aarch64-linux-gnu-gcc` cross-compiler, automatically installed during the workflow.

## Build Configuration

The `.cargo/config.toml` file provides:
- Cross-compilation linker configuration
- Release profile optimizations for smaller binaries
- Performance optimizations (LTO, single codegen unit)

## Artifact Handling

Workflows handle compilation errors gracefully:
- Core libraries (goldbull-core, goldbull-tokenizer) are expected to build successfully
- Application binaries (goldbull-text, goldbull-code) may have compilation issues
- Artifacts are created only for successfully built binaries
- Build failures don't stop the entire workflow

## Usage

### Manual Workflow Dispatch
The nightly workflow can be triggered manually from the GitHub Actions tab.

### Creating Releases
1. Create a git tag with version (e.g., `v1.0.0`)
2. Push the tag: `git push origin v1.0.0`
3. The release workflow will automatically create build artifacts
4. Optionally create a GitHub release for automatic asset upload