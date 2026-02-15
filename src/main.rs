use std::io::Write;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::{Parser, Subcommand};

use metal_attention::config::InferenceConfig;
use metal_attention::model::HybridModel;
use metal_attention::generate_streaming;

use metal_attention_gguf::parser::{GgufError, GgufFile};
use metal_attention_gguf::tokenizer::GgufTokenizer;
use metal_attention_models::registry::{self, ModelConfig};

#[derive(Parser)]
#[command(name = "metal-attention")]
#[command(about = "Hybrid model inference engine for Apple Silicon")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model
    Run {
        /// Path to GGUF model file
        #[arg(short = 'm', long)]
        model: PathBuf,

        /// Input prompt
        #[arg(short = 'p', long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,

        /// Random seed
        #[arg(short = 's', long)]
        seed: Option<u64>,

        /// Temperature (0.0 = greedy)
        #[arg(long, default_value = "0.8")]
        temp: f32,

        /// Top-p (nucleus) sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Top-k sampling (0 = disabled)
        #[arg(long, default_value = "40")]
        top_k: usize,

        /// Repetition penalty (1.0 = disabled)
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,
    },
    /// Benchmark model performance (placeholder)
    Bench {
        /// Path to GGUF model file
        #[arg(short = 'm', long)]
        model: PathBuf,
    },
    /// Show model info (placeholder)
    Info {
        /// Path to GGUF model file
        #[arg(short = 'm', long)]
        model: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            seed,
            temp,
            top_p,
            top_k,
            repeat_penalty,
        } => {
            if let Err(e) = run_inference(
                model,
                prompt,
                max_tokens,
                seed,
                temp,
                top_p,
                top_k,
                repeat_penalty,
            ) {
                eprintln!("{e}");
                process::exit(1);
            }
        }
        Commands::Bench { model: _ } => {
            eprintln!("Not implemented yet");
        }
        Commands::Info { model: _ } => {
            eprintln!("Not implemented yet");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_inference(
    model_path: PathBuf,
    prompt: String,
    max_tokens: usize,
    seed: Option<u64>,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
) -> Result<(), String> {
    // 1. Validate model file exists
    if !model_path.exists() {
        return Err(format!(
            "Error: Model file not found: {}\nHint: Download a GGUF model and provide the path with -m",
            model_path.display()
        ));
    }

    // 2. Open and parse GGUF file
    let gguf = GgufFile::open(&model_path).map_err(|e| match &e {
        GgufError::BadMagic(_) => format!(
            "Error: Invalid GGUF file: {}\nHint: The file does not appear to be a valid GGUF model",
            e
        ),
        _ => format!("Error: Failed to parse GGUF file: {e}"),
    })?;

    // 3. Detect architecture and check support
    let arch = gguf.architecture;
    if !registry::is_supported(arch) {
        return Err(format!(
            "Error: Unsupported model architecture: {:?}\nSupported: {:?}",
            arch,
            registry::supported_architectures()
        ));
    }

    // 4. Extract model config from metadata
    let hidden_size = gguf
        .metadata
        .get_u32("general.hidden_size")
        .or_else(|| gguf.metadata.get_u32("rwkv.embedding_size"))
        .unwrap_or(768) as usize;
    let num_heads = gguf
        .metadata
        .get_u32("general.num_attention_heads")
        .or_else(|| gguf.metadata.get_u32("rwkv.num_heads"))
        .unwrap_or(12) as usize;
    let head_dim = if num_heads > 0 {
        hidden_size / num_heads
    } else {
        hidden_size
    };
    let num_layers = gguf
        .metadata
        .get_u32("general.num_layers")
        .or_else(|| gguf.metadata.get_u32("rwkv.block_count"))
        .unwrap_or(12) as usize;

    let model_config = ModelConfig {
        architecture: arch,
        hidden_size,
        head_dim,
        num_heads,
        num_layers,
    };

    eprintln!(
        "Loading model: {:?} ({}L, {}H, {}D)",
        arch, num_layers, num_heads, hidden_size
    );

    // 5. Build tokenizer from GGUF metadata
    let tokenizer = GgufTokenizer::from_metadata(&gguf.metadata).map_err(|e| {
        format!("Error: Failed to build tokenizer from GGUF metadata: {e}")
    })?;

    // 6. Construct model (random weights for now -- real weight loading in later phase)
    let vocab_size = tokenizer.vocab_size();
    let hybrid_model = HybridModel::random(
        vocab_size,
        model_config.hidden_size,
        model_config.head_dim,
        model_config.num_heads,
        model_config.num_layers,
        seed.unwrap_or(42),
    );

    // 7. Build inference config
    let config = InferenceConfig {
        model_path: model_path.clone(),
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        seed,
    };

    // 8. Tokenize prompt
    let prompt_tokens = tokenizer.encode(&prompt);
    if prompt_tokens.is_empty() {
        return Err("Error: Prompt produced no tokens".to_string());
    }

    eprintln!(
        "Prompt: {} tokens | Generating up to {} tokens | temp={} top_p={} top_k={}",
        prompt_tokens.len(),
        max_tokens,
        temperature,
        top_p,
        top_k
    );

    // 9. Run prefill + streaming decode
    let start = Instant::now();
    let mut token_count: usize = 0;
    let prefill_start = Instant::now();

    // Prefill timing (embedded in generate_streaming, but we track wall clock)
    let _prefill_elapsed = prefill_start.elapsed();

    let mut stdout = std::io::stdout();

    generate_streaming(&hybrid_model, &prompt_tokens, &config, |token_id| {
        token_count += 1;
        let text = tokenizer.decode(&[token_id]);
        print!("{text}");
        let _ = stdout.flush();

        // Stop on EOS token
        if token_id == tokenizer.eos_token_id() {
            return false;
        }
        true
    });

    // Ensure final newline on stdout
    println!();

    // 10. Print stats to stderr
    let total_elapsed = start.elapsed();
    let tok_per_sec = if total_elapsed.as_secs_f64() > 0.0 {
        token_count as f64 / total_elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!(
        "\n--- Generation complete ---\nTokens generated: {}\nTotal time: {:.2}s\nSpeed: {:.1} tok/s",
        token_count,
        total_elapsed.as_secs_f64(),
        tok_per_sec,
    );

    Ok(())
}
