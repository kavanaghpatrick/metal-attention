use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::{Parser, Subcommand};

use metal_attention::config::InferenceConfig;
use metal_attention::generate_streaming;
use metal_attention::model::HybridModel;
use metal_attention::GpuForwardPass;

use metal_attention_gguf::parser::{GgufError, GgufFile};
use metal_attention_gguf::quantize::GgufType;
use metal_attention_gguf::tokenizer::GgufTokenizer;
// GPU device and pipeline state cache for weight dequantization
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::PsoCache;

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

        /// Use GPU forward pass (Metal compute kernels)
        #[arg(long)]
        gpu: bool,
    },
    /// Benchmark model performance
    Bench {
        /// Path to GGUF model file (ignored if --synthetic)
        #[arg(short = 'm', long)]
        model: Option<PathBuf>,

        /// Comma-separated sequence lengths to benchmark
        #[arg(long, default_value = "128,256,512")]
        seq_lengths: String,

        /// Number of decode tokens to generate per run
        #[arg(long, default_value = "64")]
        gen_length: usize,

        /// Number of iterations per sequence length
        #[arg(long, default_value = "3")]
        iterations: usize,

        /// Use synthetic random data (no real model needed)
        #[arg(long)]
        synthetic: bool,

        /// Output results as JSONL
        #[arg(long)]
        json: bool,

        /// Random seed
        #[arg(short = 's', long, default_value = "42")]
        seed: u64,

        /// Use GPU forward pass (Metal compute kernels)
        #[arg(long)]
        gpu: bool,
    },
    /// Show model info from GGUF metadata
    Info {
        /// Path to GGUF model file
        #[arg(short = 'm', long)]
        model: PathBuf,

        /// Output as JSON
        #[arg(long)]
        json: bool,
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
            gpu,
        } => {
            if gpu {
                if let Err(e) = run_inference_gpu(model, prompt, max_tokens) {
                    eprintln!("{e}");
                    process::exit(1);
                }
            } else if let Err(e) = run_inference(
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
        Commands::Bench {
            model,
            seq_lengths,
            gen_length,
            iterations,
            synthetic,
            json,
            seed,
            gpu,
        } => {
            if gpu {
                if let Err(e) = run_bench_gpu(model, &seq_lengths, gen_length, iterations, json) {
                    eprintln!("{e}");
                    process::exit(1);
                }
            } else if let Err(e) = run_bench(
                model,
                &seq_lengths,
                gen_length,
                iterations,
                synthetic,
                json,
                seed,
            ) {
                eprintln!("{e}");
                process::exit(1);
            }
        }
        Commands::Info { model, json } => {
            if let Err(e) = run_info(model, json) {
                eprintln!("{e}");
                process::exit(1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// info subcommand
// ---------------------------------------------------------------------------

fn run_info(model_path: PathBuf, json_output: bool) -> Result<(), String> {
    if !model_path.exists() {
        return Err(format!(
            "Error: Model file not found: {}",
            model_path.display()
        ));
    }

    let gguf = GgufFile::open(&model_path).map_err(|e| match &e {
        GgufError::BadMagic(_) => format!("Error: Invalid GGUF file: {e}"),
        _ => format!("Error: Failed to parse GGUF file: {e}"),
    })?;

    // Extract metadata fields
    let arch = gguf.architecture;
    let arch_str = format!("{:?}", arch);

    let arch_prefix = arch_str.to_lowercase();
    let hidden_size = gguf
        .metadata
        .get_u32(&format!("{arch_prefix}.embedding_length"))
        .or_else(|| gguf.metadata.get_u32("general.hidden_size"))
        .or_else(|| gguf.metadata.get_u32("rwkv.embedding_size"))
        .unwrap_or(0) as usize;

    let num_heads = gguf
        .metadata
        .get_u32(&format!("{arch_prefix}.attention.head_count"))
        .or_else(|| gguf.metadata.get_u32("general.num_attention_heads"))
        .or_else(|| gguf.metadata.get_u32("rwkv.num_heads"))
        .unwrap_or(0) as usize;

    let head_dim = if num_heads > 0 && hidden_size > 0 {
        hidden_size / num_heads
    } else {
        0
    };

    let num_layers = gguf
        .metadata
        .get_u32(&format!("{arch_prefix}.block_count"))
        .or_else(|| gguf.metadata.get_u32("general.num_layers"))
        .or_else(|| gguf.metadata.get_u32("rwkv.block_count"))
        .unwrap_or(0) as usize;

    let num_kv_heads = gguf
        .metadata
        .get_u32(&format!("{arch_prefix}.attention.head_count_kv"))
        .or_else(|| gguf.metadata.get_u32("general.num_kv_heads"))
        .unwrap_or(num_heads as u32) as usize;

    // Detect dominant quantization type from tensors
    let quant_type = detect_dominant_quant(&gguf);
    let quant_str = format!("{:?}", quant_type);

    // Vocab size from tokenizer tokens array or metadata
    let vocab_size = gguf
        .metadata
        .get_array_string("tokenizer.ggml.tokens")
        .map(|t| t.len())
        .unwrap_or(0);

    // Tokenizer type
    let tokenizer_type = gguf
        .metadata
        .get_string("tokenizer.ggml.model")
        .unwrap_or("unknown")
        .to_string();

    // Model name
    let model_name = gguf
        .metadata
        .get_string("general.name")
        .unwrap_or("unknown")
        .to_string();

    // Estimated memory: sum of all tensor byte sizes
    let total_bytes: usize = gguf.tensors.iter().map(|t| t.byte_size()).sum();
    let memory_mb = total_bytes as f64 / (1024.0 * 1024.0);

    // File size
    let file_size = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);

    if json_output {
        let json = format!(
            concat!(
                "{{",
                "\"name\":\"{}\",",
                "\"architecture\":\"{}\",",
                "\"num_layers\":{},",
                "\"hidden_size\":{},",
                "\"head_dim\":{},",
                "\"num_heads\":{},",
                "\"num_kv_heads\":{},",
                "\"quantization\":\"{}\",",
                "\"vocab_size\":{},",
                "\"tokenizer_type\":\"{}\",",
                "\"num_tensors\":{},",
                "\"estimated_memory_mb\":{:.1},",
                "\"file_size_bytes\":{}",
                "}}"
            ),
            escape_json(&model_name),
            escape_json(&arch_str),
            num_layers,
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            escape_json(&quant_str),
            vocab_size,
            escape_json(&tokenizer_type),
            gguf.tensors.len(),
            memory_mb,
            file_size,
        );
        println!("{json}");
    } else {
        println!("Model Info: {}", model_path.display());
        println!("  Name:           {model_name}");
        println!("  Architecture:   {arch_str}");
        println!("  Layers:         {num_layers}");
        println!("  Hidden size:    {hidden_size}");
        println!("  Head dim:       {head_dim}");
        println!("  Num heads:      {num_heads}");
        println!("  Num KV heads:   {num_kv_heads}");
        println!("  Quantization:   {quant_str}");
        println!("  Vocab size:     {vocab_size}");
        println!("  Tokenizer:      {tokenizer_type}");
        println!("  Tensors:        {}", gguf.tensors.len());
        println!("  Est. memory:    {memory_mb:.1} MB");
        println!("  File size:      {} bytes", file_size);
    }

    Ok(())
}

/// Detect the most common quantization type among weight tensors.
fn detect_dominant_quant(gguf: &GgufFile) -> GgufType {
    let mut counts: HashMap<GgufType, usize> = HashMap::new();
    for t in &gguf.tensors {
        *counts.entry(t.gguf_type).or_insert(0) += 1;
    }
    // Return the type with the most tensors (excluding F32 if others exist,
    // since F32 is often used for norms/biases)
    let non_f32: Vec<_> = counts
        .iter()
        .filter(|(k, _)| **k != GgufType::F32)
        .collect();
    if non_f32.is_empty() {
        GgufType::F32
    } else {
        *non_f32.iter().max_by_key(|(_, count)| *count).unwrap().0
    }
}

/// Minimal JSON string escaping.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// bench subcommand
// ---------------------------------------------------------------------------

fn run_bench(
    model_path: Option<PathBuf>,
    seq_lengths_str: &str,
    gen_length: usize,
    iterations: usize,
    synthetic: bool,
    json_output: bool,
    seed: u64,
) -> Result<(), String> {
    // Parse sequence lengths
    let seq_lengths: Vec<usize> = seq_lengths_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| format!("Invalid sequence length: '{}'", s.trim()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if seq_lengths.is_empty() {
        return Err("No sequence lengths specified".to_string());
    }

    // Determine model config
    let (hidden_size, head_dim, num_heads, num_layers, vocab_size) = if synthetic {
        // Synthetic defaults: small model
        (768, 64, 12, 12, 32000)
    } else {
        let path = model_path
            .as_ref()
            .ok_or("Error: --model required unless --synthetic is set")?;
        if !path.exists() {
            return Err(format!("Error: Model file not found: {}", path.display()));
        }
        let gguf = GgufFile::open(path).map_err(|e| format!("Error: {e}"))?;
        let hs = gguf
            .metadata
            .get_u32("general.hidden_size")
            .or_else(|| gguf.metadata.get_u32("rwkv.embedding_size"))
            .unwrap_or(768) as usize;
        let nh = gguf
            .metadata
            .get_u32("general.num_attention_heads")
            .or_else(|| gguf.metadata.get_u32("rwkv.num_heads"))
            .unwrap_or(12) as usize;
        let hd = if nh > 0 { hs / nh } else { hs };
        let nl = gguf
            .metadata
            .get_u32("general.num_layers")
            .or_else(|| gguf.metadata.get_u32("rwkv.block_count"))
            .unwrap_or(12) as usize;
        let vs = gguf
            .metadata
            .get_array_string("tokenizer.ggml.tokens")
            .map(|t| t.len())
            .unwrap_or(32000);
        (hs, hd, nh, nl, vs)
    };

    let model = HybridModel::random(
        vocab_size,
        hidden_size,
        head_dim,
        num_heads,
        num_layers,
        seed,
    );

    if !json_output {
        eprintln!(
            "Benchmark config: {}H, {}D, {}L, vocab={}, gen_length={}, iterations={}",
            num_heads, hidden_size, num_layers, vocab_size, gen_length, iterations,
        );
        if synthetic {
            eprintln!("Mode: synthetic (random weights + random tokens)");
        }
        eprintln!();
    }

    let config = InferenceConfig {
        model_path: model_path.clone().unwrap_or_default(),
        max_tokens: gen_length,
        temperature: 0.0, // greedy for reproducible bench
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        seed: Some(seed),
    };

    for &seq_len in &seq_lengths {
        // Generate synthetic prompt tokens
        let prompt_tokens: Vec<u32> = (0..seq_len)
            .map(|i| ((i * 7 + 13) % vocab_size) as u32)
            .collect();

        let mut prefill_times = Vec::with_capacity(iterations);
        let mut decode_times = Vec::with_capacity(iterations);
        let mut decode_tokens_counts = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            // Prefill: measure time for first forward pass through all prompt tokens
            let prefill_start = Instant::now();
            let mut token_count = 0usize;
            let mut first_token_time = None;

            generate_streaming(&model, &prompt_tokens, &config, |_token_id| {
                if first_token_time.is_none() {
                    first_token_time = Some(prefill_start.elapsed());
                }
                token_count += 1;
                token_count < gen_length
            });

            let total_time = prefill_start.elapsed();
            let prefill_dur = first_token_time.unwrap_or(total_time);
            let decode_dur = total_time.saturating_sub(prefill_dur);

            prefill_times.push(prefill_dur.as_secs_f64());
            decode_times.push(decode_dur.as_secs_f64());
            decode_tokens_counts.push(if token_count > 0 { token_count - 1 } else { 0 });
        }

        // Compute averages
        let avg_prefill = prefill_times.iter().sum::<f64>() / iterations as f64;
        let avg_decode = decode_times.iter().sum::<f64>() / iterations as f64;
        let avg_decode_tokens =
            decode_tokens_counts.iter().sum::<usize>() as f64 / iterations as f64;
        let prefill_tok_s = if avg_prefill > 0.0 {
            seq_len as f64 / avg_prefill
        } else {
            0.0
        };
        let decode_tok_s = if avg_decode > 0.0 {
            avg_decode_tokens / avg_decode
        } else {
            0.0
        };
        let total_time_avg = avg_prefill + avg_decode;

        if json_output {
            println!(
                concat!(
                    "{{",
                    "\"seq_len\":{},",
                    "\"gen_length\":{},",
                    "\"iterations\":{},",
                    "\"prefill_tok_s\":{:.1},",
                    "\"decode_tok_s\":{:.1},",
                    "\"avg_prefill_s\":{:.4},",
                    "\"avg_decode_s\":{:.4},",
                    "\"total_time_s\":{:.4},",
                    "\"synthetic\":{}",
                    "}}"
                ),
                seq_len,
                gen_length,
                iterations,
                prefill_tok_s,
                decode_tok_s,
                avg_prefill,
                avg_decode,
                total_time_avg,
                synthetic,
            );
        } else {
            println!("--- seq_len={seq_len} ---");
            println!("  Prefill: {prefill_tok_s:.1} tok/s ({avg_prefill:.4}s avg)");
            println!("  Decode:  {decode_tok_s:.1} tok/s ({avg_decode:.4}s avg)");
            println!("  Total:   {total_time_avg:.4}s avg over {iterations} iterations");
            println!();
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// run subcommand
// ---------------------------------------------------------------------------

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

    // 2. Parse GGUF for tokenizer (before loading weights)
    let gguf = GgufFile::open(&model_path).map_err(|e| match &e {
        GgufError::BadMagic(_) => format!(
            "Error: Invalid GGUF file: {}\nHint: The file does not appear to be a valid GGUF model",
            e
        ),
        _ => format!("Error: Failed to parse GGUF file: {e}"),
    })?;

    // 3. Build tokenizer from GGUF metadata
    let tokenizer = GgufTokenizer::from_metadata(&gguf.metadata)
        .map_err(|e| format!("Error: Failed to build tokenizer from GGUF metadata: {e}"))?;
    drop(gguf); // Free mmap before loading weights (from_gguf opens its own)

    // 4. Construct model from GGUF weights
    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let hybrid_model = HybridModel::from_gguf(&model_path, Some(&device), Some(&mut pso_cache))
        .map_err(|e| format!("Error: Failed to load model from GGUF: {e}"))?;

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

// ---------------------------------------------------------------------------
// GPU run subcommand
// ---------------------------------------------------------------------------

fn run_inference_gpu(model_path: PathBuf, prompt: String, max_tokens: usize) -> Result<(), String> {
    if !model_path.exists() {
        return Err(format!(
            "Error: Model file not found: {}",
            model_path.display()
        ));
    }

    // Parse GGUF for tokenizer
    let gguf = GgufFile::open(&model_path).map_err(|e| format!("Error: {e}"))?;
    let tokenizer = GgufTokenizer::from_metadata(&gguf.metadata)
        .map_err(|e| format!("Error: Failed to build tokenizer: {e}"))?;
    let eos_id = tokenizer.eos_token_id();
    drop(gguf);

    // Construct GPU forward pass
    eprintln!("Loading GPU forward pass...");
    let mut gpu = GpuForwardPass::from_gguf(&model_path)?;

    // Tokenize prompt
    let prompt_tokens = tokenizer.encode(&prompt);
    if prompt_tokens.is_empty() {
        return Err("Error: Prompt produced no tokens".to_string());
    }

    eprintln!(
        "Prompt: {} tokens | Generating up to {} tokens (GPU, greedy)",
        prompt_tokens.len(),
        max_tokens
    );

    let start = Instant::now();
    let mut stdout = std::io::stdout();

    // Prefill: run forward_token for each prompt token except last (discard logits)
    let prefill_start = Instant::now();
    for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
        gpu.forward_token(tok)?;
    }
    // Last prompt token uses greedy path to get first generated token
    let mut next_token = gpu.forward_token_greedy(*prompt_tokens.last().unwrap())?;
    let prefill_elapsed = prefill_start.elapsed();

    // Decode loop: GPU-side greedy argmax
    let mut token_count: usize = 0;
    let decode_start = Instant::now();

    for _ in 0..max_tokens {
        // Stop on EOS
        if next_token == eos_id {
            break;
        }

        token_count += 1;
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        let _ = stdout.flush();

        // Forward next token with GPU-side argmax
        next_token = gpu.forward_token_greedy(next_token)?;
    }
    let decode_elapsed = decode_start.elapsed();

    println!();

    let total_elapsed = start.elapsed();
    let decode_tok_s = if decode_elapsed.as_secs_f64() > 0.0 {
        token_count as f64 / decode_elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!(
        "\n--- GPU Generation complete ---\n\
         Tokens generated: {}\n\
         Prefill time: {:.3}s ({} tokens, {:.1} tok/s)\n\
         Decode time:  {:.3}s ({} tokens, {:.1} tok/s)\n\
         Total time:   {:.2}s",
        token_count,
        prefill_elapsed.as_secs_f64(),
        prompt_tokens.len(),
        prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64().max(1e-9),
        decode_elapsed.as_secs_f64(),
        token_count,
        decode_tok_s,
        total_elapsed.as_secs_f64(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// GPU bench subcommand
// ---------------------------------------------------------------------------

fn run_bench_gpu(
    model_path: Option<PathBuf>,
    seq_lengths_str: &str,
    gen_length: usize,
    iterations: usize,
    json_output: bool,
) -> Result<(), String> {
    let path = model_path
        .as_ref()
        .ok_or("Error: --model required for GPU bench")?;
    if !path.exists() {
        return Err(format!("Error: Model file not found: {}", path.display()));
    }

    let seq_lengths: Vec<usize> = seq_lengths_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| format!("Invalid sequence length: '{}'", s.trim()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if seq_lengths.is_empty() {
        return Err("No sequence lengths specified".to_string());
    }

    if !json_output {
        eprintln!(
            "GPU Benchmark: gen_length={}, iterations={}, seq_lengths={:?}",
            gen_length, iterations, seq_lengths
        );
    }

    for &seq_len in &seq_lengths {
        let mut decode_times = Vec::with_capacity(iterations);

        for iter in 0..iterations {
            // Fresh GPU forward pass each iteration (clean KV cache)
            let mut gpu = GpuForwardPass::from_gguf(path)?;

            // Prefill with synthetic tokens
            let prompt_tokens: Vec<u32> = (0..seq_len)
                .map(|i| ((i * 7 + 13) % 49152) as u32)
                .collect();

            for &tok in &prompt_tokens {
                gpu.forward_token(tok)?;
            }

            // Warmup: 3 decode steps (only on first iteration)
            if iter == 0 {
                for w in 0..3u32 {
                    let _ = gpu.forward_token_greedy(w + 1)?;
                }
            }

            // Timed decode loop
            let mut next = gpu.forward_token_greedy(1u32)?; // seed token
            let decode_start = Instant::now();
            for _ in 0..gen_length {
                next = gpu.forward_token_greedy(next)?;
            }
            let decode_elapsed = decode_start.elapsed();
            decode_times.push(decode_elapsed.as_secs_f64());
        }

        let avg_decode = decode_times.iter().sum::<f64>() / iterations as f64;
        let decode_tok_s = if avg_decode > 0.0 {
            gen_length as f64 / avg_decode
        } else {
            0.0
        };

        if json_output {
            println!(
                concat!(
                    "{{",
                    "\"seq_len\":{},",
                    "\"gen_length\":{},",
                    "\"iterations\":{},",
                    "\"decode_tok_s\":{:.1},",
                    "\"avg_decode_s\":{:.4},",
                    "\"gpu\":true",
                    "}}"
                ),
                seq_len, gen_length, iterations, decode_tok_s, avg_decode,
            );
        } else {
            println!("--- GPU bench seq_len={seq_len} ---");
            println!("  Decode: {decode_tok_s:.1} tok/s ({avg_decode:.4}s avg over {iterations} iterations)");
            println!();
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Utility: greedy argmax
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}
