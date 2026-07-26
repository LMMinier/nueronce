use std::env;
use std::fs;
use std::time::Instant;

use nuetorch::{EntropyConfig, Model, ModelConfig, StepStats};

fn value(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
        .unwrap_or_else(|| default.to_owned())
}

fn present(args: &[String], flag: &str) -> bool {
    args.iter().any(|argument| argument == flag)
}

fn parse_config(args: &[String]) -> Result<ModelConfig, Box<dyn std::error::Error>> {
    let ema_decay = value(args, "--entropy-ema-decay", "0.95");
    let config = ModelConfig {
        width: value(args, "--width", "256").parse()?,
        lr: value(args, "--lr", "0.001").parse()?,
        entropy: EntropyConfig {
            global_threshold: value(args, "--entropy-global-threshold", "0.75").parse()?,
            relative_threshold: value(args, "--entropy-relative-threshold", "0.10").parse()?,
            ema_decay: if ema_decay == "off" {
                None
            } else {
                Some(ema_decay.parse()?)
            },
            entropy_loss_weight: value(args, "--entropy-loss-weight", "0.1").parse()?,
            boundary_loss_weight: value(args, "--boundary-loss-weight", "0.1").parse()?,
            boundary_positive_weight: value(args, "--boundary-positive-weight", "2.0").parse()?,
            boundary_probability_threshold: value(args, "--boundary-threshold", "0.5").parse()?,
            patch_min: value(args, "--patch-min", "2").parse()?,
            patch_max: value(args, "--patch-max", "16").parse()?,
            routing_enabled: !present(args, "--disable-routing"),
        },
    };
    config
        .validate()
        .map_err(|error| format!("invalid configuration: {error}"))?;
    Ok(config)
}

fn config_conflicts(saved: &ModelConfig, requested: &ModelConfig) -> Vec<&'static str> {
    let mut conflicts = Vec::new();
    if saved.width != requested.width {
        conflicts.push("width");
    }
    if saved.lr.to_bits() != requested.lr.to_bits() {
        conflicts.push("learning rate");
    }
    let saved = &saved.entropy;
    let requested = &requested.entropy;
    if saved.global_threshold.to_bits() != requested.global_threshold.to_bits() {
        conflicts.push("global entropy threshold");
    }
    if saved.relative_threshold.to_bits() != requested.relative_threshold.to_bits() {
        conflicts.push("relative entropy threshold");
    }
    if saved.ema_decay.map(f32::to_bits) != requested.ema_decay.map(f32::to_bits) {
        conflicts.push("entropy EMA decay");
    }
    if saved.entropy_loss_weight.to_bits() != requested.entropy_loss_weight.to_bits() {
        conflicts.push("entropy loss weight");
    }
    if saved.boundary_loss_weight.to_bits() != requested.boundary_loss_weight.to_bits() {
        conflicts.push("boundary loss weight");
    }
    if saved.boundary_positive_weight.to_bits() != requested.boundary_positive_weight.to_bits() {
        conflicts.push("boundary positive weight");
    }
    if saved.boundary_probability_threshold.to_bits()
        != requested.boundary_probability_threshold.to_bits()
    {
        conflicts.push("boundary probability threshold");
    }
    if saved.patch_min != requested.patch_min {
        conflicts.push("patch minimum");
    }
    if saved.patch_max != requested.patch_max {
        conflicts.push("patch maximum");
    }
    if saved.routing_enabled != requested.routing_enabled {
        conflicts.push("routing mode");
    }
    conflicts
}

fn print_stats(step: u64, stats: StepStats) {
    println!(
        "step={step} total_loss={:.6} next_loss={:.6} bpb={:.6} entropy_loss={:.6} boundary_loss={:.6} grad_norm={:.5} applied_grad_norm={:.5}",
        stats.loss, stats.next_byte_loss, stats.next_byte_loss / std::f32::consts::LN_2,
        stats.entropy_loss, stats.boundary_loss, stats.grad_norm, stats.applied_grad_norm
    );
    println!(
        "entropy_nats={:.6} entropy_normalized={:.6} entropy_bits={:.6} predicted_entropy_nats={:.6}",
        stats.entropy_nats, stats.normalized_entropy, stats.entropy_bits,
        stats.predicted_entropy_nats
    );
    println!(
        "boundary_accuracy={:.4} precision={:.4} recall={:.4} predicted_rate={:.4} target_rate={:.4}",
        stats.boundary_accuracy, stats.boundary_precision, stats.boundary_recall,
        stats.predicted_boundary_rate, stats.target_boundary_rate
    );
    println!(
        "patches={} average_patch_length={:.3} minimum_patch_length={} maximum_patch_length={} forced_boundaries={}",
        stats.patch_count, stats.average_patch_length, stats.minimum_patch_length,
        stats.maximum_patch_length, stats.forced_boundary_count
    );
    println!(
        "gate_average={:.5} gate_minimum={:.5} gate_maximum={:.5} gates_above_half_percent={:.2} routed_grad_norm={:.6}",
        stats.average_gate, stats.minimum_gate, stats.maximum_gate,
        stats.gates_above_half_percent, stats.routed_parameter_grad_norm
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: nuetorch train --input FILE [--width 256 --steps 100 --seq 128 --lr 0.001 --checkpoint model.ntrs] [entropy/boundary/patch options]");
        std::process::exit(2);
    }
    match args[1].as_str() {
        "train" => {
            let input = value(&args, "--input", "");
            if input.is_empty() {
                return Err("--input is required".into());
            }
            let steps: usize = value(&args, "--steps", "100").parse()?;
            let warmup_steps: usize = value(&args, "--warmup-steps", "0").parse()?;
            let log_every: usize = value(&args, "--log-every", "10").parse()?;
            let sequence_length: usize = value(&args, "--seq", "128").parse()?;
            if steps == 0 || sequence_length == 0 || log_every == 0 {
                return Err("--steps, --seq, and --log-every must be positive".into());
            }
            let requested_config = parse_config(&args)?;
            let checkpoint = value(&args, "--checkpoint", "nuetorch.ntrs");
            let data = fs::read(&input)?;
            if data.len() < sequence_length + 1 {
                return Err("input is shorter than seq + 1".into());
            }
            let mut model = if fs::metadata(&checkpoint).is_ok() {
                let mut loaded = Model::load(&checkpoint)?;
                let conflicts = config_conflicts(&loaded.config(), &requested_config);
                if !conflicts.is_empty() && !present(&args, "--override-config") {
                    return Err(format!("resume configuration conflicts: {}; pass --override-config to use the requested runtime values", conflicts.join(", ")).into());
                }
                if present(&args, "--override-config") {
                    if loaded.width != requested_config.width {
                        return Err("checkpoint width cannot be structurally overridden".into());
                    }
                    loaded.lr = requested_config.lr;
                    loaded.entropy_config = requested_config.entropy.clone();
                }
                loaded
            } else {
                Model::with_config(requested_config, 20260726)
            };
            println!(
                "params={} tensor_storage_bytes={} resume_step={}",
                model.parameter_count(),
                model.resident_tensor_bytes(),
                model.step
            );
            for _ in 0..warmup_steps {
                let offset =
                    ((model.step as usize).wrapping_mul(104729)) % (data.len() - sequence_length);
                model.train_bytes(&data[offset..offset + sequence_length + 1], 1.0);
            }
            if warmup_steps > 0 {
                println!(
                    "warmup_steps={warmup_steps} measured_start_step={}",
                    model.step
                );
            }
            let started = Instant::now();
            let start_step = model.step;
            for local in 0..steps {
                let offset =
                    ((model.step as usize).wrapping_mul(104729)) % (data.len() - sequence_length);
                let stats = model.train_bytes(&data[offset..offset + sequence_length + 1], 1.0);
                if local % log_every == 0 || local + 1 == steps {
                    print_stats(model.step, stats);
                }
            }
            let elapsed = started.elapsed().as_secs_f64();
            model.save_atomic(&checkpoint)?;
            println!("completed_steps={} seconds={:.3} steps_per_second={:.3} bytes_per_second={:.3} checkpoint={}",
                model.step - start_step, elapsed, steps as f64 / elapsed,
                (steps * sequence_length) as f64 / elapsed, checkpoint);
        }
        "inspect" => {
            let checkpoint = value(&args, "--checkpoint", "nuetorch.ntrs");
            let model = Model::load(&checkpoint)?;
            println!("step={} width={} params={} tensor_storage_bytes={} lr={} entropy_global_threshold={} entropy_relative_threshold={} entropy_ema={:?} routing_enabled={} patch_min={} patch_max={}",
                model.step, model.width, model.parameter_count(), model.resident_tensor_bytes(),
                model.lr, model.entropy_config.global_threshold,
                model.entropy_config.relative_threshold, model.entropy_ema,
                model.entropy_config.routing_enabled, model.entropy_config.patch_min,
                model.entropy_config.patch_max);
        }
        other => return Err(format!("unknown command: {other}").into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_conflicts_cover_required_configuration() {
        let saved = ModelConfig {
            width: 16,
            lr: 0.001,
            entropy: EntropyConfig::default(),
        };
        let mut changed = saved.clone();
        changed.width = 32;
        changed.lr = 0.002;
        changed.entropy.global_threshold = 0.8;
        changed.entropy.relative_threshold = 0.2;
        changed.entropy.entropy_loss_weight = 0.3;
        changed.entropy.boundary_loss_weight = 0.4;
        changed.entropy.patch_min = 3;
        changed.entropy.patch_max = 20;
        changed.entropy.routing_enabled = false;
        let conflicts = config_conflicts(&saved, &changed);
        for required in [
            "width",
            "learning rate",
            "global entropy threshold",
            "relative entropy threshold",
            "entropy loss weight",
            "boundary loss weight",
            "patch minimum",
            "patch maximum",
            "routing mode",
        ] {
            assert!(
                conflicts.contains(&required),
                "missing conflict: {required}"
            );
        }
    }
}
