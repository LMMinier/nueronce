use super::*;

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("nuetorch-{name}-{}.bin", std::process::id()))
}

#[test]
fn entropy_known_distributions() {
    let uniform = vec![1.0 / 256.0; 256];
    assert!((distribution_entropy_nats(&uniform) - entropy_uniform_nats()).abs() < 1e-5);
    assert!((entropy_bits(&uniform) - 8.0).abs() < 1e-4);
    let mut one_hot = vec![0.0; 256];
    one_hot[7] = 1.0;
    assert!(distribution_entropy_nats(&one_hot).abs() < 1e-7);
    for probabilities in [&uniform[..], &one_hot[..]] {
        assert!((-1e-6..=1.0 + 1e-6).contains(&normalized_entropy(probabilities)));
    }
}

#[test]
fn entropy_head_gradient_matches_finite_difference() {
    let config = EntropyConfig {
        boundary_loss_weight: 0.0,
        routing_enabled: false,
        ..Default::default()
    };
    let mut model = Model::with_config(
        ModelConfig {
            width: 4,
            lr: 0.001,
            entropy: config,
        },
        17,
    );
    let parameter = model.entropy_weight.offset;
    let original = model.arena.values[parameter];
    let epsilon = 1e-3;
    let loss_at = |model: &mut Model, value: f32| {
        model.arena.values[parameter] = value;
        let (_, _, _) = model.forward(b'a' as usize, false);
        let target = distribution_entropy_nats(&model.probs);
        let prediction = dot(&model.hidden, model.arena.get(model.entropy_weight))
            + model.arena.values[model.entropy_bias.offset];
        0.5 * model.entropy_config.entropy_loss_weight * (prediction - target).powi(2)
    };
    let plus = loss_at(&mut model, original + epsilon);
    let minus = loss_at(&mut model, original - epsilon);
    model.arena.values[parameter] = original;
    model.train_bytes(b"ab", f32::MAX);
    let analytic = model.arena.grads[parameter];
    let numerical = (plus - minus) / (2.0 * epsilon);
    let tolerance = 2e-4 + 2e-3 * numerical.abs();
    assert!(
        (analytic - numerical).abs() <= tolerance,
        "analytic={analytic} numerical={numerical}"
    );
}

#[test]
fn syntax_and_entropy_boundary_targets_are_independent() {
    for byte in [b' ', b'\n', b'.', b'!', b'(', b']', b'`'] {
        assert!(syntax_boundary(byte));
    }
    assert!(!syntax_boundary(b'a'));
    let config = EntropyConfig {
        global_threshold: 0.5,
        relative_threshold: 0.1,
        ..Default::default()
    };
    let high = entropy_uniform_nats() * 0.8;
    assert!(entropy_boundary_target(high, Some(high - 0.2), &config));
    assert!(!syntax_boundary(b'a'));
    assert!(combined_boundary_target(b'a', true));
    assert!(combined_boundary_target(b' ', false));
    assert!(!combined_boundary_target(b'a', false));
}

#[test]
fn dynamic_patch_invariants() {
    let minimum = dynamic_patches(&[1.0, 1.0, 1.0, 1.0, 1.0], 0.5, 3, 8);
    assert_eq!(minimum.ranges, vec![0..3, 3..5]);
    assert!(minimum
        .ranges
        .iter()
        .take(minimum.ranges.len() - 1)
        .all(|r| r.len() >= 3));
    let maximum = dynamic_patches(&[0.0; 10], 0.5, 2, 4);
    assert_eq!(maximum.ranges, vec![0..4, 4..8, 8..10]);
    assert_eq!(maximum.forced_boundary_count, 2);
    assert!(maximum.ranges.iter().all(|r| !r.is_empty() && r.len() <= 4));
    let empty = dynamic_patches(&[], 0.5, 2, 4);
    assert!(empty.ranges.is_empty());
}

#[test]
fn routing_changes_logits_and_receives_gradients() {
    let mut model = Model::new(8, 0.001, 11);
    let disabled = model.predict_logits(b'a', false);
    let enabled = model.predict_logits(b'a', true);
    assert_ne!(disabled, enabled);
    let stats = model.train_bytes(b"abca", 1.0);
    assert!(stats.routed_parameter_grad_norm.is_finite());
    assert!(stats.routed_parameter_grad_norm > 0.0);
    // forward() accepts only the input byte; the target byte is unavailable to the gate.
}

#[test]
fn checkpoint_round_trip_is_complete_and_exact() {
    let mut model = Model::new(16, 0.001, 9);
    model.train_bytes(b"hello hello hello", 1.0);
    let path = temp_path("roundtrip");
    model.save_atomic(&path).unwrap();
    let loaded = Model::load(&path).unwrap();
    assert_eq!(loaded.step, model.step);
    assert_eq!(loaded.config(), model.config());
    assert_eq!(loaded.entropy_ema, model.entropy_ema);
    assert_eq!(loaded.arena.values, model.arena.values);
    assert_eq!(loaded.optimizer, model.optimizer);
    fs::remove_file(path).unwrap();
}

#[test]
fn deterministic_continuation_is_exact() {
    let data = b"abc def! abc def! abc def!";
    let mut uninterrupted = Model::new(8, 0.001, 23);
    for _ in 0..20 {
        uninterrupted.train_bytes(data, 1.0);
    }
    let mut resumed = Model::new(8, 0.001, 23);
    for _ in 0..10 {
        resumed.train_bytes(data, 1.0);
    }
    let path = temp_path("continuation");
    resumed.save_atomic(&path).unwrap();
    resumed = Model::load(&path).unwrap();
    for _ in 0..10 {
        resumed.train_bytes(data, 1.0);
    }
    assert_eq!(resumed.arena.values, uninterrupted.arena.values);
    assert_eq!(resumed.optimizer, uninterrupted.optimizer);
    assert_eq!(resumed.entropy_ema, uninterrupted.entropy_ema);
    fs::remove_file(path).unwrap();
}

#[test]
fn gradient_clipping_limits_applied_norm() {
    let mut model = Model::new(8, 0.01, 31);
    let stats = model.train_bytes(b"ab", 1e-4);
    assert!(stats.grad_norm > 1e-4);
    assert!(
        stats.applied_grad_norm <= 1.001e-4,
        "{}",
        stats.applied_grad_norm
    );
}

#[test]
fn malformed_checkpoints_are_rejected_without_allocating_declared_vectors() {
    let path = temp_path("invalid-parameters");
    let model = Model::new(4, 0.001, 1);
    model.save_atomic(&path).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    // Parameter length follows the fixed v2 header/config, step, and absent EMA.
    let parameter_length_offset = 8 + 8 + 4 + 4 + 4 + 1 + 4 + 4 + 4 + 4 + 8 + 8 + 1 + 8 + 1;
    bytes[parameter_length_offset..parameter_length_offset + 8]
        .copy_from_slice(&u64::MAX.to_le_bytes());
    fs::write(&path, bytes).unwrap();
    assert_eq!(
        Model::load(&path).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    fs::remove_file(&path).unwrap();

    let path = temp_path("invalid-routing");
    let model = Model::new(4, 0.001, 1);
    model.save_atomic(&path).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    // Truncate inside the final route-gate optimizer state.
    bytes.truncate(bytes.len() - 5);
    fs::write(&path, bytes).unwrap();
    assert!(Model::load(&path).is_err());
    fs::remove_file(path).unwrap();
}

#[test]
fn old_checkpoint_version_has_precise_error() {
    let path = temp_path("v1");
    fs::write(&path, MAGIC_V1).unwrap();
    let error = Model::load(&path).unwrap_err();
    assert!(error
        .to_string()
        .contains("unsupported NUETORCH checkpoint version NUETRS01"));
    fs::remove_file(path).unwrap();
}

#[test]
fn learning_smoke_500_steps() {
    let data = b"fn main() { alpha beta; }\nfn test() { beta alpha; }\n";
    let mut model = Model::new(16, 0.001, 7);
    let first = model.train_bytes(data, 1.0);
    let mut last = first;
    for _ in 0..500 {
        last = model.train_bytes(data, 1.0);
        assert!(last.loss.is_finite() && last.grad_norm.is_finite());
    }
    assert!(last.next_byte_loss < first.next_byte_loss);
    assert!(last.next_byte_loss / std::f32::consts::LN_2 < 8.0);
    // The detached target moves downward while the byte model learns. For this
    // smoke corpus, require half-MSE below the documented 0.6 nats^2 bound.
    assert!(last.entropy_loss < 0.6);
    for metric in [
        last.boundary_accuracy,
        last.boundary_precision,
        last.boundary_recall,
    ] {
        assert!(metric.is_finite());
    }
    assert!(last.routed_parameter_grad_norm > 0.0);
}
