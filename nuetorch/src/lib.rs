use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC_V1: &[u8; 8] = b"NUETRS01";
const MAGIC: &[u8; 8] = b"NUETRS02";
const VOCAB: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slice {
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Slice {
    pub fn len(self) -> usize {
        self.rows * self.cols
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Default)]
pub struct Arena {
    pub values: Vec<f32>,
    pub grads: Vec<f32>,
}

impl Arena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc(&mut self, rows: usize, cols: usize) -> Slice {
        let out = Slice {
            offset: self.values.len(),
            rows,
            cols,
        };
        let len = rows.checked_mul(cols).expect("tensor dimensions overflow");
        self.values.resize(self.values.len() + len, 0.0);
        self.grads.resize(self.grads.len() + len, 0.0);
        out
    }

    pub fn get(&self, s: Slice) -> &[f32] {
        &self.values[s.offset..s.offset + s.len()]
    }
    pub fn grad_mut(&mut self, s: Slice) -> &mut [f32] {
        &mut self.grads[s.offset..s.offset + s.len()]
    }
    pub fn zero_grad(&mut self) {
        self.grads.fill(0.0);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EntropyConfig {
    pub global_threshold: f32,
    /// Absolute change in entropy nats required for an entropy boundary.
    pub relative_threshold: f32,
    pub ema_decay: Option<f32>,
    pub entropy_loss_weight: f32,
    pub boundary_loss_weight: f32,
    pub boundary_positive_weight: f32,
    pub boundary_probability_threshold: f32,
    pub patch_min: usize,
    pub patch_max: usize,
    pub routing_enabled: bool,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            global_threshold: 0.75,
            relative_threshold: 0.10,
            ema_decay: Some(0.95),
            entropy_loss_weight: 0.1,
            boundary_loss_weight: 0.1,
            boundary_positive_weight: 2.0,
            boundary_probability_threshold: 0.5,
            patch_min: 2,
            patch_max: 16,
            routing_enabled: true,
        }
    }
}

impl EntropyConfig {
    pub fn validate(&self) -> Result<(), String> {
        fn probability(name: &str, x: f32) -> Result<(), String> {
            if x.is_finite() && (0.0..=1.0).contains(&x) {
                Ok(())
            } else {
                Err(format!("{name} must be finite and in [0, 1]"))
            }
        }
        probability("global threshold", self.global_threshold)?;
        probability(
            "boundary probability threshold",
            self.boundary_probability_threshold,
        )?;
        if !self.relative_threshold.is_finite() || self.relative_threshold < 0.0 {
            return Err("relative threshold must be finite and nonnegative".into());
        }
        if let Some(decay) = self.ema_decay {
            probability("EMA decay", decay)?;
        }
        for (name, x) in [
            ("entropy loss weight", self.entropy_loss_weight),
            ("boundary loss weight", self.boundary_loss_weight),
            ("boundary positive weight", self.boundary_positive_weight),
        ] {
            if !x.is_finite() || x < 0.0 {
                return Err(format!("{name} must be finite and nonnegative"));
            }
        }
        if self.patch_min == 0 {
            return Err("patch minimum must be positive".into());
        }
        if self.patch_max < self.patch_min {
            return Err("patch maximum must be at least patch minimum".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub width: usize,
    pub lr: f32,
    pub entropy: EntropyConfig,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 {
            return Err("width must be positive".into());
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        self.entropy.validate()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FactorState {
    pub row: Vec<f32>,
    pub col: Vec<f32>,
}

impl FactorState {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            row: vec![0.0; rows],
            col: vec![0.0; cols],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PatchStats {
    pub patch_count: usize,
    pub average_patch_length: f32,
    pub minimum_patch_length: usize,
    pub maximum_patch_length: usize,
    pub forced_boundary_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchResult {
    pub ranges: Vec<std::ops::Range<usize>>,
    pub forced_boundary_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    /// Combined weighted objective.
    pub loss: f32,
    pub next_byte_loss: f32,
    pub entropy_loss: f32,
    pub boundary_loss: f32,
    pub tokens: usize,
    /// Gradient norm before clipping.
    pub grad_norm: f32,
    pub applied_grad_norm: f32,
    pub entropy_nats: f32,
    pub normalized_entropy: f32,
    pub entropy_bits: f32,
    pub predicted_entropy_nats: f32,
    pub boundary_accuracy: f32,
    pub boundary_precision: f32,
    pub boundary_recall: f32,
    pub predicted_boundary_rate: f32,
    pub target_boundary_rate: f32,
    pub patch_count: usize,
    pub average_patch_length: f32,
    pub minimum_patch_length: usize,
    pub maximum_patch_length: usize,
    pub forced_boundary_count: usize,
    pub average_gate: f32,
    pub minimum_gate: f32,
    pub maximum_gate: f32,
    pub gates_above_half_percent: f32,
    pub routed_parameter_grad_norm: f32,
}

#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[derive(Debug)]
pub struct Model {
    pub width: usize,
    pub arena: Arena,
    embed: Slice,
    output: Slice,
    bias: Slice,
    entropy_weight: Slice,
    entropy_bias: Slice,
    boundary_weight: Slice,
    boundary_bias: Slice,
    route_scale: Slice,
    /// [predicted normalized entropy coefficient, boundary probability coefficient, bias].
    route_gate: Slice,
    pub optimizer: Vec<FactorState>,
    pub step: u64,
    pub lr: f32,
    pub entropy_config: EntropyConfig,
    pub entropy_ema: Option<f32>,
    hidden: Vec<f32>,
    routed_hidden: Vec<f32>,
    hidden_grad: Vec<f32>,
    routed_hidden_grad: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
}

impl Model {
    pub fn new(width: usize, lr: f32, seed: u64) -> Self {
        Self::with_config(
            ModelConfig {
                width,
                lr,
                entropy: EntropyConfig::default(),
            },
            seed,
        )
    }

    pub fn with_config(config: ModelConfig, seed: u64) -> Self {
        config.validate().expect("invalid model configuration");
        let width = config.width;
        let mut arena = Arena::new();
        let embed = arena.alloc(VOCAB, width);
        let output = arena.alloc(width, VOCAB);
        let bias = arena.alloc(1, VOCAB);
        let entropy_weight = arena.alloc(1, width);
        let entropy_bias = arena.alloc(1, 1);
        let boundary_weight = arena.alloc(1, width);
        let boundary_bias = arena.alloc(1, 1);
        let route_scale = arena.alloc(1, width);
        let route_gate = arena.alloc(1, 3);
        let slices = [
            embed,
            output,
            bias,
            entropy_weight,
            entropy_bias,
            boundary_weight,
            boundary_bias,
            route_scale,
            route_gate,
        ];
        let optimizer = slices
            .iter()
            .map(|s| FactorState::new(s.rows, s.cols))
            .collect();
        let mut rng = Rng::new(seed);
        let scale = (1.0 / width as f32).sqrt();
        for v in &mut arena.values {
            *v = (rng.f32() * 2.0 - 1.0) * scale;
        }
        arena.values[bias.offset..bias.offset + bias.len()].fill(0.0);
        arena.values[entropy_bias.offset] = entropy_uniform_nats();
        arena.values[boundary_bias.offset] = -2.0;
        arena.values[route_scale.offset..route_scale.offset + width].fill(0.05);
        arena.values[route_gate.offset..route_gate.offset + 3].copy_from_slice(&[1.0, 1.0, -1.0]);
        Self {
            width,
            arena,
            embed,
            output,
            bias,
            entropy_weight,
            entropy_bias,
            boundary_weight,
            boundary_bias,
            route_scale,
            route_gate,
            optimizer,
            step: 0,
            lr: config.lr,
            entropy_config: config.entropy,
            entropy_ema: None,
            hidden: vec![0.0; width],
            routed_hidden: vec![0.0; width],
            hidden_grad: vec![0.0; width],
            routed_hidden_grad: vec![0.0; width],
            logits: vec![0.0; VOCAB],
            probs: vec![0.0; VOCAB],
        }
    }

    fn slices(&self) -> [Slice; 9] {
        [
            self.embed,
            self.output,
            self.bias,
            self.entropy_weight,
            self.entropy_bias,
            self.boundary_weight,
            self.boundary_bias,
            self.route_scale,
            self.route_gate,
        ]
    }

    pub fn config(&self) -> ModelConfig {
        ModelConfig {
            width: self.width,
            lr: self.lr,
            entropy: self.entropy_config.clone(),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.arena.values.len()
    }

    /// Parameter, gradient, optimizer-factor, and reusable activation tensors only.
    pub fn resident_tensor_bytes(&self) -> usize {
        let parameters_and_grads = 2 * self.arena.values.len() * size_of::<f32>();
        let factors: usize = self
            .optimizer
            .iter()
            .map(|s| (s.row.len() + s.col.len()) * size_of::<f32>())
            .sum();
        let activations = (self.hidden.len()
            + self.routed_hidden.len()
            + self.hidden_grad.len()
            + self.routed_hidden_grad.len()
            + self.logits.len()
            + self.probs.len())
            * size_of::<f32>();
        parameters_and_grads + factors + activations
    }

    pub fn predict_logits(&mut self, byte: u8, route_enabled: bool) -> Vec<f32> {
        self.forward(byte as usize, route_enabled);
        self.logits.clone()
    }

    fn forward(&mut self, x: usize, route_enabled: bool) -> (f32, f32, f32) {
        self.hidden.copy_from_slice(
            &self.arena.values
                [self.embed.offset + x * self.width..self.embed.offset + (x + 1) * self.width],
        );
        let entropy_prediction = dot(&self.hidden, self.arena.get(self.entropy_weight))
            + self.arena.values[self.entropy_bias.offset];
        let boundary_logit = dot(&self.hidden, self.arena.get(self.boundary_weight))
            + self.arena.values[self.boundary_bias.offset];
        let boundary_probability = sigmoid(boundary_logit);
        let predicted_normalized = (entropy_prediction / entropy_uniform_nats()).clamp(0.0, 1.0);
        let gate_input = self.arena.values[self.route_gate.offset] * predicted_normalized
            + self.arena.values[self.route_gate.offset + 1] * boundary_probability
            + self.arena.values[self.route_gate.offset + 2];
        let gate = if route_enabled {
            sigmoid(gate_input)
        } else {
            0.0
        };
        for i in 0..self.width {
            self.routed_hidden[i] =
                self.hidden[i] * (1.0 + gate * self.arena.values[self.route_scale.offset + i]);
        }
        for j in 0..VOCAB {
            let mut z = self.arena.values[self.bias.offset + j];
            for (i, &h) in self.routed_hidden.iter().enumerate() {
                z += h * self.arena.values[self.output.offset + i * VOCAB + j];
            }
            self.logits[j] = z;
        }
        softmax(&self.logits, &mut self.probs);
        (entropy_prediction, boundary_probability, gate)
    }

    pub fn train_bytes(&mut self, bytes: &[u8], max_grad_norm: f32) -> StepStats {
        assert!(bytes.len() >= 2);
        assert!(max_grad_norm.is_finite() && max_grad_norm >= 0.0);
        self.arena.zero_grad();
        let token_count = bytes.len() - 1;
        let inv = 1.0 / token_count as f32;
        let mut next_loss = 0.0;
        let mut entropy_loss = 0.0;
        let mut boundary_loss = 0.0;
        let mut entropy_sum = 0.0;
        let mut predicted_entropy_sum = 0.0;
        let mut previous_target_entropy = None;
        let mut previous_predicted_entropy = None;
        let mut boundary_probabilities = Vec::with_capacity(token_count);
        let mut tp = 0usize;
        let mut tn = 0usize;
        let mut fp = 0usize;
        let mut fn_count = 0usize;
        let mut target_boundaries = 0usize;
        let mut gate_sum = 0.0;
        let mut gate_min = f32::INFINITY;
        let mut gate_max = f32::NEG_INFINITY;
        let mut gates_above_half = 0usize;

        for pair in bytes.windows(2) {
            let x = pair[0] as usize;
            let y = pair[1] as usize;
            let (entropy_prediction, boundary_probability, gate) =
                self.forward(x, self.entropy_config.routing_enabled);
            let target_entropy = distribution_entropy_nats(&self.probs);
            let entropy_target = entropy_boundary_target(
                target_entropy,
                previous_target_entropy,
                &self.entropy_config,
            );
            let boundary_target = syntax_boundary(pair[1]) || entropy_target;
            let target = f32::from(boundary_target);
            let predicted_boundary =
                boundary_probability >= self.entropy_config.boundary_probability_threshold;
            match (predicted_boundary, boundary_target) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_count += 1,
                (false, false) => tn += 1,
            }
            target_boundaries += usize::from(boundary_target);
            boundary_probabilities.push(boundary_probability);
            entropy_sum += target_entropy;
            predicted_entropy_sum += entropy_prediction;
            gate_sum += gate;
            gate_min = gate_min.min(gate);
            gate_max = gate_max.max(gate);
            gates_above_half += usize::from(gate > 0.5);

            next_loss -= self.probs[y].max(1e-12).ln();
            let entropy_error = entropy_prediction - target_entropy;
            entropy_loss += 0.5 * entropy_error * entropy_error;
            let positive_weight = if boundary_target {
                self.entropy_config.boundary_positive_weight
            } else {
                1.0
            };
            boundary_loss -= positive_weight
                * (target * boundary_probability.max(1e-7).ln()
                    + (1.0 - target) * (1.0 - boundary_probability).max(1e-7).ln());

            self.routed_hidden_grad.fill(0.0);
            for j in 0..VOCAB {
                let dz = (self.probs[j] - usize::from(j == y) as f32) * inv;
                self.arena.grads[self.bias.offset + j] += dz;
                for (i, grad) in self.routed_hidden_grad.iter_mut().enumerate() {
                    let oi = self.output.offset + i * VOCAB + j;
                    self.arena.grads[oi] += self.routed_hidden[i] * dz;
                    *grad += self.arena.values[oi] * dz;
                }
            }

            self.hidden_grad.fill(0.0);
            let mut entropy_prediction_grad =
                self.entropy_config.entropy_loss_weight * entropy_error * inv;
            let boundary_scale = if boundary_target {
                self.entropy_config.boundary_positive_weight
            } else {
                1.0
            };
            let mut boundary_logit_grad = self.entropy_config.boundary_loss_weight
                * boundary_scale
                * (boundary_probability - target)
                * inv;
            if self.entropy_config.routing_enabled {
                let mut gate_grad = 0.0;
                for i in 0..self.width {
                    let scale = self.arena.values[self.route_scale.offset + i];
                    self.hidden_grad[i] += self.routed_hidden_grad[i] * (1.0 + gate * scale);
                    self.arena.grads[self.route_scale.offset + i] +=
                        self.routed_hidden_grad[i] * self.hidden[i] * gate;
                    gate_grad += self.routed_hidden_grad[i] * self.hidden[i] * scale;
                }
                let gate_input_grad = gate_grad * gate * (1.0 - gate);
                let predicted_normalized =
                    (entropy_prediction / entropy_uniform_nats()).clamp(0.0, 1.0);
                self.arena.grads[self.route_gate.offset] += gate_input_grad * predicted_normalized;
                self.arena.grads[self.route_gate.offset + 1] +=
                    gate_input_grad * boundary_probability;
                self.arena.grads[self.route_gate.offset + 2] += gate_input_grad;
                if entropy_prediction > 0.0 && entropy_prediction < entropy_uniform_nats() {
                    entropy_prediction_grad += gate_input_grad
                        * self.arena.values[self.route_gate.offset]
                        / entropy_uniform_nats();
                }
                boundary_logit_grad += gate_input_grad
                    * self.arena.values[self.route_gate.offset + 1]
                    * boundary_probability
                    * (1.0 - boundary_probability);
            } else {
                self.hidden_grad.copy_from_slice(&self.routed_hidden_grad);
            }
            for (i, hidden_gradient) in self.hidden_grad.iter_mut().enumerate() {
                self.arena.grads[self.entropy_weight.offset + i] +=
                    entropy_prediction_grad * self.hidden[i];
                *hidden_gradient +=
                    entropy_prediction_grad * self.arena.values[self.entropy_weight.offset + i];
                self.arena.grads[self.boundary_weight.offset + i] +=
                    boundary_logit_grad * self.hidden[i];
                *hidden_gradient +=
                    boundary_logit_grad * self.arena.values[self.boundary_weight.offset + i];
                self.arena.grads[self.embed.offset + x * self.width + i] += *hidden_gradient;
            }
            self.arena.grads[self.entropy_bias.offset] += entropy_prediction_grad;
            self.arena.grads[self.boundary_bias.offset] += boundary_logit_grad;
            let _ = predicted_entropy_boundary(
                entropy_prediction,
                previous_predicted_entropy,
                self.entropy_ema,
                &self.entropy_config,
            );
            previous_target_entropy = Some(target_entropy);
            previous_predicted_entropy = Some(entropy_prediction);
            self.update_entropy_ema(entropy_prediction);
        }

        next_loss *= inv;
        entropy_loss *= inv;
        boundary_loss *= inv;
        let total_loss = next_loss
            + self.entropy_config.entropy_loss_weight * entropy_loss
            + self.entropy_config.boundary_loss_weight * boundary_loss;
        let grad_norm = l2_norm(&self.arena.grads);
        let clip = if grad_norm == 0.0 {
            1.0
        } else {
            (max_grad_norm / (grad_norm + 1e-12)).min(1.0)
        };
        if clip < 1.0 {
            for g in &mut self.arena.grads {
                *g *= clip;
            }
        }
        let applied_grad_norm = l2_norm(&self.arena.grads);
        let routed_parameter_grad_norm = self.routed_grad_norm();
        self.step += 1;
        for (index, slice) in self.slices().into_iter().enumerate() {
            self.factor_update(slice, index);
        }
        let patches = dynamic_patches(
            &boundary_probabilities,
            self.entropy_config.boundary_probability_threshold,
            self.entropy_config.patch_min,
            self.entropy_config.patch_max,
        );
        let patch_stats = patch_statistics(&patches);
        let entropy_nats = entropy_sum * inv;
        StepStats {
            loss: total_loss,
            next_byte_loss: next_loss,
            entropy_loss,
            boundary_loss,
            tokens: token_count,
            grad_norm,
            applied_grad_norm,
            entropy_nats,
            normalized_entropy: (entropy_nats / entropy_uniform_nats()).clamp(0.0, 1.0),
            entropy_bits: entropy_nats / std::f32::consts::LN_2,
            predicted_entropy_nats: predicted_entropy_sum * inv,
            boundary_accuracy: safe_ratio(tp + tn, token_count),
            boundary_precision: safe_ratio(tp, tp + fp),
            boundary_recall: safe_ratio(tp, tp + fn_count),
            predicted_boundary_rate: safe_ratio(tp + fp, token_count),
            target_boundary_rate: safe_ratio(target_boundaries, token_count),
            patch_count: patch_stats.patch_count,
            average_patch_length: patch_stats.average_patch_length,
            minimum_patch_length: patch_stats.minimum_patch_length,
            maximum_patch_length: patch_stats.maximum_patch_length,
            forced_boundary_count: patch_stats.forced_boundary_count,
            average_gate: gate_sum * inv,
            minimum_gate: gate_min,
            maximum_gate: gate_max,
            gates_above_half_percent: 100.0 * safe_ratio(gates_above_half, token_count),
            routed_parameter_grad_norm,
        }
    }

    fn update_entropy_ema(&mut self, prediction: f32) {
        if let Some(decay) = self.entropy_config.ema_decay {
            self.entropy_ema = Some(match self.entropy_ema {
                Some(old) => decay * old + (1.0 - decay) * prediction,
                None => prediction,
            });
        }
    }

    fn routed_grad_norm(&self) -> f32 {
        let mut sum = 0.0f64;
        for slice in [self.route_scale, self.route_gate] {
            for &g in &self.arena.grads[slice.offset..slice.offset + slice.len()] {
                sum += f64::from(g) * f64::from(g);
            }
        }
        sum.sqrt() as f32
    }

    fn factor_update(&mut self, slice: Slice, state_index: usize) {
        let beta = 0.999f32;
        let eps = 1e-8f32;
        let correction = (1.0 - beta.powi(self.step.min(i32::MAX as u64) as i32)).max(eps);
        let state = &mut self.optimizer[state_index];
        for row in 0..slice.rows {
            let sum: f32 = (0..slice.cols)
                .map(|column| {
                    let g = self.arena.grads[slice.offset + row * slice.cols + column];
                    g * g
                })
                .sum();
            state.row[row] = beta * state.row[row] + (1.0 - beta) * sum / slice.cols as f32;
        }
        for column in 0..slice.cols {
            let sum: f32 = (0..slice.rows)
                .map(|row| {
                    let g = self.arena.grads[slice.offset + row * slice.cols + column];
                    g * g
                })
                .sum();
            state.col[column] = beta * state.col[column] + (1.0 - beta) * sum / slice.rows as f32;
        }
        let row_mean = state.row.iter().sum::<f32>() / state.row.len() as f32;
        for row in 0..slice.rows {
            for column in 0..slice.cols {
                let i = slice.offset + row * slice.cols + column;
                let denominator = ((state.row[row] / correction)
                    * (state.col[column] / correction)
                    / (row_mean / correction).max(eps))
                .sqrt()
                    + eps;
                self.arena.values[i] -= self.lr * self.arena.grads[i] / denominator;
            }
        }
    }

    pub fn save_atomic(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let temporary = path.with_extension(format!("tmp-{stamp}"));
        let result = (|| {
            let mut file = File::create(&temporary)?;
            file.write_all(MAGIC)?;
            write_config(&mut file, &self.config())?;
            write_u64(&mut file, self.step)?;
            write_option_f32(&mut file, self.entropy_ema)?;
            write_vec(&mut file, &self.arena.values)?;
            write_u64(&mut file, self.optimizer.len() as u64)?;
            for state in &self.optimizer {
                write_vec(&mut file, &state.row)?;
                write_vec(&mut file, &state.col)?;
            }
            file.sync_all()?;
            fs::rename(&temporary, path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_length = file.metadata()?.len();
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic == MAGIC_V1 {
            return Err(invalid_data(
                "unsupported NUETORCH checkpoint version NUETRS01; version NUETRS02 is required",
            ));
        }
        if &magic != MAGIC {
            return Err(invalid_data(
                "bad NUETORCH checkpoint magic or unsupported version",
            ));
        }
        let config = read_config(&mut file)?;
        config
            .validate()
            .map_err(|e| invalid_data(&format!("invalid checkpoint configuration: {e}")))?;
        let step = read_u64(&mut file)?;
        let entropy_ema = read_option_f32(&mut file)?;
        if entropy_ema.is_some_and(|x| !x.is_finite()) {
            return Err(invalid_data("non-finite entropy EMA"));
        }
        let parameter_count = config
            .width
            .checked_mul(2 * VOCAB + 3)
            .and_then(|value| value.checked_add(VOCAB + 5))
            .ok_or_else(|| invalid_data("checkpoint parameter layout overflows usize"))?;
        let parameter_bytes = parameter_count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| invalid_data("checkpoint parameter byte count overflows usize"))?;
        if u64::try_from(parameter_bytes).unwrap_or(u64::MAX) > file_length {
            return Err(invalid_data(
                "checkpoint width requires more parameter bytes than the file contains",
            ));
        }
        let mut model = Self::with_config(config, 1);
        model.arena.values = read_vec(&mut file, model.parameter_count(), "model parameters")?;
        model.arena.grads.resize(model.arena.values.len(), 0.0);
        let count = read_len(&mut file, "optimizer state count")?;
        if count != model.optimizer.len() {
            return Err(invalid_data("optimizer layout mismatch"));
        }
        for (index, state) in model.optimizer.iter_mut().enumerate() {
            state.row = read_vec(
                &mut file,
                state.row.len(),
                &format!("optimizer row state {index}"),
            )?;
            state.col = read_vec(
                &mut file,
                state.col.len(),
                &format!("optimizer column state {index}"),
            )?;
        }
        let mut trailing = [0];
        if file.read(&mut trailing)? != 0 {
            return Err(invalid_data("trailing checkpoint data"));
        }
        model.step = step;
        model.entropy_ema = entropy_ema;
        Ok(model)
    }
}

pub fn distribution_entropy_nats(probabilities: &[f32]) -> f32 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

pub fn normalized_entropy(probabilities: &[f32]) -> f32 {
    (distribution_entropy_nats(probabilities) / entropy_uniform_nats()).clamp(0.0, 1.0)
}

pub fn entropy_bits(probabilities: &[f32]) -> f32 {
    distribution_entropy_nats(probabilities) / std::f32::consts::LN_2
}

pub fn entropy_uniform_nats() -> f32 {
    (VOCAB as f32).ln()
}

pub fn syntax_boundary(byte: u8) -> bool {
    byte.is_ascii_whitespace()
        || matches!(
            byte,
            b'.' | b','
                | b';'
                | b':'
                | b'!'
                | b'?'
                | b'('
                | b')'
                | b'['
                | b']'
                | b'{'
                | b'}'
                | b'<'
                | b'>'
                | b'`'
                | b'\''
                | b'"'
                | b'/'
                | b'\\'
        )
}

pub fn entropy_boundary_target(
    current_nats: f32,
    previous_nats: Option<f32>,
    config: &EntropyConfig,
) -> bool {
    current_nats / entropy_uniform_nats() >= config.global_threshold
        && previous_nats
            .is_some_and(|previous| (current_nats - previous).abs() >= config.relative_threshold)
}

pub fn predicted_entropy_boundary(
    current_nats: f32,
    previous_nats: Option<f32>,
    ema: Option<f32>,
    config: &EntropyConfig,
) -> bool {
    let reference = if config.ema_decay.is_some() {
        ema.or(previous_nats)
    } else {
        previous_nats
    };
    current_nats / entropy_uniform_nats() >= config.global_threshold
        && reference
            .is_some_and(|previous| (current_nats - previous).abs() >= config.relative_threshold)
}

pub fn combined_boundary_target(byte: u8, entropy_target: bool) -> bool {
    syntax_boundary(byte) || entropy_target
}

pub fn dynamic_patches(
    probabilities: &[f32],
    threshold: f32,
    minimum: usize,
    maximum: usize,
) -> PatchResult {
    assert!(minimum > 0 && maximum >= minimum);
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut forced = 0;
    for (index, &probability) in probabilities.iter().enumerate() {
        let length = index + 1 - start;
        let at_maximum = length >= maximum;
        let predicted = length >= minimum && probability >= threshold;
        if at_maximum || predicted {
            ranges.push(start..index + 1);
            start = index + 1;
            forced += usize::from(at_maximum && !predicted);
        }
    }
    if start < probabilities.len() {
        ranges.push(start..probabilities.len());
    }
    PatchResult {
        ranges,
        forced_boundary_count: forced,
    }
}

pub fn patch_statistics(result: &PatchResult) -> PatchStats {
    if result.ranges.is_empty() {
        return PatchStats::default();
    }
    let lengths: Vec<usize> = result.ranges.iter().map(|range| range.len()).collect();
    PatchStats {
        patch_count: lengths.len(),
        average_patch_length: lengths.iter().sum::<usize>() as f32 / lengths.len() as f32,
        minimum_patch_length: *lengths.iter().min().unwrap_or(&0),
        maximum_patch_length: *lengths.iter().max().unwrap_or(&0),
        forced_boundary_count: result.forced_boundary_count,
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
fn softmax(logits: &[f32], probabilities: &mut [f32]) {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0;
    for (probability, &logit) in probabilities.iter_mut().zip(logits) {
        *probability = (logit - maximum).exp();
        denominator += *probability;
    }
    for probability in probabilities {
        *probability /= denominator;
    }
}
fn l2_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt() as f32
}
fn safe_ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}
fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn write_u8(writer: &mut impl Write, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}
fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut b = [0];
    reader.read_exact(&mut b)?;
    Ok(b[0])
}
fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
fn write_f32(writer: &mut impl Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}
fn write_bool(writer: &mut impl Write, value: bool) -> io::Result<()> {
    write_u8(writer, u8::from(value))
}
fn read_bool(reader: &mut impl Read, name: &str) -> io::Result<bool> {
    match read_u8(reader)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid_data(&format!("invalid {name} boolean"))),
    }
}
fn write_option_f32(writer: &mut impl Write, value: Option<f32>) -> io::Result<()> {
    write_bool(writer, value.is_some())?;
    if let Some(x) = value {
        write_f32(writer, x)?;
    }
    Ok(())
}
fn read_option_f32(reader: &mut impl Read) -> io::Result<Option<f32>> {
    if read_bool(reader, "optional f32")? {
        Ok(Some(read_f32(reader)?))
    } else {
        Ok(None)
    }
}
fn read_len(reader: &mut impl Read, name: &str) -> io::Result<usize> {
    usize::try_from(read_u64(reader)?)
        .map_err(|_| invalid_data(&format!("{name} does not fit usize")))
}
fn write_vec(writer: &mut impl Write, values: &[f32]) -> io::Result<()> {
    write_u64(writer, values.len() as u64)?;
    for &value in values {
        write_f32(writer, value)?;
    }
    Ok(())
}
fn read_vec(reader: &mut impl Read, expected: usize, name: &str) -> io::Result<Vec<f32>> {
    let length = read_len(reader, name)?;
    if length != expected {
        return Err(invalid_data(&format!(
            "{name} length mismatch: expected {expected}, got {length}"
        )));
    }
    let byte_count = length
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid_data(&format!("{name} is too large")))?;
    let mut bytes = vec![0; byte_count];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four-byte chunk")))
        .collect())
}
fn write_config(writer: &mut impl Write, config: &ModelConfig) -> io::Result<()> {
    write_u64(writer, config.width as u64)?;
    write_f32(writer, config.lr)?;
    let e = &config.entropy;
    write_f32(writer, e.global_threshold)?;
    write_f32(writer, e.relative_threshold)?;
    write_option_f32(writer, e.ema_decay)?;
    write_f32(writer, e.entropy_loss_weight)?;
    write_f32(writer, e.boundary_loss_weight)?;
    write_f32(writer, e.boundary_positive_weight)?;
    write_f32(writer, e.boundary_probability_threshold)?;
    write_u64(writer, e.patch_min as u64)?;
    write_u64(writer, e.patch_max as u64)?;
    write_bool(writer, e.routing_enabled)
}
fn read_config(reader: &mut impl Read) -> io::Result<ModelConfig> {
    Ok(ModelConfig {
        width: read_len(reader, "width")?,
        lr: read_f32(reader)?,
        entropy: EntropyConfig {
            global_threshold: read_f32(reader)?,
            relative_threshold: read_f32(reader)?,
            ema_decay: read_option_f32(reader)?,
            entropy_loss_weight: read_f32(reader)?,
            boundary_loss_weight: read_f32(reader)?,
            boundary_positive_weight: read_f32(reader)?,
            boundary_probability_threshold: read_f32(reader)?,
            patch_min: read_len(reader, "patch minimum")?,
            patch_max: read_len(reader, "patch maximum")?,
            routing_enabled: read_bool(reader, "routing enabled")?,
        },
    })
}

#[cfg(test)]
mod tests;
