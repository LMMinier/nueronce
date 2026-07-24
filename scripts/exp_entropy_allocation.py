"""EXPERIMENT: is next-byte entropy a valid compute-allocation signal, and does
caveat #3 (entropy wastes compute on *irreducible* uncertainty) actually bite?

This tests the shared premise of the entropy-patching hypothesis stack on REAL
text, at small (CPU) scale. It does NOT claim to be the 35M A/B — it measures
signal *relationships* that decide whether the downstream hypotheses are worth
GPU time. Every number printed is measured, with random and oracle baselines
so nothing is graded on a curve.

Design (one real trained byte model, disjoint held-out text):
  Signals available AT INFERENCE for each position t (predicting byte t+1):
    H_t  = predictive entropy (the entropy-patching signal)
    S_t  = syntactic boundary (the CURRENT patcher's signal: word onset)
  Ground truth (NOT available at inference):
    L_t  = actual next-byte loss  (where the model is really wrong)

  Q1  Does H_t predict L_t better than S_t? (is entropy a better difficulty signal)
  Q2  Allocation quality: if you place K% of "compute boundaries" by a signal,
      what fraction of total held-out loss lands on them? Compare
      entropy vs syntax vs random(floor) vs oracle(ceiling).
  Q3  Caveat #3: train weak->strong, split each position's uncertainty into
      REDUCIBLE (dropped with more training = epistemic) vs IRREDUCIBLE
      (residual = aleatoric). Does H_t correlate with the reducible part
      (good) or the irreducible part (wasted compute)?
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from nueronce.adaptive_patching import ByteEntropyHead, predictive_entropy
from nueronce.segment import syntax_table

SEED = 0
SEQ = 256
BATCH = 16
WEAK_STEPS = 150
STRONG_STEPS = 600
CORPUS = Path("origin.txt")          # real English prose (Darwin), ~950KB
OUT = Path("metrics/entropy_allocation_experiment.json")

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_bytes(path, limit=600_000):
    data = path.read_bytes()[:limit]
    return np.frombuffer(data, dtype=np.uint8).astype(np.int64)


def sample_batch(arr, rng):
    idx = rng.integers(0, len(arr) - SEQ - 1, size=BATCH)
    return torch.tensor(np.stack([arr[i:i + SEQ + 1] for i in idx]))


def train_head(arr, steps, seed):
    rng = np.random.default_rng(seed)
    head = ByteEntropyHead(dim=64, kernel=5, layers=2)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for _ in range(steps):
        b = sample_batch(arr, rng)
        opt.zero_grad()
        loss = head.lm_loss(b)
        loss.backward()
        opt.step()
    head.eval()
    return head, float(loss.item())


@torch.no_grad()
def per_position(head, arr):
    """Over the held-out stream, return aligned arrays for predicting byte t+1:
    H (entropy), L (true-byte loss), S (syntax boundary target)."""
    syn = syntax_table()
    Hs, Ls, Ss = [], [], []
    # non-overlapping windows for a clean, unbiased pass
    for start in range(0, len(arr) - SEQ - 1, SEQ):
        window = torch.tensor(arr[start:start + SEQ + 1])[None]   # [1, SEQ+1]
        logits = head(window[:, :-1])                            # predicts bytes 1..SEQ
        H = predictive_entropy(logits)[0]                        # [SEQ]  H_t for byte t+1
        tgt = window[0, 1:]                                      # [SEQ]  true byte t+1
        L = torch.nn.functional.cross_entropy(
            logits[0].float(), tgt, reduction="none")            # [SEQ]  actual loss
        # syntax boundary: is true byte t+1 a word onset (non-syntax after syntax)?
        prev = window[0, :-1]
        S = (syn[prev] & ~syn[tgt]).float()                      # [SEQ]
        Hs.append(H.numpy()); Ls.append(L.numpy()); Ss.append(S.numpy())
    return np.concatenate(Hs), np.concatenate(Ls), np.concatenate(Ss)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(ra.astype(float), rb.astype(float))


def loss_captured(signal, L, frac):
    """Fraction of total held-out loss that lands on the top-`frac` positions
    ranked by `signal` (higher signal = allocate compute here)."""
    k = max(1, int(len(signal) * frac))
    top = np.argsort(signal)[-k:]
    return float(L[top].sum() / L.sum())


def main():
    arr = load_bytes(CORPUS)
    split = int(len(arr) * 0.85)
    train_arr, held_arr = arr[:split], arr[split:]
    print(f"corpus {CORPUS} | train {len(train_arr)/1e3:.0f}KB | held-out {len(held_arr)/1e3:.0f}KB")

    weak, weak_loss = train_head(train_arr, WEAK_STEPS, seed=1)
    strong, strong_loss = train_head(train_arr, STRONG_STEPS, seed=1)
    print(f"weak head final train loss {weak_loss:.3f} | strong {strong_loss:.3f} (nats/byte)")

    H, L, S = per_position(weak, held_arr)          # entropy patcher uses the (weaker) inference-time model
    _, Ls, _ = per_position(strong, held_arr)       # strong model's residual loss
    print(f"held-out positions: {len(H):,}")
    print(f"held-out mean loss  weak {L.mean():.3f} | strong {Ls.mean():.3f} bpb {L.mean()/math.log(2):.3f}->{Ls.mean()/math.log(2):.3f}")

    # Q1: which signal predicts actual difficulty (L) better?
    q1 = {
        "pearson_entropy_vs_loss": pearson(H, L),
        "spearman_entropy_vs_loss": spearman(H, L),
        "pearson_syntax_vs_loss": pearson(S, L),
        "spearman_syntax_vs_loss": spearman(S, L),
    }

    # Q2: allocation quality at several budgets (fraction of loss captured)
    q2 = {}
    rng = np.random.default_rng(SEED)
    for frac in (0.1, 0.25, 0.5):
        rand = float(np.mean([loss_captured(rng.random(len(L)), L, frac) for _ in range(5)]))
        q2[f"budget_{int(frac*100)}pct"] = {
            "entropy": loss_captured(H, L, frac),
            "syntax": loss_captured(S + rng.random(len(S)) * 1e-6, L, frac),  # break ties randomly
            "random_floor": rand,
            "oracle_ceiling": loss_captured(L, L, frac),
        }

    # Q3: caveat #3 — does entropy target REDUCIBLE (epistemic) or IRREDUCIBLE (aleatoric)?
    reducible = np.clip(L - Ls, 0, None)   # loss the extra training removed = epistemic
    irreducible = Ls                       # residual after more training = aleatoric floor
    q3 = {
        "pearson_entropy_vs_reducible": pearson(H, reducible),
        "pearson_entropy_vs_irreducible": pearson(H, irreducible),
        "spearman_entropy_vs_reducible": spearman(H, reducible),
        "spearman_entropy_vs_irreducible": spearman(H, irreducible),
        "mean_reducible": float(reducible.mean()),
        "mean_irreducible": float(irreducible.mean()),
    }

    report = {"config": {"corpus": str(CORPUS), "seq": SEQ, "weak_steps": WEAK_STEPS,
                         "strong_steps": STRONG_STEPS, "held_positions": int(len(H))},
              "Q1_signal_predicts_difficulty": q1,
              "Q2_allocation_quality": q2,
              "Q3_epistemic_vs_aleatoric": q3}

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print("\n===== RESULTS (all measured on held-out, real text) =====")
    print("\nQ1  which signal predicts where the model is actually wrong?")
    print(f"  entropy->loss  Pearson {q1['pearson_entropy_vs_loss']:+.3f}  Spearman {q1['spearman_entropy_vs_loss']:+.3f}")
    print(f"  syntax ->loss  Pearson {q1['pearson_syntax_vs_loss']:+.3f}  Spearman {q1['spearman_syntax_vs_loss']:+.3f}")
    print("\nQ2  fraction of total held-out LOSS captured by top-K% positions:")
    print(f"  {'budget':>8} {'entropy':>8} {'syntax':>8} {'random':>8} {'oracle':>8}")
    for k, v in q2.items():
        print(f"  {k:>8} {v['entropy']:8.3f} {v['syntax']:8.3f} {v['random_floor']:8.3f} {v['oracle_ceiling']:8.3f}")
    print("\nQ3  caveat #3 — does entropy chase reducible (good) or irreducible (wasted)?")
    print(f"  entropy->REDUCIBLE (epistemic, good) Pearson {q3['pearson_entropy_vs_reducible']:+.3f}")
    print(f"  entropy->IRREDUCIBLE (aleatoric,bad) Pearson {q3['pearson_entropy_vs_irreducible']:+.3f}")
    print(f"  (mean reducible {q3['mean_reducible']:.3f} vs irreducible {q3['mean_irreducible']:.3f} nats)")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
