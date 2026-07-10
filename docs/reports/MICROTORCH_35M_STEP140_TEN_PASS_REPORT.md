# Nueronce Engine NUERONCE 35M — Ten-Pass Training Report

- Steps: **40 → 140** (100 updates)
- Mix: **80 pretraining + 20 SFT**
- Targets: **40,880 pretraining bytes + 1,192 SFT bytes**
- Finite gradients: **100/100 steps, 177/177 tensors each**
- Held-out BPB: **5.4935 → 4.6437** (15.47% lower)
- Held-out accuracy: **14.02% → 18.88%**

## Pass trend

| Pass | Step | BPB | Accuracy | Δ BPB |
|---:|---:|---:|---:|---:|
| 1 | 50 | 5.2667 | 15.51% | -0.2268 |
| 2 | 60 | 5.1109 | 15.57% | -0.1558 |
| 3 | 70 | 5.0465 | 14.16% | -0.0644 |
| 4 | 80 | 4.9436 | 16.32% | -0.1029 |
| 5 | 90 | 4.9014 | 16.32% | -0.0422 |
| 6 | 100 | 4.8592 | 17.60% | -0.0422 |
| 7 | 110 | 4.8119 | 16.62% | -0.0472 |
| 8 | 120 | 4.7537 | 18.28% | -0.0582 |
| 9 | 130 | 4.7186 | 18.75% | -0.0351 |
| 10 | 140 | 4.6437 | 18.88% | -0.0749 |

## Final checkpoint

- SHA-256: `066ae348b4fc0cd28b096eb3d296e889f53f9a808db726e8163615fbe9f6ee0c`
- Canonical state SHA-256: `8a9c5c5b12f4fede8e8c9c6f6ba207e31f506ed1dc58ea52aa6a06bd66d233b4`
- Bytes: `138262296`
- Optimizer step: `140`

## Honest status

Held-out BPB and next-byte accuracy improved materially, but greedy samples remain repetitive and unreadable. The checkpoint is learning byte distributions but is not yet a usable language or instruction model.
