# Adaptive Spectral Operator Cost Experiment

This experiment tests the broader NUERONCE training strategy in which dense, low-rank, FFT, DCT, and canonical RFT operators compete under the same byte-language-model objective.

The goal is not to force the model into one transform. The goal is to measure which operator gives the best quality, memory, and compute tradeoff.

## Sandbox setup

- two random seeds: 3 and 11;
- 80 training steps per variant;
- batch size 8;
- sequence length 64 bytes;
- identical corpus, optimizer, learning rate, recurrent backbone, and output head;
- one CPU thread;
- fixed hidden dimension 48;
- validation over 12 held-out batches;
- AdamW optimizer;
- adaptive variant uses hard Gumbel-softmax routing among all five experts.

This is a controlled pilot rather than a full 35M or 355M run.

## Mean results across two seeds

| Variant | Trainable params | Parameter MB | Fixed buffer MB | Adam state MB | Tokens/s | Validation BPB |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 48,256 | 0.1841 | 0.0000 | 0.3682 | 60,006.6 | **1.8195** |
| Low-rank | 40,192 | 0.1533 | 0.0000 | 0.3067 | **64,854.2** | 2.0906 |
| FFT | 39,208 | 0.1496 | 0.0000 | 0.2991 | 61,730.8 | 2.2046 |
| DCT | 39,208 | 0.1496 | 0.0088 | 0.2991 | 63,431.0 | **2.1212** |
| RFT | 39,208 | 0.1496 | 0.0176 | 0.2991 | 58,914.3 | 2.2165 |
| Adaptive mixture | 50,157 | 0.1915 | 0.0264 | 0.3825 | 45,956.2 | 2.1757 |

Lower BPB is better. Higher tokens/s is better.

## Relative computational cost

Using dense as the reference:

| Variant | Parameter reduction | Adam-state reduction | Throughput change | BPB degradation |
|---|---:|---:|---:|---:|
| Low-rank | 16.7% | 16.7% | **+8.1%** | +14.9% |
| FFT | 18.7% | 18.8% | +2.9% | +21.2% |
| DCT | 18.7% | 18.8% | **+5.7%** | +16.6% |
| RFT | 18.7% | 18.8% | -1.8% | +21.8% |
| Adaptive | -3.9% | -3.9% | -23.4% | +19.6% |

## Router behavior

The adaptive model selected experts at these average rates:

| Expert | Selection share |
|---|---:|
| Dense | 48.52% |
| Low-rank | 18.05% |
| DCT | 12.73% |
| FFT | 11.72% |
| RFT | 8.98% |

The router preferred dense nearly half the time. Among the compressed paths, it preferred low-rank first, then DCT, then FFT, and selected RFT least often.

## Conclusions

1. **Dense remains the best quality baseline.**
   It achieved the lowest held-out BPB.

2. **Low-rank is the best current efficiency path.**
   It reduced parameters and optimizer state by 16.7% while increasing throughput by 8.1%. Its quality loss is still too large for frontier replacement, but it is the strongest residual expert candidate.

3. **DCT is the best current fixed spectral basis.**
   It was faster and more accurate than FFT and RFT in this pilot, consistent with the earlier checkpoint-compression experiment.

4. **RFT is not currently competitive as a generic FFN replacement.**
   It reduced parameter and optimizer storage, but was slower than dense and had the worst validation BPB among the fixed transformed variants.

5. **The soft discovery mixture is too expensive for production training.**
   Computing all experts before hard selection reduced throughput by 23.4% and increased model and optimizer storage. Expert competition should therefore happen in short discovery runs or isolated layer ablations, followed by architectural pruning.

## Frontier training policy

The evidence supports this ordering:

- keep dense paths as the quality anchor;
- add low-rank residuals first;
- use DCT for trained-weight compression and selected FFN layers;
- use FFT for sequence/convolution workloads where its fast kernel is useful;
- reserve RFT for memory, quasi-periodic, phase-sensitive, or long-structured tasks where it can demonstrate a task-specific advantage;
- do not execute all experts simultaneously in the final model.

## Next required experiment

Run per-layer operator discovery rather than whole-model mixtures:

1. begin from a trained dense checkpoint;
2. freeze the model;
3. replace one layer at a time with low-rank, DCT, FFT, or RFT alternatives;
4. fine-tune only the replacement and a zero-initialized residual gate;
5. measure validation BPB, tokens/s, optimizer bytes, and activation bytes;
6. keep the cheapest operator within 5% of the dense layer's validation quality;
7. restore dense for layers where no alternative passes.

This produces an adaptive architecture without paying the runtime cost of executing every expert.

## Scientific boundary

These results are measured sandbox pilot results. They establish cost direction and operator ranking for a small recurrent byte model. They do not yet establish the final layer assignments for the 35M or 355M NUERONCE checkpoints.
