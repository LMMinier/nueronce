# H2 — Typed-Channel Specialization Probe (Lane G)

- seed: 0 | model: 119,621 params | steps: 600
- channels: sem, str, goal, evid, unc, auth, proc
- each channel has a *fixed* retention timescale (unc=0.90 fastest … auth=0.999 slowest); the probe asks whether training makes tasks depend on different channels.

## Base bits/byte per task

- local: 3.4737
- recall: 3.1065
- count: 0.9674

## Relative loss increase when each channel is ablated

| channel | local | recall | count |
|---|---|---|---|
| sem | +0.000 | +0.000 | +0.000 |
| str | +0.000 | +0.000 | +0.000 |
| goal | +0.000 | +0.003 | -0.000 |
| evid | +0.000 | +0.001 | -0.000 |
| unc | -0.000 | +0.002 | -0.000 |
| auth | -0.000 | +0.002 | -0.000 |
| proc | +0.000 | -0.001 | -0.000 |

## Verdict

- max *single-channel* relative ablation effect: 0.003
- *all-channels-off* control (total typed-memory contribution): local +0.000, recall +0.035, count -0.009
- distinct 'most-hurt task' across channels: 2 of 3
- **H2 supported: False**

Reading: the all-channels-off control confirms the ablation is real and bounds the typed memory's total contribution; a single-channel effect far below that total means the channels are a *diffuse, redundant pool*, not specialized roles — the dense read-out redistributes across whatever channels survive. That is the precise sense in which H2 is unsupported here.

Per the concurrent plan's specialization-claim requirement, H2 counts as supported only if ablating some channel materially hurts (>10% relative) and channels are not interchangeable (different tasks depend on different channels). This is a tiny-scale probe; a negative result here falsifies H2 *at this scale/training*, not universally — but per the plan's rule, a hypothesis without supporting evidence is recorded as unsupported until shown otherwise.
