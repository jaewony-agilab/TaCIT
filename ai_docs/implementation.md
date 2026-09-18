# Training Implementation Specification

**Companion to:** `architecture.md` (design rationale)
**Scope:** the **training codebase only**. Model implementation, training loop, validation, logging, checkpointing.

## Repository boundaries

Three separate codebases. This document specifies the second.

| Repo | Responsibility | Interface |
|---|---|---|
| `tactile-datagen` | Sim rollouts, rendering, format sampling, writing episodes | **Produces** the on-disk contract in §2 |
| **`tactile-train`** | **Model, training loop, validation, checkpoints** | **Consumes** §2, **produces** checkpoints per §8 |
| `tactile-eval` | Sim and real rollouts, success metrics, deployment | Consumes checkpoints per §8 |

This repo **never** generates data, renders, or runs a simulator. It reads a directory and trains. If a run needs different data, that is a datagen change, not a change here.

Validation in this repo is **loss-based on a held-out split**, not rollout success. Rollout evaluation belongs to `tactile-eval`.

---

## 1. Conventions

### 1.1 Coordinate frame

All tactile geometry is expected in the **`eef` frame**: origin at the wrist flange, +Z along the palm normal (out of the palm), +X toward the index finger base, +Y = Z × X.

The training repo assumes this and does not verify it. Datagen owns the convention.

### 1.2 Units

| Quantity | Unit |
|---|---|
| Position | metres |
| Force | newtons |
| Joint angle | radians |
| Control rate | 30 Hz |

### 1.3 Dtypes

| Data | On disk | In model |
|---|---|---|
| Tactile signals | `float32` | `bfloat16` |
| Taxel positions / normals | `float32` | `float32` — kept fp32 through Fourier encoding |
| Validity mask | `uint8` | `bool` |
| Proprioception | `float32` | `bfloat16` |
| Actions | `float32` | `float32` — loss in fp32 |
| Video frames | `uint8` | `bfloat16` after normalization |
| `h_T` | `float32` | `float32` |

Positions stay fp32 because high-frequency Fourier bands are sensitive to bf16 rounding. Loss is fp32 to avoid gradient underflow.

---

## 2. Expected data contract

The loader expects this layout. Anything else is a datagen bug.

```
{data_root}/
  formats/
    {format_id}/
      taxels.npz
      h_T.json
  episodes/
    {episode_id}/
      meta.json
      tactile.npz
      proprio.npz
      actions.npz
      frames/            # or a pointer to a shared render
  splits.json
```

### 2.1 Shapes

`T` = episode length, `N` = raw taxel count, `D_j` = joint DoF, `V` = camera views.

| File | Key | Shape | Dtype |
|---|---|---|---|
| `taxels.npz` | `positions` | `(N, 3)` | f32 |
| | `normals` | `(N, 3)` | f32 |
| | `link_id` | `(N,)` | i16 |
| `tactile.npz` | `signal` | `(T, N, 3)` | f32 |
| | `validity` | `(T, N, 3)` | u8 |
| `proprio.npz` | `q` | `(T, D_j)` | f32 |
| | `qd` | `(T, D_j)` | f32 |
| `actions.npz` | `target_q` | `(T, D_j)` | f32 |
| `frames/` | per-view images | `(T, V, 224, 224, 3)` | u8 |

`signal` is the canonical contact vector `[normal, shear_x, shear_y]` in newtons. `validity` is 1 where a channel is observed.

### 2.2 `meta.json`

```json
{
  "episode_id":    "ep_000123",
  "format_id":     "fmt_017",
  "embodiment_id": "openarm_xhand1",
  "task_id":       "pick_cup_stack",
  "num_frames":    312,
  "num_views":     2
}
```

### 2.3 `h_T.json`

Fixed field order — the model indexes positionally.

```json
{
  "density": 600, "mean_spacing": 0.0021, "coverage_area": 0.00084,
  "curvature": 0.0085, "modality": [1, 0, 0], "gain": [1.02, 0.97, 1.05],
  "nonlinearity": 0.03, "saturation": 12.0, "noise_sigma": 0.015,
  "dropout_rate": 0.04, "latency_frames": 1, "stiffness_k": 2400.0
}
```

Normalization applied at load, with **fixed constants** (not data-derived — a descriptor must normalize identically across runs):

| Field | Dim | Normalization |
|---|---|---|
| `density` | 1 | `log(x) / log(600)` |
| `mean_spacing` | 1 | `x / 0.01` |
| `coverage_area` | 1 | `x / 0.001` |
| `curvature` | 1 | `x / 0.02` |
| `modality` | 3 | pass through |
| `gain` | 3 | `(x - 1.0) / 0.5` |
| `nonlinearity` | 1 | `x / 0.1` |
| `saturation` | 1 | `x / 20.0` |
| `noise_sigma` | 1 | `x / 0.05` |
| `dropout_rate` | 1 | `x / 0.15` |
| `latency_frames` | 1 | `x / 3.0` |
| `stiffness_k` | 1 | `log(x) / log(5000)` |

**`h_T_dim = 16`.**

### 2.4 `splits.json`

Datagen declares the splits. This repo does not construct them.

```json
{
  "train":     ["ep_000001", "..."],
  "val":       ["ep_009001", "..."],
  "val_heldout_format": ["ep_009501", "..."]
}
```

- `train` / `val` — same format distribution. `val` measures ordinary generalization.
- `val_heldout_format` — formats absent from `train`. **This is the split that matters**; it is the loss-based proxy for the zero-shot claim.

If `val_heldout_format` is missing, the trainer warns loudly and continues.

### 2.5 Startup validation

On first load, assert and fail fast:

- every `format_id` in `meta.json` exists under `formats/`
- `tactile.signal.shape[1] == taxels.positions.shape[0]`
- `num_frames` matches array lengths
- no episode id appears in two splits
- `h_T.json` has all 12 keys
- `val_heldout_format` format ids are disjoint from `train` format ids

Fail with the offending episode id. Do not silently skip.

---

## 3. Preprocessing (in-repo, at load time)

### 3.1 Patch pooling

Fixed budget **K = 128**, identical at train and val.

```python
def build_patches(positions, normals, link_id, K=128):
    """Precomputed ONCE per format, cached to disk in {cache_dir}.
    returns centroid_idx (K,), member_idx (K, 16), member_mask (K, 16),
            patch_pos (K, 3), patch_normal (K, 3), patch_link (K,),
            patch_mask (K,)
    """
```

1. **Centroids** — farthest-point sampling over `positions`, `min(N, K)` points. Seed is deterministic per `format_id` so layout is stable across epochs and runs.
2. **`N < K`** — use all N as centroids, pad to K with `patch_mask = False`. Never duplicate taxels.
3. **Grouping** — ball query, radius `2 × mean_spacing`, max 16 members. Every taxel is assigned to its nearest centroid at minimum, so none are dropped.
4. **Per-batch aggregation:**
   - `patch_signal` = **mean** over valid members
   - `patch_validity` = channel-wise OR over members
   - `patch_pos` = centroid position
   - `patch_normal` = normalized mean of member normals

**Mean, not max.** Force is extensive; max pooling would make a 600-taxel sensor read systematically higher than a 128-taxel one on identical contact — reintroducing the density dependence the design removes.

Steps 1–3 cache to `{cache_dir}/{format_id}.npz`. Only step 4 runs per batch.

### 3.2 Temporal window

Tactile enters as a causal history window, `W = 8` frames (~0.27 s), stacked along the feature dim. Pad with frame repetition at episode start.

### 3.3 Action normalization

Computed once per embodiment over `train` only, cached, and **written into every checkpoint**:

```python
action_norm[embodiment_id] = {"mean": (D_j,), "std": (D_j,)}  # std clamped >= 1e-3
```

A checkpoint without these is not deployable.

---

## 4. Modules

| Symbol | Value |
|---|---|
| `K` | 128 patches |
| `W` | 8 frames |
| `d_tok` | 256 |
| `d_model` | 512 |
| `d_h` | 32 |
| `h_T_dim` | 16 |
| `H` | 100 action chunk |

### 4.1 Tokenizer

```
fourier(p): L = 8 bands, p scaled by 1/0.1 -> 48 dims

token_in = concat([
    patch_signal (W stacked),   # 3 * 8 = 24
    patch_validity,             # 3
    fourier(patch_pos),         # 48
    patch_normal,               # 3
    link_embed,                 # 16 learned
])                              # = 94

Linear(94 -> 256) -> LayerNorm -> GELU -> Linear(256 -> 256)
out: (B, K, 256)
```

Empty patches zeroed after embedding, masked in all downstream attention.

### 4.2 h-encoder

```
in:  (B, K, 256), patch_mask (B, K)
     4-layer transformer, d=256, 4 heads, FFN 1024, pre-LN
     masked mean-pool -> Linear(256 -> 64) -> split
out: mu (B, 32), logvar (B, 32)
```

`h = mu + eps·exp(0.5·logvar)` at train; `h = mu` at eval.

**Conditional prior:** `Linear(16 -> 128) -> GELU -> Linear(128 -> 64) -> split`.
KL is closed-form between two diagonal Gaussians; no sampling.

**Attraction head:** `Linear(32 -> 128) -> GELU -> Linear(128 -> 16)`.

### 4.3 State encoder

```
tactile (B, K, 256) -> Linear(256 -> 512)
proprio (B, 2*D_j)  -> MLP -> (B, 2, 512)
vision  (B, V, 16, 512)   [§4.4]

seq = concat([tactile, proprio, vision], dim=1)
seq += type_embed     # 3 learned types
seq += view_embed     # per view, vision tokens only

8 layers, d=512, 8 heads, FFN 2048, pre-LN, dropout 0.1
```

**FiLM on `h`**, one generator per layer, applied after each sublayer's LayerNorm and before the sublayer op, on **both** attention and FFN sublayers:

```python
gamma, beta = film_gen_l(h)                        # each (B, 512)
x = gamma[:, None, :] * layernorm(x) + beta[:, None, :]
```

`film_gen_l = Linear(32 -> 1024)`, weight zero-init, bias `[1...1, 0...0]`. An untrained FiLM is then an exact no-op, and the ablation `gamma=1, beta=0` disables conditioning cleanly.

Outputs **both**:
- `z_t = masked_mean_pool(seq)` → `(B, 512)`, feeds the adversary and the probe
- `z_seq = seq` → `(B, L, 512)`, feeds the decoder

The adversary sees the pooled vector because that is what must be format-agnostic in aggregate; the decoder needs the full sequence for spatial detail.

### 4.4 Vision backbone

**DINOv3 ViT-S/16, frozen.**

```
(B, V, 3, 224, 224) -> frozen forward -> (B, V, 196, 384)
Linear(384 -> 512)                                 [trainable]
attention pool 196 -> 16 learned queries           [trainable]
out: (B, V, 16, 512)
```

Learned query-attention pooling, not average pooling — averaging discards spatial layout.

**View dropout** p=0.1 per view during training.

### 4.5 Action decoder

```
queries: (H=100, 512) learned positional embeddings
7 layers: self-attn(queries) -> cross-attn(z_seq) -> FFN
d=512, 8 heads, FFN 2048, pre-LN, dropout 0.1
head: Linear(512 -> D_j)     per embodiment, nn.ModuleDict
out: (B, 100, D_j) normalized joint targets
```

`H = 100` rather than ACT's 50 — heavier noise and latency randomization makes short chunks jittery.

### 4.6 Adversary

```
z_t -> GradientReversal(lambda)
Linear(512 -> 256) -> GELU -> Dropout(0.1)
Linear(256 -> 256) -> GELU
Linear(256 -> 16)
```

```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None
```

---

## 5. Losses

| Loss | Formula | Weight | Schedule |
|---|---|---|---|
| `L_action` | L1(pred, target), fp32 | 1.0 | constant |
| `L_attract` | MSE(attract_head(h), h_T_norm) | — | 1.0 for 5k steps, then 0.1 |
| `L_adv` | MSE(adversary(GRL(z_t)), h_T_norm) | 1.0 | λ ramps 0 → 0.3 |
| `L_kl` | KL(q(h·) ‖ p(h·h_T)) | 1e-4 | linear warmup over 10k steps |

```
L_total = L_action + w_attract(step)·L_attract + L_adv + beta_kl·L_kl
```

`L_adv` keeps weight 1.0; **the adversarial knob is λ inside the GRL**, not the loss weight. This keeps the discriminator training at full strength while the encoder's adversarial pressure ramps independently.

```python
def grl_lambda(step, ramp=10_000, lam_max=0.3):
    p = min(step / ramp, 1.0)
    return lam_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)
```

`beta_kl = 1e-4` is deliberately small — the conditional prior already constrains `h`, and a large KL causes collapse.

**Fallback:** if GRL oscillates, swap `L_adv` for a CLUB mutual-information upper bound behind a config flag. Do not replace GRL outright.

---

## 6. Training loop

### 6.1 Optimizer

```python
AdamW([
  {"params": encoder_params,   "lr": 1e-4, "weight_decay": 1e-4},
  {"params": decoder_params,   "lr": 1e-4, "weight_decay": 1e-4},
  {"params": adversary_params, "lr": 3e-4, "weight_decay": 0.0},
], betas=(0.9, 0.95))
```

The adversary gets 3× lr and no weight decay so it stays a meaningful opponent — a weak discriminator makes the reversal gradient noise.

| Setting | Value |
|---|---|
| Schedule | cosine to 1e-6, 2000 step linear warmup |
| Precision | bf16 autocast, fp32 master weights |
| Grad clip | 1.0 global norm |
| Steps | 200k |
| Batch | 96 (4090) / 256 (A6000) |

### 6.2 Batch sampler

**Hard requirement: ≥8 distinct formats per batch.**

```python
class FormatBalancedSampler(Sampler):
    """Partitions each batch into G >= 8 groups.
    Each group draws from a single format; formats sampled
    without replacement within a batch.
    Embodiment sampled per group, not per batch."""
```

Shuffle is not sufficient. One format per batch gives the adversary no within-batch variation and a useless gradient. Per-group embodiment sampling keeps every per-embodiment head receiving gradient.

### 6.3 Step

```python
for step in range(total_steps):
    batch = next(train_loader)
    lam, beta, w_att = grl_lambda(step), kl_warmup(step), w_attract(step)

    tokens        = tokenizer(batch)
    mu, logvar    = h_encoder(tokens, batch.patch_mask)
    h             = reparameterize(mu, logvar) if cfg.use_h_encoder else batch.h_T_norm
    p_mu, p_lv    = prior_net(batch.h_T_norm)

    z_t, z_seq    = state_encoder(tokens, batch.proprio, batch.vision, h)
    pred          = decoder(z_seq, batch.embodiment_id)

    L_action  = l1(pred, batch.actions_norm)
    L_attract = mse(attract_head(h), batch.h_T_norm)  if cfg.use_h_encoder else 0
    L_adv     = mse(adversary(grl(z_t, lam)), batch.h_T_norm) if cfg.use_adversary else 0
    L_kl      = kl_gaussian(mu, logvar, p_mu, p_lv)   if cfg.use_h_encoder else 0

    loss = L_action + w_att*L_attract + L_adv + beta*L_kl
    loss.backward()
    clip_grad_norm_(params, 1.0)
    opt.step(); sched.step(); opt.zero_grad()
```

Single backward pass — GRL handles the sign flip, so no alternating optimization.

### 6.4 Stage flags

| Stage | `use_h_encoder` | `use_adversary` | `h` source |
|---|---|---|---|
| 1 baseline | False | False | `h_T_norm` direct into FiLM |
| 2 + adversary | False | True | `h_T_norm` direct |
| 3 full | True | True | sampled from `q(h·obs)` |

Stage 1 must train and fit before anything else is enabled, and remains a permanent baseline.

---

## 7. Validation and logging

### 7.1 Validation

Every **2000 steps**, on both `val` and `val_heldout_format`:

| Metric | Note |
|---|---|
| `L_action` | primary; the gap between the two splits is the headline number |
| `L_attract`, `L_kl` | diagnostic |
| Per-timestep L1 across the chunk | reveals whether late-chunk predictions degrade |
| Per-format L1 breakdown | identifies which formats are hard |

Cap at 200 batches per split so validation stays under ~60 s.

### 7.2 Contamination probe — every 2000 steps

**The single most important diagnostic.** Ridge probe on frozen `z_t` predicting `h_T`:

```python
def contamination_probe(z_t_cache, h_T_cache, alpha=1.0):
    """Ridge, 5-fold CV, on a fixed 2000-sample val slice.
    Returns per-field R^2 and mean R^2. Baseline (predict mean) = 0."""
```

- **Near 0** → `z_t` is format-agnostic; mechanism working.
- **Climbing while `L_action` looks healthy** → the silent failure mode. Everything downstream is measuring the wrong thing.

Build this in from day one. It needs no simulator.

### 7.3 wandb

**Every step:** `L_action`, `L_attract`, `L_adv`, `L_kl`, `lambda`, `beta_kl`, `w_attract`, grad norm, lr per group, samples/sec.

**Every 500 steps:**

| Metric | Why |
|---|---|
| Per-dimension KL | detects posterior collapse — dims pinned at ~0 are dead |
| `‖gamma - 1‖`, `‖beta‖` per FiLM layer | is conditioning used, or still a no-op |
| Adversary R² per `h_T` field | which properties leak most |
| Fraction of empty patches per format | pooling sanity |
| Formats per batch (assert ≥8) | sampler sanity |

**Every 2000 steps:** everything in §7.1 and §7.2, plus a `z_t` t-SNE colored by format id (should show no format clustering).

**Config:** log the full resolved config, git SHA, and dataset root hash at run start.

### 7.4 Checkpointing

Every **5000 steps**. Keep last 3 rolling + best by `val_heldout_format` `L_action`.

```python
{
  "model_state", "optimizer_state", "scheduler_state", "step", "epoch",
  "action_norm",          # per embodiment, required for deployment
  "h_T_norm_constants",   # §2.3 table
  "format_registry",      # format_id -> descriptor
  "config",               # full resolved
  "git_sha",
  "rng_state",            # torch, numpy, python
}
```

A checkpoint missing `action_norm`, `h_T_norm_constants`, or `format_registry` is not deployable. Assert on save.

**Resume** restores optimizer, scheduler, step, RNG, and dataloader epoch. A resumed run must be bit-comparable to an uninterrupted one on the same seed — worth an actual test.

---

## 8. Checkpoint interface for `tactile-eval`

The eval repo needs, and this repo guarantees:

- `model_state` loadable by the same module definitions at the pinned `git_sha`
- `action_norm` for denormalizing predictions
- `h_T_norm_constants` so a real sensor's descriptor normalizes identically
- `format_registry` to check whether a deploy format was seen in training
- `config` to reconstruct the model

Eval computes `h` from a descriptor via `prior_net`, or from `h_encoder` if inferring online. Both paths are in `model_state`.

---

## 9. Repository layout

```
tactile-train/
  src/
    data/
      dataset.py        # episode loading, window assembly, split handling
      contract.py       # §2.5 startup validation
      sampler.py        # FormatBalancedSampler
      pooling.py        # FPS + ball query, cached per format
      normalize.py      # h_T + action normalization constants
    models/
      tokenizer.py
      h_encoder.py      # + prior_net, attract_head
      state_encoder.py  # + FiLM
      decoder.py        # + per-embodiment heads
      adversary.py      # + GradientReversal
      vision.py         # frozen DINOv3 + attention pool
    train/
      loop.py
      losses.py
      schedules.py      # grl_lambda, kl_warmup, w_attract
      validate.py       # §7.1
      probe.py          # §7.2
      checkpoint.py     # §7.4, §8
  configs/
    stage1_baseline.yaml
    stage2_adversary.yaml
    stage3_full.yaml
  tests/
    test_contract.py    # synthetic episodes, malformed cases
    test_pooling.py     # N < K, N > K, determinism across runs
    test_resume.py      # bit-comparability
```

**Tests matter more than usual here.** `test_pooling.py` in particular — a nondeterministic FPS seed would silently change patch layout between epochs and quietly destroy the representation.

---

## 10. Failure modes

| Symptom | Cause | Response |
|---|---|---|
| Probe R² > ~0.3 | `z_t` contaminated | raise `lam_max` to 0.5; assert ≥8 formats/batch |
| `L_adv` oscillates, `L_action` spikes | GRL instability | lower `lam_max`, lengthen ramp, or switch to CLUB |
| Per-dim KL → 0 everywhere | posterior collapse | lower `beta_kl`, raise `w_attract` |
| FiLM `gamma ≈ 1, beta ≈ 0` at 50k | `h` unused | check `h` actually varies across the batch |
| `val_heldout_format` ≫ worse than `val` | memorization | a datagen issue — widen randomization, not model size |
| Both splits poor | underfitting | scale **state encoder** (8→12 layers, d 512→768). Leave decoder alone. |

---

## 11. Open hyperparameters

| Item | Default | Note |
|---|---|---|
| `K` patch budget | 128 | sweep {64, 128, 256} early |
| `W` history window | 8 | sweep {4, 8, 16} if slip behavior is poor |
| `d_h` | 32 | should exceed `h_T_dim = 16`; unlikely to need tuning |
| `lam_max` | 0.3 | tune against the §7.2 probe |
