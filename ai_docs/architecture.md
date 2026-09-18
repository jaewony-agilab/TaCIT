# Format-Invariant Tactile Policy Learning — Architecture

**This document is the design rationale: what the system is and why it is built this way.**

It contains no hyperparameters, tensor shapes, or generation counts. Those live elsewhere:

| Question | Document |
|---|---|
| Why is it built this way? | **this document** |
| What data exists, and how much? | `data_spec.md` |
| Exact shapes, losses, training loop? | `implementation.md` |

Where any two documents disagree on a number, `implementation.md` wins.

---

## 1. Problem and thesis

### 1.1 What this solves

Zero-shot real-world deployment of a tactile manipulation policy, where the real sensor's **format** — taxel count, spatial layout, surface curvature, measured physical quantity, and calibration — was never seen during training.

This is distinct from cross-embodiment generalization. The invariance target is the *measurement format of tactile data*, not the robot.

### 1.2 Core thesis: bracket, don't match

Tactile sim2real currently fails because the field tries to **match** the real sensor in simulation — gel deformation, marker dynamics, shear response, illumination. That problem is unsolved: FEM/IPC methods are accurate but too slow for RL loops, and penalty-based models are fast but crude.

This design **brackets** instead. Train over a distribution of sensor formats wide enough that the real sensor is an unseen but in-distribution draw. Sim *fidelity* is traded for sim *coverage*, which is a far easier target.

Precedent: GCS (Princeton, 2025) deployed zero-shot on an uncalibrated magnetic sensor off by ~50%, bridged purely by domain randomization. DeXtreme achieved real-world dexterity with no real fine-tuning through DR + ADR alone.

### 1.3 Positioning against prior work

| Prior work | Limitation this design addresses |
|---|---|
| T3, AnyTouch, Sparsh | Treat tactile as an image; require a sensor-specific encoder head per sensor |
| Robot Synesthesia, NeuralFeels | Geometric and sim2real-friendly, but single sensor configuration |
| Bottlenecked Latent Reconstruction, PTLD | Oracle-latent sim2real, but no cross-format generalization |
| TacSL, DIFFTACTILE | Fidelity-first simulation; expensive and sensor-specific |

The whitespace: **geometric tokenization + systematic format randomization + explicit nuisance disentanglement**, evaluated with a true leave-one-format-out split and real zero-shot deployment.

---

## 2. System overview

```
                         tactile readings (any format)
                                    |
                    [ Canonical Tokenizer ]  (no parameters)
                                    |
                          fixed-size token set
                        /                       \
                       /                         \
           [ h-encoder ]                  [ State Encoder ]
                 |                         <- proprioception
                 v                         <- vision
                h  ------ FiLM ------->           |
                                                  v
                                                 z_t
                                                  |
                                [ GRL Adversary ] (train only)
                                                  |
                                     [ Action Decoder ]
                                                  |
                                          action chunk
```

At deployment only the tokenizer, state encoder, and action decoder run. `h` is supplied from the known sensor descriptor.

---

## 3. Canonical Tokenizer

**The core contribution.** It converts any tactile sensor into a permutation-invariant token set.

### 3.1 What a token represents

One token per taxel patch, carrying: the measured contact vector, the patch's position and surface normal in the end-effector frame, a per-channel observation mask, and a link identifier.

Expressing tactile data this way makes **taxel count and layout into data rather than architecture**. A flat 32-taxel pad and a 600-taxel curved skin over the same surface produce token sets the encoder reads identically.

Positions live in the end-effector frame, not the world frame, so contact geometry is invariant to arm pose. Arm pose enters separately through proprioception.

### 3.2 Modality canonicalization

Rather than passing the measured quantity as a label, all sensors are mapped to a common physical quantity at tokenization — force-like contact vectors with a per-channel validity mask for unobserved channels.

This is stronger than a modality label because it **degrades gracefully**. A label-based model hits a cliff on an unseen modality flag; a canonicalized model just sees a contact vector with some channels masked.

**Known approximation error:** depth→force conversion assumes a contact model that is only roughly correct — nonlinear at large deformation, wrong near edges, drifting with gel wear. This is acceptable and intended. The conversion constants are themselves randomized during training, so the encoder never learns to trust them. The training distribution must contain whatever error the real conversion has: the same bracketing logic as §1.2.

### 3.3 Fixed patch budget

Every sensor pools to the **same number of patches**, at train and test alike. A sparse sensor yields mostly-empty patches handled by the mask; a dense sensor pools several taxels per patch.

The encoder therefore sees a constant token count, and format variation lives entirely in patch content and spatial coverage.

**Do not subsample only at deployment.** Applying a downsample to the real sensor but not to training formats produces token-spacing statistics matching no training format — a distribution shift introduced exactly where it hurts most. It would also reduce the contribution to "we preprocess our way out of density variation," which is the assumption the design claims to remove.

**Aggregation must be mean, not max.** Force is an extensive quantity. Max pooling would make a dense sensor read systematically higher than a sparse one on identical contact, reintroducing the density dependence the design exists to eliminate.

---

## 4. h-encoder

Produces `h`, a sensor-characterization latent, from tactile tokens alone.

### 4.1 Role

The h-encoder is trained *by* the sensor descriptor, not in competition with it. During training the descriptor is known and serves as a supervision label. At deployment `h` normally comes from the descriptor instead, through a conditional prior.

The h-encoder is therefore **optional for the base system**. It earns its place by enabling **online inference of `h`** when the descriptor is unknown, wrong, or drifting — gel wear, decalibration. Build it as the upgrade, not the baseline.

### 4.2 Why the prior must be conditional

A fixed N(0, I) prior with `h` zeroed at test creates a knife-edge: strong KL causes posterior collapse, weak KL means `h ≠ 0` at test and zeroing becomes a distribution shift.

Conditioning the prior on the descriptor sidesteps both, and handles an unseen sensor by descriptor alone — no data from that sensor required.

### 4.3 Why attraction must be regression, not classification

Classifying hardware IDs gives `h` a `K`-point structure with no meaningful metric between clusters; an unseen sensor has nowhere well-defined to land. Regressing continuous properties gives a space that interpolates.

---

## 5. State Encoder

Consumes tactile tokens, proprioception, and vision; conditioned on `h`; produces `z_t`.

**Target semantics for `z_t`:** *"contact at this location, this magnitude, this shear direction"* — a physical contact state carrying no trace of which sensor produced it.

### 5.1 Conditioning: FiLM, not concatenation

Concatenating a single conditioning vector onto a long token sequence gets attention-diluted. FiLM applies the conditioning at every layer and ablates cleanly — neutral scale and shift disables it exactly, which makes the "does conditioning matter" experiment unambiguous.

### 5.2 Vision backbone is frozen

A frozen self-supervised backbone removes a source of sim-overfitting. Since only a projection layer trains on vision, vision data quantity matters far less than tactile. What matters is that the renders have natural-image statistics close enough for the frozen features to be meaningful.

### 5.3 Two outputs

The encoder emits both a pooled vector and the full sequence. The adversary operates on the pooled vector, because that is what must be format-agnostic in aggregate. The decoder receives the full sequence, because it needs spatial detail.

---

## 6. Action Decoder

ACT-style chunked action prediction, cross-attending to the encoder sequence, with one lightweight output head per embodiment.

### 6.1 Per-embodiment heads do not break the zero-shot claim

A per-embodiment head requires data from that embodiment. That would be fatal if the zero-shot axis were the robot — but it is the *sensor format*. Every embodiment deployed on exists in simulation, so its head is trained on sim data with no real data required.

This is why the design does not need an embodiment-agnostic action space (fingertip deltas resolved by IK). That machinery would only be required if the zero-shot target were an unseen robot.

### 6.2 Why chunks are long

Heavier noise and latency randomization makes short chunks jittery: each replan sees a differently-corrupted signal. Longer chunks smooth this. This is a temporal-smoothing consideration, not a capacity one.

---

## 7. GRL Adversary (training only)

Predicts the sensor descriptor from `z_t`, through a gradient reversal layer, so the encoder is trained to make that prediction fail.

### 7.1 Why format diversity alone is insufficient

Diversity makes format information **useless** but not **absent**. Two distinct failure paths:

1. **Nothing pushes it out.** Encoding format features costs nothing, so they persist as unused-but-encoded dimensions. Harmless in-distribution. At deployment the real sensor produces values in those dimensions that no training format produced — the decoder was never trained in that region and its behavior there is undefined.

2. **Format information is often genuinely useful.** A sparse pad and a dense skin reporting the same force norm warrant different actions. The task loss therefore actively *rewards* retaining format information in `z_t`. This is not laziness to be trained away; it is correctness, and diversity cannot remove a feature the objective wants.

The adversary converts "unused" into "absent." Without it, `h` is pure KL cost with no benefit and collapses.

### 7.2 The adversary must stay strong

A weak discriminator makes the reversal gradient noise. It needs a higher learning rate and no weight decay relative to the encoder.

Adversarial pressure must also ramp from zero — applying it from step 0 destabilizes the encoder before it has learned anything.

---

## 8. Information routing — what goes where

The design decision most likely to be gotten wrong.

| Information | Destination | Rationale |
|---|---|---|
| Per-taxel position, surface normal | **Token features** | Required to localize contact. **Not nuisance** — routing these to the descriptor and stripping them would delete *where* the object is being touched. |
| Measured quantity (depth vs. force vs. marker) | **Canonicalized away** at tokenization | Stronger than a label; no unseen-modality cliff. |
| Gain error, nonlinearity, saturation, noise, latency | **Descriptor** | Residual calibration uncertainty that cannot be canonicalized. |
| Global layout summary: count, density, spacing, curvature | **Descriptor** | Nuisance at the summary level; helps the encoder normalize. |

### 8.1 The disambiguation test

For any quantity in doubt:

> Would two different sensors observing the **identical physical contact** disagree on this value?
> - **Yes** → it belongs in the descriptor.
> - **No** (it is a property of the contact, not the sensor) → it belongs in `z_t` and must **not** be adversarially removed.

### 8.2 Division of labor

- **Tokens** tell the encoder *where the sensors are*.
- **The descriptor** tells it *how to read them*.
- **`z_t`** comes out sensor-agnostic.

---

## 9. Build order

Three stages, gated by config flags in a single codebase:

1. **Baseline** — tokenizer + state encoder + decoder, conditioned directly on the descriptor. No h-encoder, no adversary.
2. **Add the adversary.**
3. **Add the h-encoder** and the stochastic latent.

Stage 1 must train and fit before anything else is enabled, and remains a permanent baseline. If Stage 1 already achieves zero-shot real deployment, that is the cleaner paper and the extra machinery becomes ablation rather than contribution.

---

## 10. Scale discipline

**Do not scale the model up to absorb randomization.** The intuition that heavy randomization needs a bigger model is right for a *fidelity* framing (memorize every case) and wrong for a *bracketing* framing (learn the invariant).

Capacity is how the invariance fails. Given enough parameters, the encoder will memorize per-format solutions — a lookup table over training configurations — rather than learning a format-invariant representation. In-distribution metrics look excellent; held-out format performance falls off a cliff.

Diagnose before scaling:

| Symptom | Diagnosis | Action |
|---|---|---|
| Both seen and held-out formats mediocre, loss plateaus high | genuine underfitting | scale the **state encoder**; leave the decoder alone |
| Seen formats good, held-out poor | memorization | widen randomization or shrink the encoder |

The second is far more likely.

**Task scope.** Single-task per model is the correct scoping decision, not a compromise. The claim is format-invariance; one task across many formats isolates it, whereas multi-task adds a confound — did transfer come from format-invariance or task diversity?

---

## 11. Deployment concept

Only the state encoder and action decoder run.

1. Write the real sensor's descriptor.
2. Obtain `h` from the conditional prior.
3. Tokenize real readings through the canonical tokenizer, using measured taxel positions in the end-effector frame.
4. Run the policy.

**Optional online inference:** obtain `h` from the h-encoder instead. Useful when the descriptor is unknown, wrong, or drifting. This is the h-encoder's justification.

---

## 12. Evaluation intent

Operational detail belongs to the eval repo. The design commitments it must test:

1. **Leave-one-format-out.** Held-out sensor configurations, including one denser than anything trained on. Produces the generalization curve.
2. **Disentanglement.** A linear probe on frozen `z_t` predicting the sensor descriptor should sit near chance. This is the figure that *proves* the mechanism rather than asserting it.
3. **Real zero-shot deployment.** Two sensors of different modality is the strongest available result — the vision↔non-vision gap is the thinnest part of the literature.
4. **Degradation robustness.** Deploy, then physically occlude taxels or use a worn gel. Directly demonstrates what format-robustness buys.
5. **Ablation grid:** attraction × repulsion. The neither-cell is the diversity-only hypothesis — a legitimate and publishable negative result if it holds.

### 12.1 The metric that must be tracked during training

**Sensor-format predictability from `z_t`.** If it stays near chance, the mechanism is working. If it climbs while training loss looks healthy, the model is contaminated with good-looking curves — the silent failure mode.

This is a training-time diagnostic requiring no simulator. See `implementation.md` §7.2.

### 12.2 The reviewer question to prepare for

> *Why not just simulate your one real sensor accurately?*

Answer with the bracketing argument (§1.2), supported by an explicit baseline: a carefully calibrated single-sensor sim (TacSL-style) versus the randomized-format model, both evaluated zero-shot on the real rig.

---

## 13. Open design risks

| Risk | Mitigation |
|---|---|
| Posterior collapse of `h` | conditional prior + attraction loss; monitor per-dimension KL |
| Adversarial instability | ramp the reversal coefficient; fall back to an MI upper bound |
| `h` absorbs task or object structure instead of format | randomize task and object independently of format (`data_spec.md`) |
| Canonicalization error dominates | randomize conversion constants; validate against real sensor readings early |
| Contamination with healthy-looking curves | track §12.1 from the first run |
| Overfitting to training formats | keep the model small; widen randomization before widening the network |
