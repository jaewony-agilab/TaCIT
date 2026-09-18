# Data Specification

**This document specifies what data to generate: which embodiments, tasks, sensor formats, how much, rendered how, from which viewpoints.**

It does not specify the on-disk file format — that is the training repo's contract. Generation must conform to it.

| Question | Document |
|---|---|
| Why is it built this way? | `architecture.md` |
| What data exists, and how much? | **this document** |
| On-disk layout, shapes, dtypes, field schemas | `implementation.md` §2 |

**Generation must match `implementation.md` §2 exactly.** That section is the interface; this one is the content.

---

## 0. Fixed decisions

| Item | Decision |
|---|---|
| Physical deploy target | OpenArm + XHand1 |
| Primary task | Pick cup from stack |
| Training embodiments | 3 (see §1) |
| Discrete sensor formats | ~40 (see §3) |
| Render | photorealistic, scene-randomized |

**The deploy embodiment trains in simulation.** The zero-shot axis is *sensor format*, not embodiment, and the per-embodiment action head requires sim data from that robot. What is held out for OpenArm + XHand1 is the **sensor format**, not the platform.

---

## 1. Embodiments

| Role | Platform |
|---|---|
| Train + physical deploy | **OpenArm + XHand1** |
| Train | Wuji hand |
| Train | Ludi Dex3 hand |
| Optional 4th | Fourier 6dof + GR1 |

**Selection criterion:** maximize variation in *fingertip and palm surface geometry*, not kinematics. Surfaces are what taxels are placed on.

### 1.1 Why no embodiment is held out

Cross-embodiment transfer is not the claim. Additional embodiments exist to **widen format diversity** — more surface geometries to place taxels on. This is a data-scaling argument, not a transfer claim.

**Optional secondary result:** if budget allows, train a 4th embodiment's head and evaluate cross-embodiment transfer separately. Report as a secondary finding; do not make the paper depend on it.

---

## 2. Tasks

| Priority | Task |
|---|---|
| Primary | Pick cup from stack |
| Optional | 2 additional pinch-grasp tasks |

Single primary task is intentional. Secondary tasks, if added, must stay in the **same contact mode** (pinch grasp). Different contact modes are a separate paper.

---

## 3. Sensor formats

Two layers: a **discrete format set** (~40 named configurations, which the held-out split operates on) and **continuous perturbations** resampled every episode on top.

### 3.1 Discrete axes

| Axis | Values | Count |
|---|---|---|
| Density | 32, 64, 128, 256, 400, 600 taxels | 6 |
| Layout | uniform grid, Poisson-disk, tip-clustered | 3 |
| Curvature | flat pad, mild, full fingertip radius | 3 |
| Native modality | depth, force (3-axis), normal-only | 3 |

Full crossing is 162; sample **~40 combinations**. This keeps episodes-per-format above the floor in §4.3.

**600 is the real XHand1 density** (120 taxels per fingertip × 5). Its taxel positions already exist in sim — use the real layout as the anchor at that density rather than a synthetic one.

### 3.2 Continuous perturbations (per episode)

| Axis | Range |
|---|---|
| Gain error | ±50%, independent per axis |
| Nonlinearity | mild quadratic, randomized coefficient |
| Saturation ceiling | randomized within plausible sensor range |
| Noise | Gaussian, randomized σ |
| Dropout | 0–15% of taxels dead |
| Latency | 0–3 frames |
| Bias drift | slow random walk over the episode |
| Stiffness `k` (depth→force conversion) | randomized, so the encoder never trusts the conversion |

Go wide on these. There is no bound to find.

### 3.3 Bounding the layout axes

Generate all ~40 formats. **Do not filter upfront.**

Very sparse formats may carry too little information to be useful, but this is not knowable before training. After the first multi-format run, break down held-out performance per format. If sparse configurations hurt, trim them then — with evidence rather than a guess.

Expect the sparse end to be where a floor appears: at 32 taxels most patches are empty and the validity mask carries the work.

If a pre-generation filter is wanted, compute it analytically: drop formats where the expected taxel count inside the contact patch during grasp falls below 2–3. Derivable from density and contact patch area, no rollouts needed.

### 3.4 Held-out formats

Declare **before** generation. These populate the `val_heldout_format` split.

- One density **above** the training range (800) — extrapolation check
- One layout × modality combination absent from the training set
- One curvature value between trained values — interpolation check
- **The real XHand1 sensor's exact format combination**

The last is the headline holdout. 600-density configurations appear in training, but **not the specific combination** of 600 × real XHand1 layout × real curvature × real modality. The claim is then precise: the policy saw that density and that layout family, but never the real sensor's exact configuration, on the robot it deploys to.

Keep the real sensor's continuous parameters comfortably *inside* the §3.2 ranges — bracketed, not extrapolated.

---

## 4. Scale

### 4.1 Per task, per embodiment

| Quantity | Count |
|---|---|
| Trajectories | 800 |
| Formats sampled per trajectory | 6 |
| Tactile episodes | 4,800 |
| Videos rendered | 800 |

**Vision renders once per trajectory and is reused across all 6 formats.** Sensor format does not affect the scene. This decouples the render budget from the format budget entirely.

Formats are sampled per trajectory, not fully crossed.

### 4.2 Totals

| Quantity | Count |
|---|---|
| Videos | ~2,400 |
| Video frames (≈300/episode) | ~720k |
| Tactile episodes | ~14,400 |

Weight toward the deploy rig: ~1,200 trajectories for OpenArm + XHand1, ~600 each for Wuji and Ludi Dex3.

Add ~200 trajectories per embodiment on **held-out formats only** for the `val_heldout_format` split.

### 4.3 Coverage floor

**Episodes per format ≥ 100.** At 14,400 episodes across ~40 formats that is ~360 each. If format count grows, episode count must grow with it.

---

## 5. Render

**All photorealistic.**

### 5.1 Scene randomization (per episode)

Photorealism alone transfers *worse* than varied renders — the policy latches onto one lighting and material configuration. Randomize on top:

| Axis | Range |
|---|---|
| Lighting direction | full hemisphere |
| Lighting intensity | wide |
| Color temperature | 2700K–6500K |
| Table / background texture | randomized from a texture set |
| Object albedo, roughness | randomized |
| Camera pose jitter | ±3cm position, ±5° rotation |
| Exposure / white balance | mild shift |

Costs nothing extra at render time — config, not additional frames.

### 5.2 Fast-render diagnostic slice

Generate ~300 paired frames (same scenes, MuJoCo-fast render) purely as a diagnostic: the frozen vision backbone should produce similar embeddings for both render modes, indicating it reads geometry rather than rendering artifacts.

Not part of training data. Store separately.

---

## 6. Camera views

| View | Purpose |
|---|---|
| Wrist-mounted | Primary; close-up of the contact region |
| Static third-person | Scene context, object pose |
| Second static (optional) | Only if the real rig has it |

**Rule: only render views the real rig actually has.** A view present in sim but absent at deployment is a train/test mismatch.

Since the deploy rig is OpenArm + XHand1, its real camera setup determines the view set for **all** embodiments. Match sim camera intrinsics and mounting pose to the physical rig before bulk generation — a wrist camera 5cm off from the real one is a systematic gap no amount of scene randomization fixes.

---

## 7. Pairing

**Re-render the same trajectory under multiple formats.** This produces exact format-controlled pairs: identical physical contact, different sensor readings.

Why it matters: the adversary's job is to detect format information in the latent. Paired data isolates format as the only varying factor, making that signal far cleaner.

Keep ~15% unpaired (independently generated trajectories) so the model does not overfit to the paired structure.

---

## 8. Independence requirements

Sample independently of one another:

- sensor format
- object identity, pose, mass, friction
- embodiment
- scene randomization

**If format correlates with embodiment** (hand A always gets layout A), the model cannot separate them and the disentanglement probe measures something unintended. Put every layout on every hand.

**If format correlates with task or object**, the sensor latent absorbs task structure instead of format structure.

---

## 9. Object and scene variation

| Axis | Range |
|---|---|
| Object pose | full reachable workspace |
| Object mass | ±40% |
| Friction coefficient | 0.3–1.2 |
| Object scale | ±15% |
| Stack height / neighbor count | 2–5 |

---

## 10. Splits

Generation declares the splits; the training repo does not construct them. Emit `splits.json` per `implementation.md` §2.4:

| Split | Content |
|---|---|
| `train` | training formats |
| `val` | same format distribution as `train`, disjoint episodes |
| `val_heldout_format` | §3.4 formats only, absent from `train` |

Held-out format ids must be **disjoint** from training format ids. The training repo asserts this at startup.

---

## 11. Generation order

1. **Camera calibration** — match sim camera intrinsics and mounting pose to the physical OpenArm + XHand1 rig (§6).
2. **Real-sensor range check** — confirm canonicalized readings from the physical sensor fall inside the §3.2 ranges. **Gate: they do.** Cheap, and the only check that can invalidate the whole design. Also fixes the held-out format in §3.4.
3. **Pilot** — 50 trajectories, OpenArm + XHand1 only, full format sampling. Verify the output conforms to `implementation.md` §2 by running the training repo's contract validator against it.
4. **Bulk generation** — per embodiment, weighted per §4.2.
5. **Held-out generation** — §3.4 formats, `val_heldout_format` only.

Step 2 is the one that can send you back to the drawing board, and it is cheap. Run it before committing render budget.

Step 3 must pass the training repo's startup validation (`implementation.md` §2.5) before any bulk generation begins.
