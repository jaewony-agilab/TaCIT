# tactile-train

Format-invariant tactile policy training. Trains a policy that conditions on a
per-format sensor descriptor (`h_T`) so the same model works across tactile
sensors with different taxel counts, noise, gain, etc. Full spec in `ai_docs/`
(`architecture.md`, `data_spec.md`, `implementation.md`) — this README covers
running it and how `h`/`h_T` flow through train vs. eval.

This repo only trains. It does not generate data (`tactile-datagen`) or run
rollouts (`tactile-eval`) — see `ai_docs/implementation.md` §1 for the repo
split.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"       # add [log] too for wandb
```

Requires a `data_root` that already satisfies the data contract
(`ai_docs/implementation.md` §2) — produced by `tactile-datagen`, not this
repo.

## Running tests

```bash
pytest tests/
```

`tests/conftest.py` builds a small synthetic contract-compliant dataset, so
the suite runs without any real data. Includes `test_resume.py`, which checks
that interrupting and resuming a run reproduces bit-identical parameters.

## Validating a dataset

Before pointing a real run at a `data_root`, check it satisfies the contract
standalone (this also runs automatically on `train()` startup, but failing
fast here is cheaper):

```bash
python -c "from data.contract import validate_contract; validate_contract('/path/to/data_root')"
```

Raises `ContractError` naming the offending episode/format if anything's
wrong (missing format dir, shape mismatch, an episode in two splits, an
incomplete `h_T.json`, etc.)

## Running training

```bash
python scripts/train.py configs/stage1_baseline.yaml
python scripts/train.py configs/stage2_adversary.yaml --resume checkpoints/stage1_baseline/latest.pt
python scripts/train.py configs/stage3_full.yaml --resume checkpoints/stage2_adversary/latest.pt
```

Stages must be run in order — each stage's config assumes the previous
stage's representation already fits (`ai_docs/implementation.md` §6.4):

| Stage | Config | `use_h_encoder` | `use_adversary` | `h` source |
|---|---|---|---|---|
| 1 — baseline | `stage1_baseline.yaml` | `false` | `false` | `h_T_norm` straight into FiLM |
| 2 — adversary | `stage2_adversary.yaml` | `false` | `true` | `h_T_norm` straight into FiLM |
| 3 — full | `stage3_full.yaml` | `true` | `true` | learned latent from `h_encoder` |

Before a real run, edit each config's `data_root`, `cache_dir`, `ckpt_dir`,
and the `embodiments` DOF map (currently placeholders — see the
`# ponytail:` comment in the yaml). `batch_size: 96` is sized for a 4090;
bump to `256` on an A6000 node (`ai_docs/implementation.md` §6.1).

Checkpoints are self-contained for deployment: they assert `action_norm`,
`h_T_norm_constants`, and `format_registry` are present on save
(`src/train/checkpoint.py`), and roll over keeping the last 3 plus the
best-by-heldout-L1.

## How `h` is used — training vs. test

`h_T` is a fixed, 12-field physical descriptor of a tactile sensor *format*
(density, spacing, gain, noise, etc. — `ai_docs/implementation.md` §2.3),
stored once per format, not per episode. It's not measured from the tactile
signal on the fly; it's the sensor's own known configuration, normalized to
`h_T_norm` (16-dim) with fixed constants so it's identical across runs.

**What it's for.** Two sensors can report very different raw numbers for the
same physical contact — different gain, noise floor, spatial resolution.
Without knowing which sensor produced a reading, that ambiguity can't be
resolved. `h` tells the state encoder (via FiLM conditioning in every
transformer layer) which sensor it's looking at, so it can compensate for
that sensor's known quirks and map different sensors' raw signals to the same
internal representation for the same physical event. This is what makes the
policy work across formats, including ones never seen in training — it's not
that the model is blind to hardware, it's that it's told enough about the
hardware to correctly normalize for it.

**Stages 1–2**: no learned encoder — `h` *is* `h_T_norm`, fed straight into
FiLM. Simplest possible case: it's data you already have.

**Stage 3**: adds a learned 32-dim latent. During training, `h_encoder` looks
at the actual tokenized tactile signal and produces `mu, logvar`; `h = mu +
eps·exp(0.5·logvar)` (sampled). A closed-form KL pulls this toward a
*conditional* prior `prior_net(h_T_norm)` (not a fixed `N(0,I)`), and an
attraction loss regresses an auxiliary head back to `h_T_norm` — both push
the learned latent to actually encode the same information `h_T_norm` would
have given directly.

**At eval/deployment** (`ai_docs/implementation.md` line 522), sampling is
dropped — `h = mu`, deterministic, same principle as ACT's CVAE-style latent
being fixed at test time. The difference from ACT: ACT's style variable is
unconditional nuisance noise (which human demonstrator produced this
sequence — genuinely unobservable at test time), so it's zeroed. `h`'s prior
is *conditional* on the sensor descriptor, because unlike demonstrator
identity, the sensor in use is knowable — either directly (compute `h_T_norm`
from the new sensor's spec and run it through `prior_net`) or by inferring it
from a batch of that sensor's actual data via `h_encoder`. Zeroing `h` would
be equivalent to assuming an average/null sensor and would miscalibrate real
readings; the eval-time analog of ACT's trick is `h = prior_net(h_T_norm).mu`
(or `h_encoder(tokens).mu` if inferring online), not `h = 0`.

**The adversary is a different mechanism, one step downstream.** It acts on
`z_t` (the pooled state *after* the encoder), not on `h`. Its job is to
scrub any *leftover* format signature from `z_t` beyond what `h` already
explicitly carries — so the action decoder can't learn shortcuts like
"format X correlates with task Y in this dataset" instead of reading actual
contact state. `h` is supposed to carry format info (that's how the
compensation works); `z_t` is supposed to be clean of it. The contamination
probe (`src/train/probe.py`, Ridge regression per `h_T` dimension) is the
diagnostic that checks whether this actually held.
