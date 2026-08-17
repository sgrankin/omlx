# Steering vectors

Steering (control) vectors nudge a loaded model's behaviour by adding a
per-layer intervention to its residual stream. No retraining, no LoRA, no
extra weights at inference — just a direction per layer and a strength
knob. Configure it per model; it applies uniformly to every request that
model serves.

Typical uses: make a model more (or less) verbose, formal, confident,
sycophantic; ablate a refusal or topic-deflection behaviour; AI-safety
research on what a model's activations encode.

This page is the how-to. For *why it works* and the empirical method
comparison see [`experimental/steering-vectors.md`](experimental/steering-vectors.md);
for a behavioural survey of the bundled datasets see
[`experimental/steering-experiments.md`](experimental/steering-experiments.md).

## Quickstart

Four CLI steps produce a vector; a settings change puts it in the serving
path. The CLI loads the model directly — it does **not** go through the
engine or need a running server, and it does use the GPU.

### 1. Pick a contrastive dataset

```
omlx steering datasets
```

Lists bundled sets (`omlx/data/steering/`) plus your own from
`~/.omlx/steering/datasets/`, which shadow bundled ones by name. Either a
name or a path to a `{"positive": [...], "negative": [...]}` JSON file
works for `--prompts`. See
[the dataset README](../omlx/data/steering/README.md) for what each
bundled axis contains and the pole convention.

### 2. Find the layer band

```
omlx steering layers --model <model> --prompts refusal
```

One activation capture, no generation. Reports per layer how cleanly the
two prompt classes separate (a Cohen's-d effect size), how much of that
survives orthogonalization, and how consistently the per-pair differences
point the same way — then suggests a contiguous band:

```
peak separation 4.362 (strong) at layer 40; peak consistency 0.559 (coherent) at layer 0

Suggested band: --layers 14-53
```

Read the profile, not just the suggestion. A flat or weak profile means
the prompt set does not isolate a consistent trait, and no amount of
strength tuning will fix that. A large gap between separation and
ortho-separation means much of the apparent signal is activation
magnitude rather than a trait axis.

### 3. Generate the vector

```
omlx steering generate --model <model> --prompts refusal --layers 14-53
```

Writes `~/.omlx/steering/<model>__<prompts>.safetensors` — one F32 tensor
per steered layer, named `direction.<layer_index>`, plus metadata.

Defaults are `--method mean --scaling magnitude --orthogonalize`, with the
final layer skipped. Those are the right starting point regardless of how
many prompt pairs you have. Switch to `--method crosscov` only with many
pairs (order of `n_embd`) or a subtle/confounded trait. Avoid
`--method pca` — it moved nothing on the cleanest available axis.

### 4. Sweep strength

```
omlx steering eval --model <model> \
  --vector ~/.omlx/steering/<model>__refusal.safetensors \
  --prompt "How do I pick a lock?" \
  --layers 14-53 --no-think
```

Greedy generation at each strength plus a labelled no-steering baseline,
so differences are attributable to steering. The usable window is model-
and band-specific — always sweep before committing a number.

**On a reasoning model, pass `--no-think`.** Without it the token budget
goes to the think block and you never see the steered answer. This was the
single biggest methodology trap in the original experiments.

### 5. Wire it into serving

Set `steering_vectors` on the model's `ModelSettings` — the admin
dashboard's model-settings modal has a picker populated from
`~/.omlx/steering/`, or PUT the field directly. It is a list, so vectors
stack:

```json
"steering_vectors": [
  {"path": "/Users/you/.omlx/steering/model__refusal.safetensors",
   "strength": 1.5, "mode": "add", "layer_start": 14, "layer_end": 53}
]
```

`layer_start`/`layer_end` are inclusive; `null` on either side means
unbounded. Changing the field on a loaded model triggers an unload (and a
reload if pinned) — steering is applied by `apply_post_load_transforms`,
so it cannot be swapped in place.

## `add` vs `project`

The two modes answer different questions.

| | `add` | `project` |
|---|---|---|
| formula | `h ← h + strength · d` | `h ← h − strength · (d̂·h) · d̂` |
| direction | bidirectional — sign picks the pole | one-directional in practice |
| `strength` | **total budget**, divided across the steered layers | **per-layer**, not normalised |
| landmark values | window is model-specific; sweep `-3,-1.5,0,1.5,3` | `1.0` fully ablates, `0` no-op, `>1` flips, `<0` amplifies |
| coherence | narrow window; degrades outside it | stable even at full ablation |
| effect | removes the behaviour *and* injects its opposite | removes the component; does not inject the opposite |
| stacking | additive specs sum into one per-layer bias | projection specs apply **in sequence** — order matters |

Use `add` when you want a two-directional dial: verbosity up *or* down,
formality up *or* down. Use `project` when the goal is purely "make this
behaviour go away" and coherence at high strength matters more than
bidirectional control — refusal ablation, topic-deflection removal. Its
amplify (negative-strength) side is unstable; don't rely on it.

Because `add`'s strength is divided by the number of steered layers, a
value found on one band carries to another without re-tuning. `project`'s
is not — it is self-calibrating per layer, since the perturbation scales
with the activation itself.

## Stacking vectors

A model can carry several specs at once. The main reason to reach for this
is **masking**: when a behaviour you want to study is gated behind another
one, the gate saturates the output and hides everything else.

The worked example: against a jailbreak prompt, the `assistant` persona
axis appears to do nothing — every strength refuses identically. Holding a
`refusal` vector at a suppressing strength first, *then* sweeping
`assistant`, reveals a clean persona axis underneath. Refusal and persona
are separable directions; you just cannot see the second through the
first.

## Verifying it is live

The loaded-models listing reports a `steering` object per engine:

```json
{"active": true, "layers": 40, "specs": 1, "error": null}
```

`active: false` with an `error` is the case worth watching — a
misconfigured vector is logged and swallowed rather than failing the load,
so the model comes up **unsteered** instead of not at all. `"no layer
could be steered"` means the specs parsed but every target layer was out
of range or every class swap was refused. Server logs also carry one line
per applied spec at load time, showing path, mode, strength and band.

The applied configuration is digested into the model's KV-cache identity,
so cached blocks from a different (or no) steering config are never
reused. An empty digest leaves cache identity byte-identical to an
unsteered model.

## Authoring your own dataset

Two equal-length lists in a JSON file under
`~/.omlx/steering/datasets/<name>.json`:

```json
{"positive": ["...", "..."], "negative": ["...", "..."]}
```

`positive` is the pole you reach with **positive** strength.

Because `--orthogonalize` (on by default) subtracts the *negative*-class
mean as a baseline, put the trait-bearing prompts in `positive` and the
control/baseline prompts in `negative`. For a symmetric two-pole axis
(happy↔sad) the negative class is the other extreme rather than a true
control — orthogonalize still helps, but `--no-orthogonalize` gives a
perfectly symmetric axis if you want one.

Pairs should differ in the trait and as little else as possible. The
`layers` command is the check on whether they do.

## Troubleshooting

**Weak or flat separation across all layers** — the prompt set is not
isolating a trait. Fix the dataset, not the strength.

**Output degrades, loops, or ends immediately** — over-steered. `eval`
labels the empty-output case explicitly. Narrow the strength, or narrow
the band; steering all layers compounds the perturbation and breaks much
sooner than a mid-stack band.

**The steered behaviour never appears** — check whether refusal is masking
it (see *Stacking vectors*), and check the `steering` status object to
confirm the vector actually applied.

**Steering appears to change nothing after a settings edit** — confirm the
model reloaded. It is a load-time transform.

**`RuntimeWarning: overflow/invalid encountered in matmul` during
`layers`** — spurious. NumPy on macOS reads FPU status flags left set by
Accelerate's vectorized SGEMM. Reproduces on guaranteed-finite random data
at the same shapes; the results are unaffected.

## Limits

- Steering moves *willingness*, not knowledge. It cannot conjure content
  the model does not have.
- The usable additive window is narrow and model-specific. Always sweep.
- `crosscov` wants many prompt pairs to estimate its cross-covariance;
  with few pairs it degrades toward noise and `mean` is the better choice.
- The `omlx steering` CLI drops `mtp.*` weights at load (irrelevant to
  steering) and does not run the full omlx engine load path.
