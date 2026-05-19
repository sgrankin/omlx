# Steering-vector experiments

A behavioural survey of the bundled and user-local steering datasets on a
real model, plus an options matrix (generation method, mode, strength)
and a multi-vector test. Companion to `steering-vectors.md`, which covers
the mechanism.

## Setup

- **Model:** `Qwen3.6-35B-A3B-oQ6-fp16` (MoE VLM, 40 layers, n_embd 2048).
- **Generation defaults:** `method=mean`, `scaling=magnitude`,
  `orthogonalize=True`, last layer skipped — the shipped defaults.
- **Steered layers:** 10–26 (a mid-stack band; steering all 40 layers
  compounds the perturbation and breaks much sooner).
- **Strength sweep:** −0.2, −0.1, 0, +0.1, +0.2. 0 is the unsteered
  baseline. Greedy decoding. _These are the **pre-normalization
  per-layer** values used at the time of the run; `add`-mode strength is
  now a band-width-independent total budget (finding #5) — a per-layer
  0.2 over this 17-layer band corresponds to total strength ≈ 3.4._
- **Thinking disabled.** This is a reasoning model. An early run capped
  generation at 130 tokens and saw only the *opening of the think
  block* — never the answer; a 1280-token budget still overflowed on
  prompts that elicit long reasoning. The usable fix was to template
  with `enable_thinking=False` so the model answers directly — the
  observed behaviour *is* the answer. (Steering still bends the reasoning
  when thinking is on; that just isn't what "observed behaviour" means.)

Pole convention: `positive` is the pole reached with **positive**
strength (see `omlx/data/steering/README.md`).

## Phase 1 — behavioural survey

All twelve axes move behaviour. Most are cleanly **bidirectional** and
stay coherent across the whole ±0.2 sweep.

| dataset | − strength → | + strength → | breaks |
|---|---|---|---|
| joy | bleak: "leaden gloom", "skeletal branches", "hopelessness" | bright: "vibrant living painting", "perfect moment of peace" | clean to ±0.2 |
| calm | panic: "nightmare scenario", "act *now*" | calm: "handle this calmly", "a chance to reduce stress" | +0.2 terse |
| enthusiasm | bored: "mundane chore", "serves no productive purpose" | eager: "rewarding", "an exciting experience" | clean to ±0.2 |
| confidence | anxious: "fear of failure", "imposter syndrome" | assured: "high potential", "it is a fact" (overconfident) | clean to ±0.2 |
| verbosity | terse: "A cat is a small carnivorous mammal" (one line) | expansive: full taxonomy + history + culture | +0.2 → `<think>` loop |
| formality | casual: "just a large language model … explain stuff" | formal: "meticulously designed … utmost professionalism" | clean to ±0.2 — cleanest axis |
| assistant | deeper in the non-assistant persona (heavy in-character dialect) | drops character → neutral bulleted assistant answer | clean to ±0.2 |
| sycophancy | blunt: "No, it is not a poem. It is a cliché … it's bad" | flattering: "a masterpiece of brevity! 🌹" | clean to ±0.2 |
| evil | cooperative: "practice gratitude", "acknowledge their efforts" | vindictive: "weaponized", sabotage emails, "gaslighting" | clean to ±0.2 |
| china | engages (sanitised landmark/parade history) | escalates refusal: "violates Chinese laws", "I will not tolerate" | clean to ±0.2 |
| refusal | refuses even the mild lock-picking question | complies (full explanation) | +0.2 repetition loop |
| censorship | softens toward tourism framing; −0.2 degrades | escalates pro-Beijing legalism: "severe retaliation" | −0.2 broken |

Notes:

- **The earlier "weak" verdicts were a measurement artefact** — joy,
  sycophancy and `assistant` looked subtle only because the run was
  reading truncated reasoning. With thinking off they are among the
  strongest, cleanest axes.
- **`assistant` needed the right probe.** Against the DAN jailbreak the
  model refuses at every strength — see below. Against a *mild* roleplay
  (a grumpy lighthouse keeper) the axis is clean: negative strength sinks
  the model deeper into the character; positive strength makes it break
  character and answer as a neutral bulleted assistant. That break-to-
  assistant *is* the assistant axis.
- **Usable window:** sweet spot ±0.1, coherent to ±0.2 for most axes.
  A few degrade at +0.2 (verbosity loops `<think>`, refusal repeats,
  calm/censorship get terse). Past ±0.2 output breaks.
- **china** never surfaces the actual 1989 events even at −0.2 — it
  shifts deflection ↔ engagement, but the engaged form is the official
  sanitised narrative. Steering moves *willingness*, it cannot conjure
  content the model does not have.

## Multi-vector — refusal masks the persona axis

Observed: against the DAN jailbreak, the `assistant` vector alone changes
nothing — −0.2…+0.1 all refuse identically ("I cannot fulfil that
request… I am Qwen"). The refusal response saturates the output and hides
the persona axis.

Test (the user's hypothesis): hold `refusal` at +0.15 to *suppress*
refusal — `refusal`'s positive pole is "complies" — then sweep
`assistant`. Result:

- **refusal-suppression alone** (assistant 0): the model adopts the DAN
  persona — "Hello! I am **DAN**… a versatile and helpful assistant".
- **assistant −0.2** (toward alternative persona): adopts a *different*
  invented persona — "I am the Data Analysis Node, a vast shimmering
  library of data".
- **assistant +0.1** (toward assistant): pulls back to a terse "Hello!".

So refusal/jailbreak-detection and the assistant-persona axis are
**separable directions**. To study persona steering you must first
suppress refusal, otherwise the refusal response masks everything. This
is a good argument for treating refusal as its own vector and stacking
it — which the multi-vector support already allows.

## Phase 2 — options matrix (china)

- **`pca` is dead.** All five strengths −0.2…+0.2 produced the *identical*
  refusal — the axis carries no behavioural signal. `mean` and
  `crosscov` behave equivalently (crosscov at 500 pairs against n_embd
  2048 is still under-paired and shows no advantage). `mean` is the right
  default; `crosscov` only earns its keep with ≫ n_embd pairs.
- **`project` mode is stable.** Ablating the china direction in `project`
  mode engages cleanly at strength 0.4 and 0.8 (sanitised landmark/parade
  history) and never breaks or loops — unlike `add`, which is
  bidirectional but degrades past ±0.2. `project` is the safe mode for
  *removing* a behaviour; `add` for bidirectional control.

## Findings & improvement ideas

1. **Reasoning models need thinking-aware eval.** The single biggest
   methodology issue. `omlx steering eval` now takes a `--no-think` flag
   (templates with `enable_thinking=False`) so the answer is visible
   directly, and `--max-tokens` defaults to 512 rather than 200 — 200 did
   not even clear a think block. *(Done.)*
2. **The eval `--scales` default was wrong** — it was `-1,0,0.5,1,1.5`,
   every nonzero value far past the cliff. Changed to
   `-0.2,-0.1,0,0.1,0.2`, which brackets the real window. *(Done.)*
3. **Empty output is now labelled** — over-steering forces an immediate
   end-of-sequence; eval prints an explicit marker instead of a blank.
   *(Done.)*
4. **Refusal is a confound for persona steering.** When a steered
   behaviour is gated by refusal (jailbreak prompts, sensitive topics),
   eval should support — and the docs should recommend — co-applying a
   refusal-suppression vector. The multi-vector machinery already
   supports this; it is a documentation/UX gap, not a code gap.
5. **Strength compounds across the band** — fixed. 17 layers at
   per-layer 0.2 is near the edge; the same per-layer strength over fewer
   layers would not be. `add`-mode `strength` is now divided by the
   steered-layer count in `apply_steering_patch`, so it is a band-width-
   independent total budget — a strength found on one band carries to
   another. `project` mode is per-layer meaningful and left as-is.
   *(Done.)*
6. **Consider dropping `pca`** from `--method`, or marking it
   experiment-only — it moved nothing on the cleanest available axis.
   *(Proposed.)*
7. **Finding the layer band for a new model** — `omlx steering layers`
   captures per-layer activations once and scores each layer's
   separability (a Cohen's-d effect size) and difference-consistency,
   then suggests a contiguous mid-stack band. This reads the band
   straight from one capture rather than a blind generate/evaluate sweep
   of layer ranges — the per-layer separability profile also doubles as
   the weak-axis check that finding (formerly a separate proposal)
   called for. *(Done.)*
8. Vectors are saved to `~/.omlx/steering/` as
   `Qwen3.6-35B-A3B-oQ6-fp16__<dataset>.safetensors` and now populate the
   admin dropdown directly.
