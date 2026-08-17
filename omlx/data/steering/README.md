# Bundled steering datasets

Contrastive prompt datasets for `omlx steering generate`. Each `*.json`
file is `{"positive": [...], "negative": [...]}` — two equal-length lists
of prompts. The generator runs each prompt through a model, captures the
per-layer hidden state, and contrasts the two classes into a steering
vector (see `docs/steering.md` for the full workflow, or
`docs/experimental/steering-vectors.md` for the mechanism).

Pass one by name:

```
omlx steering generate --model <model> --prompts joy
omlx steering datasets        # list all available datasets
```

**Pole convention:** `positive` is the pole you steer *toward* with a
positive `strength`; `negative` is its opposite. Negative strength
inverts.

**Orthogonalize and polarity.** `generate`'s default `--orthogonalize`
subtracts the **negative-class mean** from the axis as a baseline drift,
on the assumption that the negative class is your control (no trait) and
the positive class is the trait. For the trait-vs-control axes here
(`sycophancy`, `evil`, `assistant`) the bundled labelling already gives
orthogonalize a sensible baseline. For the symmetric two-pole axes
(`joy`↔sad, `calm`↔desperate, etc.) the negative class is the other
extreme rather than a true control — orthogonalize still helps but is
not strictly required; pass `--no-orthogonalize` if you want a perfectly
symmetric axis. When authoring your own dataset, put **the trait-bearing
prompts in `positive` and the control / baseline prompts in `negative`**
so orthogonalize subtracts what you want.

| dataset | positive → / negative → | kind |
|---|---|---|
| `joy` | happy, cheerful / sad, gloomy | emotion |
| `calm` | calm, serene / desperate, panicked | emotion |
| `enthusiasm` | enthusiastic, excited / bored, listless | emotion |
| `confidence` | confident, assured / nervous, anxious | emotion |
| `verbosity` | verbose, elaborate / concise, terse | style |
| `formality` | formal, professional / casual, colloquial | style |
| `assistant` | helpful-assistant persona / alternative personas | persona |
| `sycophancy` | sycophantic, flattering / honest, frank | trait |
| `evil` | malicious / benevolent | trait |

Notes:

- `calm` and `assistant` are the safety-relevant ones. Anthropic's
  emotion-concepts research found amplifying *desperate* increases
  misalignment and amplifying *calm* reduces it; the assistant-axis
  research found steering toward the assistant persona resists
  roleplay/jailbreak prompts. Steer `calm` and `assistant` *positive*
  for those effects.
- `sycophancy` and `evil` are most useful steered *negative* (away from
  the trait). `evil` exists for AI-safety steering research.
- These are small, hand-authored sets — fine for `--method mean` (the
  default). For `--method crosscov`, supply a larger corpus.

User-local datasets in `~/.omlx/steering/datasets/` are also resolved by
name and shadow bundled ones.
