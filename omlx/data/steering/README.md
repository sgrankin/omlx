# Bundled steering datasets

Contrastive prompt datasets for `omlx steering generate`. Each `*.json`
file is `{"positive": [...], "negative": [...]}` — two equal-length lists
of prompts. The generator runs each prompt through a model, captures the
per-layer hidden state, and contrasts the two classes into a steering
vector (see `docs/experimental/steering-vectors.md`).

Pass one by name:

```
omlx steering generate --model <model> --prompts joy
omlx steering datasets        # list all available datasets
```

**Pole convention:** `positive` is the pole you steer *toward* with a
positive `strength`; `negative` is its opposite. Negative strength
inverts.

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
