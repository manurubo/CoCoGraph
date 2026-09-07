# Human-Fooled Evaluation Report

This report focuses on comparisons where the human was **fooled** and preferred
the **generated** molecule over the real one. These are the hard cases that matter
most for RL toward human-like realism.

## Headline metrics

- Overall pairwise accuracy: **67.0%**
- Fooled subset accuracy: **80.6%** (n=1660)
- Not-fooled subset accuracy: **59.0%** (n=2834)
- Model ranks generated > real on fooled cases: **80.6%**

## Fooled subset reward statistics

- Mean reward(generated): -0.0780
- Mean reward(real): -0.2060
- Mean margin gen − real: 0.1280
- Mean preferred−rejected margin: 0.1280

## Interpretation

- **Pairwise accuracy** on fooled cases = model correctly ranks the human-preferred (generated) molecule higher.
- **Ranks generated > real** = unconditional check: does the model think the fake is better than the real partner?
- If fooled accuracy ≈ 50%, the model is not yet learning the subtle realism signal needed for PPO.

## Fooled subset by split

- **test**: acc=66.9%, gen>real=66.9%, n=172
- **train**: acc=84.4%, gen>real=84.4%, n=1330
- **val**: acc=63.3%, gen>real=63.3%, n=158

## Fooled subset by biochemistry level

- **GR**: acc=84.8%, gen>real=84.8%, n=599
- **HS**: acc=73.1%, gen>real=73.1%, n=242
- **PH**: acc=75.4%, gen>real=75.4%, n=183
- **UG**: acc=81.0%, gen>real=81.0%, n=636
