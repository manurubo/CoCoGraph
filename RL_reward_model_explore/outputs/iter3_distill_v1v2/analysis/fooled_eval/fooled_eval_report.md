# Human-Fooled Evaluation Report

This report focuses on comparisons where the human was **fooled** and preferred
the **generated** molecule over the real one. These are the hard cases that matter
most for RL toward human-like realism.

## Headline metrics

- Overall pairwise accuracy: **53.8%**
- Fooled subset accuracy: **71.6%** (n=1660)
- Not-fooled subset accuracy: **43.4%** (n=2834)
- Model ranks generated > real on fooled cases: **71.6%**

## Fooled subset reward statistics

- Mean reward(generated): 0.1847
- Mean reward(real): -0.0150
- Mean margin gen − real: 0.1997
- Mean preferred−rejected margin: 0.1997

## Interpretation

- **Pairwise accuracy** on fooled cases = model correctly ranks the human-preferred (generated) molecule higher.
- **Ranks generated > real** = unconditional check: does the model think the fake is better than the real partner?
- If fooled accuracy ≈ 50%, the model is not yet learning the subtle realism signal needed for PPO.

## Fooled subset by split

- **test**: acc=63.4%, gen>real=63.4%, n=172
- **train**: acc=72.6%, gen>real=72.6%, n=1330
- **val**: acc=72.2%, gen>real=72.2%, n=158

## Fooled subset by biochemistry level

- **GR**: acc=75.8%, gen>real=75.8%, n=599
- **HS**: acc=71.1%, gen>real=71.1%, n=242
- **PH**: acc=64.5%, gen>real=64.5%, n=183
- **UG**: acc=69.8%, gen>real=69.8%, n=636
