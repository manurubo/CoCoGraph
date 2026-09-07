# Human-Fooled Evaluation Report

This report focuses on comparisons where the human was **fooled** and preferred
the **generated** molecule over the real one. These are the hard cases that matter
most for RL toward human-like realism.

## Headline metrics

- Overall pairwise accuracy: **67.6%**
- Fooled subset accuracy: **74.2%** (n=1660)
- Not-fooled subset accuracy: **63.8%** (n=2834)
- Model ranks generated > real on fooled cases: **74.2%**

## Fooled subset reward statistics

- Mean reward(generated): -0.4047
- Mean reward(real): -0.7345
- Mean margin gen − real: 0.3297
- Mean preferred−rejected margin: 0.3297

## Interpretation

- **Pairwise accuracy** on fooled cases = model correctly ranks the human-preferred (generated) molecule higher.
- **Ranks generated > real** = unconditional check: does the model think the fake is better than the real partner?
- If fooled accuracy ≈ 50%, the model is not yet learning the subtle realism signal needed for PPO.

## Fooled subset by split

- **test**: acc=61.6%, gen>real=61.6%, n=172
- **train**: acc=77.2%, gen>real=77.2%, n=1330
- **val**: acc=62.0%, gen>real=62.0%, n=158

## Fooled subset by biochemistry level

- **GR**: acc=80.0%, gen>real=80.0%, n=599
- **HS**: acc=68.6%, gen>real=68.6%, n=242
- **PH**: acc=65.6%, gen>real=65.6%, n=183
- **UG**: acc=73.3%, gen>real=73.3%, n=636
