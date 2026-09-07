# Human-Fooled Evaluation Report

This report focuses on comparisons where the human was **fooled** and preferred
the **generated** molecule over the real one. These are the hard cases that matter
most for RL toward human-like realism.

## Headline metrics

- Overall pairwise accuracy: **66.9%**
- Fooled subset accuracy: **81.6%** (n=1660)
- Not-fooled subset accuracy: **58.3%** (n=2834)
- Model ranks generated > real on fooled cases: **81.6%**

## Fooled subset reward statistics

- Mean reward(generated): 0.0187
- Mean reward(real): -0.1214
- Mean margin gen − real: 0.1401
- Mean preferred−rejected margin: 0.1401

## Interpretation

- **Pairwise accuracy** on fooled cases = model correctly ranks the human-preferred (generated) molecule higher.
- **Ranks generated > real** = unconditional check: does the model think the fake is better than the real partner?
- If fooled accuracy ≈ 50%, the model is not yet learning the subtle realism signal needed for PPO.

## Fooled subset by split

- **test**: acc=65.7%, gen>real=65.7%, n=172
- **train**: acc=85.9%, gen>real=85.9%, n=1330
- **val**: acc=62.7%, gen>real=62.7%, n=158

## Fooled subset by biochemistry level

- **GR**: acc=88.1%, gen>real=88.1%, n=599
- **HS**: acc=78.5%, gen>real=78.5%, n=242
- **PH**: acc=75.4%, gen>real=75.4%, n=183
- **UG**: acc=78.5%, gen>real=78.5%, n=636
