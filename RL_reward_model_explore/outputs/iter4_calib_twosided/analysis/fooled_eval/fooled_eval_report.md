# Human-Fooled Evaluation Report

This report focuses on comparisons where the human was **fooled** and preferred
the **generated** molecule over the real one. These are the hard cases that matter
most for RL toward human-like realism.

## Headline metrics

- Overall pairwise accuracy: **68.3%**
- Fooled subset accuracy: **58.6%** (n=1660)
- Not-fooled subset accuracy: **74.0%** (n=2834)
- Model ranks generated > real on fooled cases: **58.6%**

## Fooled subset reward statistics

- Mean reward(generated): -0.1682
- Mean reward(real): -0.2355
- Mean margin gen − real: 0.0673
- Mean preferred−rejected margin: 0.0673

## Interpretation

- **Pairwise accuracy** on fooled cases = model correctly ranks the human-preferred (generated) molecule higher.
- **Ranks generated > real** = unconditional check: does the model think the fake is better than the real partner?
- If fooled accuracy ≈ 50%, the model is not yet learning the subtle realism signal needed for PPO.

## Fooled subset by split

- **test**: acc=45.4%, gen>real=45.4%, n=172
- **train**: acc=61.6%, gen>real=61.6%, n=1330
- **val**: acc=47.5%, gen>real=47.5%, n=158

## Fooled subset by biochemistry level

- **GR**: acc=62.4%, gen>real=62.4%, n=599
- **HS**: acc=51.2%, gen>real=51.2%, n=242
- **PH**: acc=54.6%, gen>real=54.6%, n=183
- **UG**: acc=58.8%, gen>real=58.8%, n=636
