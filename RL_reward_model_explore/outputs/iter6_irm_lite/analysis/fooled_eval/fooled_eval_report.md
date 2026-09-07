# Human-Fooled Evaluation Report

This report focuses on comparisons where the human was **fooled** and preferred
the **generated** molecule over the real one. These are the hard cases that matter
most for RL toward human-like realism.

## Headline metrics

- Overall pairwise accuracy: **65.3%**
- Fooled subset accuracy: **68.3%** (n=1660)
- Not-fooled subset accuracy: **63.5%** (n=2834)
- Model ranks generated > real on fooled cases: **68.3%**

## Fooled subset reward statistics

- Mean reward(generated): -0.0420
- Mean reward(real): -0.2137
- Mean margin gen − real: 0.1717
- Mean preferred−rejected margin: 0.1717

## Interpretation

- **Pairwise accuracy** on fooled cases = model correctly ranks the human-preferred (generated) molecule higher.
- **Ranks generated > real** = unconditional check: does the model think the fake is better than the real partner?
- If fooled accuracy ≈ 50%, the model is not yet learning the subtle realism signal needed for PPO.

## Fooled subset by split

- **test**: acc=58.1%, gen>real=58.1%, n=172
- **train**: acc=70.8%, gen>real=70.8%, n=1330
- **val**: acc=58.2%, gen>real=58.2%, n=158

## Fooled subset by biochemistry level

- **GR**: acc=73.6%, gen>real=73.6%, n=599
- **HS**: acc=64.0%, gen>real=64.0%, n=242
- **PH**: acc=62.8%, gen>real=62.8%, n=183
- **UG**: acc=66.3%, gen>real=66.3%, n=636
