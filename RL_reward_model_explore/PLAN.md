# Reward-model exploration plan

Working copy of `RL_reward_model/`. **Do not edit the original tree.**
Baseline to beat: `RL_reward_model/outputs/v2_improved_run/` (v2).

## v2 failure modes (what we are actually trying to fix)

| Metric | v2 | v1 (`full_run_weighted`) | What “better” means |
|---|---|---|---|
| Overall pairwise | 69.7% | 69.8% | not the goal by itself |
| **Test accuracy** | **56.5%** | 67.5% | **raise held-out accuracy** |
| Fooled (all) | 81.8% | 49.0% | keep strong (do not collapse to ~50%) |
| **Fooled test** | **65.7%** | 43.0% | **raise, or at least hold ≥60%** |
| Fooled train | ~86% | — | **not a success metric** (overfit) |
| **Not-fooled** | **62.6%** | 81.9% | **raise toward v1** |
| Prefers-generated | 53.8% | 29.5% | humans are **~37%** |
| Mean \|score diff\| | ~0.09 | larger in v1 | **wider, usable RL advantages** |

Success is **held-out** pairwise accuracy (test fooled **and** not-fooled), less gen-bias vs 37%, usable dynamic range, without destroying fooled detection.

Protocol: same CSV, `player_id` split (seed 42), same analysis scripts, encoder init from `GINETimePredictor` when compatible, `--cache-dir` points at original feature cache **read-only**. Max 80 epochs, early-stop patience 15 (v2 stopped at 37).

---

## Idea backlog

### A. Morgan fingerprint fusion (GINE ⊕ Morgan) — **iteration 1**

- **Status:** dropped as standalone (keep-as-component: fusion idea only)
- **Hypothesis:** Global-mean-pooled GINE embeddings lose local substructure identity. Humans often catch (or miss) fakes via scaffolds / functional groups that Morgan bits encode. Fusing a 2048-d radius-3 FP (same recipe as `GINETimePredictor_MorganFP`) should help both obvious fakes (not-fooled) and realistic fakes (fooled), and widen score range.
- **Why it could beat v2:** v2’s encoder overfits player-specific graph patterns; fingerprints are molecule-identity features that transfer across `player_id`.
- **Sketch:** Keep 3× Enhanced GINE encoder. MLP `2048→512→256`. Concatenate with graph embedding, fuse to hidden, then existing dropout reward head. Still a **scalar per molecule** (RL-compatible).
- **Risk:** FP may latch onto “generated vs ChEMBL” dataset artifacts; extra capacity can overfit train fooled even harder.
- **Keep if:** test acc ≥ 58% **or** (not-fooled ≥ 66% and fooled test ≥ 60%). Drop if fooled test < 55% or test acc ≤ v2 with no other gain.
- **Result:** Extra ~1.2M unfrozen FP weights memorized train by epoch 2 (train fooled 100%, val fooled 42%). Best ckpt = epoch 1 ≈ slightly worse v2. **Drop standalone. Keep fusion only with capacity control.**

### B. Conservatism calibration + two-sided ranking — **iteration 4**

- **Status:** keep-as-component (not a standalone RM)
- **Hypothesis:** v2 only applies a **fooled** hinge, so the model learns “prefer generated.” A batch prior `mean(σ(r_gen−r_real)) ≈ 0.37` plus a **not-fooled** hinge (real should beat gen by a margin) should pull gen-bias toward humans without undoing BT.
- **Why it could beat v2:** Directly attacks 53.8% vs 37% and the not-fooled collapse (62.6% vs 82%). Iters 1–3 showed encoder/FP capacity and v1+v2 distillation do not fix this; the v2 65% fooled composite is what *saves* the gen-biased snapshot.
- **Sketch:** Original `gin` (no extra capacity). `L = L_BT + λ_m L_fooled_hinge + λ_n L_notfooled_hinge + λ_cal (mean σ(r_gen−r_real) − 0.37)² + λ_s relu(0.3 − mean|r_pref−r_rej|)`. Checkpoint on balanced score (40% overall + 30% fooled + 30% not-fooled).
- **Risk:** Prior can cap fooled accuracy; too-strong not-fooled hinge recreates v1 conservatism.
- **Keep if:** prefers-gen moves toward 37±8pp **and** not-fooled ↑ without fooled test falling below 58%.
- **Result:** Best = epoch 10 / 25, 14.0 min. Test 60.3 vs 56.5; not-fooled 74.0 vs 62.6; prefers-gen **38.0%** (human 37%); mean |diff| 0.302 vs 0.094. Fooled-all 58.6 vs 81.8; fooled-test 45.4 vs 65.7 (below 58%). Calibration λ=8 overshot into v1-style conservatism. **Keep-as-component** (two-sided hinge + balanced ckpt + milder prior). Do not ship.

### C. Distill v1 (conservative) + v2 (aggressive) into one student — **iteration 3**

- **Status:** dropped
- **Hypothesis:** v1 and v2 are complementary teachers. Soft labels on `r_pref−r_rej` teach a student the union of “crush bad fakes” and “promote realistic gens,” which hard 0/1 BT cannot express.
- **Why it could beat v2:** Explicitly transfers v1’s not-fooled skill and v2’s fooled skill. After two FP capacity failures, go back to the original GIN and change the **supervision**, not the encoder.
- **Sketch:** Precompute teacher diffs from original checkpoints (read-only). `target = α·v2 + (1−α)·v1`. `L = L_BT + λ_kd BCE(student/T, σ(target/T))`. Student is original `gin`.
- **Risk:** Averaging two wrongs; student copies v2 train overfit.
- **Result:** Best ckpt epoch 3 (val composite 0.662, even “better” than v2’s 0.626) but test 50.4%, not-fooled 43.4% (below chance), prefers-gen 62.1%. Distillation + v2 composite amplified gen-bias. **Drop.**

### D. Dual-head multi-task (BT + time/quality) — **iteration 5**

- **Status:** keep-as-component (not a standalone RM)
- **Hypothesis:** The pretrained time model is a molecule-quality prior (v2 reward–time corr ≈ −0.54 on gens). An auxiliary time head on the shared encoder regularizes away from player-id memorization.
- **Sketch:** Original `gin` + extra time head (`gin_multitask`). Freeze `GINETimePredictor`, z-score precomputed targets, MSE on the extra head. Ranking head still BT. Reuse B’s two-sided hinge + balanced checkpoint; lower λ_cal from 8 → 2 so fooled detection can recover.
- **Risk:** Time scores may not align with Turing preferences (humans fooled by “pretty” but temporally-odd graphs). First launch had a `[B,1]` vs `[B]` MSE broadcast (train loss ~3); retrained after squeezing to a scalar.
- **Keep if:** val/test gap shrinks and not-fooled ↑, without losing iter 4’s gen-bias / test gains, and fooled-all closer to ≥70%.
- **Result:** Best = epoch 29 / 44, 35.4 min. Test 57.6 vs 56.5; not-fooled 63.8 vs 62.6; fooled-all 74.2 vs 81.8 (≥70%); fooled-test 61.6 vs 65.7; prefers-gen 50.2 vs 53.8; mean |diff| 0.490. Recovered fooled vs iter 4 but lost the 38% gen-bias / 74% not-fooled. **Keep-as-component** (milder prior + balanced ckpt; time head not clearly additive). Do not ship.

### E. Player-adversarial encoder / IRM-lite — **iteration 6**

- **Status:** keep-as-component (not a standalone RM)
- **Hypothesis:** Test drop (72.8% train → 56.5% test) is player leakage. Encoder should be invariant to `player_id`.
- **Sketch:** `gin_multitask` + B/D keep-as-component losses. Hash `player_id` into 16 buckets. IRM-lite penalty on per-bucket BT loss (`--irm-weight 1.0`).
- **Risk:** Removing player style may also remove legitimate preference signal (experts vs undergrads).
- **Keep if:** test acc ↑ by ≥3pp vs iter 5 (same architecture without IRM), fooled test held.
- **Result:** Best = epoch 19 / 34, 30.7 min. Test 60.8 vs 56.5 (+4.3) and vs iter 5 57.6 (+3.2). Train 66.3 → test 60.8 (gap 5.5pp vs v2 ~16pp). Fooled-all 68.3 (<70%); fooled-test 58.1 vs iter 5 61.6 (not held); not-fooled 63.5; prefers-gen 48.2; mean |diff| 0.301. **Keep-as-component** (IRM shrinks player leakage). Do not ship.

### F. Two-stage: frozen pretrained encoder + small fused head — **iteration 2**

- **Status:** dropped as standalone (keep-as-component: “don’t give unique embeddings a huge trainable map”)
- **Hypothesis:** Updating all GINE weights on ~3.6k pairs is how v2 overfits fooled train (~86%). Freeze encoder (time-model init), train only a **small** FP bottleneck + reward head.
- **Risk:** Underfitting realism cues that are not in the time model. Unique frozen embeddings still let a head memorize pairs.
- **Result:** Best v2-composite ckpt = epoch 3. Test 54.8 vs 56.5; not-fooled 59.0 vs 62.6; fooled test 66.9 vs 65.7 (tiny bump); prefers-gen 55.6 vs 53.8. Epoch 12 had test 60.3% but was not selected. Freeze does not stop lookup-table overfitting. **Drop.**

### G. Graph-pair mixup / embedding mixup on pairs

- **Status:** pending (likely skip unless 1–3 fail)
- **Hypothesis:** Mixup on fused embeddings with mixed labels reduces overconfident fooled memorization.
- **Risk:** Mixing molecules is chemically meaningless; may blur BT.

---

## Iteration log

### Iteration 1

- **Chosen:** A — Morgan fingerprint fusion
- **Why first:** Highest-leverage **architecture** change already hinted in the original README; keeps RL-compatible scalar scores; complementary features without touching fooled loss weights.
- **Train:** `outputs/iter1_gin_fp/`, `--model-type gin_fp`, 16 epochs / 15.1 min, RTX 4090, early-stop on epoch-1 composite (later epochs memorized train).
- **Metrics vs v2:** overall 66.9 vs 69.7; test 55.1 vs 56.5; fooled 81.6 vs 81.8; not-fooled 58.3 vs 62.6; prefers-gen 56.4 vs 53.8; fooled test 65.7 vs 65.7; mean |diff| 0.142 vs 0.094.
- **Decision:** **Drop** as a standalone reward model. **Keep-as-component:** fingerprint fusion, but only with a small frozen-encoder head (iteration 2).

### Iteration 2

- **Chosen:** F — frozen GINE encoder + small Morgan bottleneck head (`gin_fp_small`)
- **Why:** Iter 1 failed by capacity, not by the fusion hypothesis. The 1.2M FP MLP + unfrozen encoder reached 100% train fooled by epoch 12. Two-stage training is the direct fix.
- **Train:** `outputs/iter2_frozen_fp_small/`, 18 epochs / 9.4 min, best = epoch 3 (v2 composite). Trainable 141k/204k.
- **Metrics vs v2:** overall 67.0 vs 69.7; test 54.8 vs 56.5; fooled 80.6 vs 81.8; not-fooled 59.0 vs 62.6; prefers-gen 55.6 vs 53.8; fooled test 66.9 vs 65.7; mean |diff| 0.145 vs 0.094.
- **Decision:** **Drop** standalone. Unique frozen embeddings still allow the head to memorize pairs; v2 composite still selects the fooled-max epoch. Note: epoch 12 had test 60.3% (not selected).

### Iteration 3

- **Chosen:** C — distill v1 + v2 into original `gin` student
- **Why:** FP capacity tricks did not fix the objective. v1 and v2 already encode the two skills we want; soft-label distillation changes supervision without adding parameters that memorize.
- **Train:** `outputs/iter3_distill_v1v2/`, α=0.5, T=2.0, λ_kd=1.0. First launch failed (v1 head index remap + logit shape); retry trained 18 epochs / 11.1 min. Best = epoch 3 (val composite 0.662).
- **Metrics vs v2:** overall 53.8 vs 69.7; test 50.4 vs 56.5; fooled 71.6 vs 81.8; not-fooled 43.4 vs 62.6; prefers-gen 62.1 vs 53.8; fooled test 63.4 vs 65.7; mean |diff| 0.272 vs 0.094.
- **Decision:** **Drop.** Soft mix + v2 composite pushed gen-bias further. Not-fooled below chance.

### Iteration 4

- **Chosen:** B — conservatism calibration + two-sided ranking + balanced checkpoint
- **Why:** Iters 1–3 failed by capacity/supervision mix; remaining failure modes are gen-bias (53.8% vs 37%) and not-fooled (62.6% vs 82%).
- **Train:** `outputs/iter4_calib_twosided/`, original `gin`, λ_n=0.5, λ_cal=8.0, λ_s=0.5, `--checkpoint-metric balanced`. 25 epochs / 14.0 min, RTX 4090, best = epoch 10 (val balanced 0.597).
- **Metrics vs v2:** overall 68.3 vs 69.7; test 60.3 vs 56.5; fooled 58.6 vs 81.8; not-fooled 74.0 vs 62.6; prefers-gen 38.0 vs 53.8; fooled test 45.4 vs 65.7; mean |diff| 0.302 vs 0.094.
- **Decision:** **Keep-as-component.** Hits the 37% prior and lifts test + not-fooled, but fooled detection collapsed (cannot ship: fooled-all < 70%). Reuse two-sided hinge + balanced ckpt; lower λ_cal on later iters.

### Iteration 5

- **Chosen:** D — aux time/quality head on original GIN, stacked on B keep-as-component (λ_cal 8→2)
- **Why:** Iter 4 showed the objective works for gen-bias/not-fooled/test but overshot fooled. A time head should regularize player-id memorization; milder calibration should let fooled recover.
- **Train:** `outputs/iter5_time_head/`, `--model-type gin_multitask --aux-time-weight 1.0`, λ_n=0.5, λ_cal=2.0, λ_s=0.5, balanced ckpt. Retrain after scalar time-MSE fix. 44 epochs / 35.4 min, best = epoch 29.
- **Metrics vs v2:** overall 67.6 vs 69.7; test 57.6 vs 56.5; fooled 74.2 vs 81.8; not-fooled 63.8 vs 62.6; prefers-gen 50.2 vs 53.8; fooled test 61.6 vs 65.7; mean |diff| 0.490 vs 0.094.
- **Decision:** **Keep-as-component.** Tiny test/not-fooled bumps and fooled ≥70%, but fooled-test still below v2 and gen-bias far from 37%. Time head not clearly better than just lowering λ_cal.

### Iteration 6

- **Chosen:** E — IRM-lite on `gin_multitask`, stacked on B/D keep-as-component
- **Why:** Iter 5 still had a large train→test drop (69.8→57.6). IRM-lite on player-id buckets is the leftover leakage hypothesis.
- **Train:** `outputs/iter6_irm_lite/`, same losses as iter 5 plus `--irm-weight 1.0`. 34 epochs / 30.7 min, best = epoch 19 (val balanced 0.612).
- **Metrics vs v2:** overall 65.3 vs 69.7; test 60.8 vs 56.5; fooled 68.3 vs 81.8; not-fooled 63.5 vs 62.6; prefers-gen 48.2 vs 53.8; fooled test 58.1 vs 65.7; mean |diff| 0.301 vs 0.094. Train 66.3 vs test 60.8 (gap 5.5pp).
- **Decision:** **Keep-as-component.** Hits E’s +3pp test bar vs iter 5, and the train–test gap collapsed, but fooled-all < 70% and fooled-test was not held. Do not ship.

---

## After 6 iterations

**Ship nothing from this copy.** Keep using `RL_reward_model/outputs/v2_improved_run/`.

Among 4–6, no official checkpoint beats v2 on the joint goal (held-out test **and** fooled-test **and** not-fooled, without fooled-all < 70%).

| Run | What it actually fixed | What it broke |
|---|---|---|
| Iter 4 (λ_cal=8) | prefers-gen 38%, not-fooled 74%, test 60.3% | fooled-all 58.6%, fooled-test 45.4% |
| Iter 5 (λ_cal=2 + time) | fooled-all back to 74.2% | only +1pp test; gen-bias 50% |
| Iter 6 (+ IRM) | test 60.8%, train–test gap 5.5pp | fooled-all 68.3%, fooled-test 58.1% |

**If there is an iteration 7:** keep two-sided hinge + balanced checkpoint + IRM, but add a **hard fooled floor** on checkpoint selection (reject snapshots with fooled-all < 70% or fooled-test < 60%) and search λ_cal in {2, 4} rather than 8. Do not add mixup (G) or more encoder/FP capacity.

---

## Keep / drop rule (applied every iteration)

Compare to v2 on the **same** `analyze_predictions.py` split:

1. **Keep** if test accuracy **and** (fooled test **or** not-fooled) improve, and fooled-all does not fall below 70%.
2. **Keep-as-component** if one failure mode clearly improves (e.g. gen-bias or not-fooled) but another regresses — reuse the mechanism, don’t ship as the standalone RM.
3. **Drop** if no held-out gain, or fooled detection collapses, or prefers-gen moves *farther* from 37% with no test gain.

Primary ranking of a run: `(test_acc, fooled_test_acc, not_fooled_acc, abs(pref_gen − 0.37), mean|score_diff|)`.
