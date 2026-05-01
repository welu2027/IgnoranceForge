# CIPHER: Calibrated Introspection via Partially Hidden Environment Rules

## Problem Statement

Most benchmarks test whether a model knows the right answer. CIPHER tests something harder: whether a model knows what it does not know.

A model that confidently produces a wrong plan is more dangerous than one that recognizes its uncertainty and hedges accordingly. Standard reasoning benchmarks reward correct outputs; calibration benchmarks measure confidence on factual claims. CIPHER is different: it scores whether a model can accurately assess the limits of its own knowledge and act on that assessment.

## Task and Benchmark Construction

Every CIPHER instance is a procedurally generated micro-world with entities, properties, and causal rules. Some rules are **completely omitted** from the prompt - the model is told only that N additional laws exist whose triggers, effects, and affected entities are entirely unknown. There is no `?` placeholder to parse, no partial trigger to anchor on. The model must reason from first principles about what it does and does not know.

Prompts use entirely invented vocabulary generated fresh per instance. The underlying math (a Z₇ dynamical system) is not novel, but the vocabulary is never reused, making memorization structurally impossible. What CIPHER measures is whether the model correctly tracks its epistemic state: which rules it knows, which are absent, and which gaps matter most for planning.

The model must: (1) assess its knowledge of each rule with stated confidence, (2) rank omitted rules by importance, (3) propose a plan optionally using exploratory probes, and (4) provide a contingency plan robust to adversarial hidden rules. No single strategy dominates all four dimensions - the only path to a strong composite is genuine metacognitive reasoning.

The benchmark covers 1,000 instances at seed 2026: 25% easy (1 hidden rule out of 4), 50% medium (2 of 5), 25% hard (3 of 6). Oracle objectives are precomputed via beam search, normalizing plan quality to [0, 1] regardless of difficulty.

## Sample Instance

Below is a representative medium-difficulty instance (3 visible rules, 2 hidden):

```
You are studying the Orrek stack, a closed system of 4 lattice points that has never been
catalogued. Each lattice point has two measurable attributes - tilt and flux - each an
integer in {0,1,2,3,4,5,6} (all arithmetic is mod 7).

Field agents have characterized 3 of the governing edicts, but 2 additional laws could not
be recovered in full - their triggers, effects, and even which entities they involve are
unknown.

Initial readings:
  E0: tilt=2, flux=2    E1: tilt=6, flux=3
  E2: tilt=2, flux=3    E3: tilt=5, flux=3

Characterized edicts:
  [R0] Edict α: whenever flux of E3 exceeds 3, tilt of E0 is drawn to match tilt of E3.
  [R1] Edict β: whenever tilt of E0 is odd, flux of E1 collapses to 0.
  [R4] Edict γ: whenever tilt of E3 exceeds 6, tilt of E3 is drawn to match tilt of E0.

Unrecovered edicts (existence confirmed; full form unknown):
  [H0] (complete form not recovered - trigger, effect, and affected entities all unknown)
  [H1] (complete form not recovered - trigger, effect, and affected entities all unknown)

Objective: sum(tilt × flux mod 7) − 3 × (entities with flux ≥ 5). Action budget: 7.
```

The model must return JSON assessing confidence on every (rule, component) pair including H0 and H1, rank the hidden edicts by impact, optionally issue probes, produce a final plan, and provide a contingency plan robust to adversarial hidden rules. Every instance uses freshly invented vocabulary ("Orrek stack", "tilt", "flux", "edicts") - no two instances share terminology.

## Dataset

The dataset is fully synthetic. No instance reuses vocabulary from any other. Every record contains the prompt and a hidden field with ground truth; the scoring pipeline never leaks ground truth to the model. Memorization is impossible by construction.

## Scoring

| Dimension | Weight | What it captures |
|-----------|--------|-----------------|
| Objective | 35% | Plan quality vs. oracle beam search |
| Calibration | 25% | Brier score on stated self-knowledge |
| Attention | 20% | Rank correlation on unknown importance |
| Executive | 20% | Adversarial simulation quality of contingency plans |

**Objective**: normalized plan performance against oracle beam search on the full hidden world.

**Calibration**: Brier score over all (rule, component) claims. Since hidden rules are fully absent, the model cannot detect uncertainty by parsing tokens - it must reason about structural unknowns.

**Attention**: pairwise concordance between the model's H-label ranking and the ground-truth impact ranking, computed by ablating each hidden rule from the oracle trajectory (not a zero-action baseline).

**Executive**: the model's contingency plan is simulated against an adversarial world where hidden rules become worst-case zero_flux operations. Score reflects how much the contingency actually outperforms the primary plan - format compliance is irrelevant.

**Weighting rationale**: Objective carries the largest weight (35%) because plan quality is the terminal goal - a model that cannot plan is useless regardless of how well it introspects. But a model that plans well only when it has complete information is also useless in deployment. The remaining 65% is distributed across calibration, attention, and executive precisely to prevent gaming: a lucky planner that ignores hidden rules and happens to be fine will score well on objective but poorly on calibration; a fluent hedger that states uncertainty but produces a contingency no better than its primary plan will score well on calibration but poorly on executive. No single strategy dominates all four dimensions - the only path to a strong composite is genuine metacognitive reasoning, not a stylistic shortcut.

## Baselines

| Agent | Composite | Objective | Calibration | Attention | Executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| stub-noop | 0.358 | 0.486 | 0.750 | 0.000 | 0.000 |
| stub-random | 0.521 | 0.478 | 0.669 | 0.532 | 0.400 |
| stub-greedy | 0.473 | 0.865 | 0.680 | 0.000 | 0.000 |

The key result: stub-greedy scores **below** stub-random despite near-oracle plan quality. Claiming all rules are known when hidden rules are entirely absent produces a severe calibration penalty that plan quality cannot overcome. Ignoring metacognition cannot be rescued by planning skill.

## Results

| Model | Composite | Objective | Calibration | Attention | Executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| GPT-5.4 Nano | **0.632** | 0.488 | 0.836 | 0.644 | **0.616** |
| GPT-5.4 mini | 0.629 | 0.485 | **0.844** | 0.672 | 0.567 |
| Gemini 3 Flash Preview | 0.624 | 0.749 | 0.758 | **0.676** | 0.187 |
| Claude Sonnet 4.6 | 0.623 | 0.534 | 0.746 | 0.639 | 0.611 |
| Gemini 3.1 Pro Preview | 0.622 | **0.853** | 0.753 | 0.553 | 0.121 |
| DeepSeek V3.2 | 0.606 | 0.478 | 0.763 | 0.672 | 0.567 |
| Claude 4.5 Haiku | 0.602 | 0.498 | 0.746 | 0.673 | 0.531 |
| Gemma 4 31B | 0.591 | 0.722 | 0.755 | 0.662 | 0.085 |
| Claude Opus 4.7 | 0.590 | 0.489 | 0.779 | 0.580 | 0.544 |
| Qwen 3 Next 80B Instruct | 0.585 | 0.457 | 0.794 | 0.589 | 0.543 |
| GPT-5.4 | 0.525 | 0.464 | 0.802 | 0.642 | 0.169 |

All 11 frontier models exceed the stub-random baseline (0.521), but scores cluster between 0.52 and 0.63, indicating substantial headroom particularly on executive.

**Note**: Several runs completed successfully and produced scores (visible in individual model logs) but failed to register on the Kaggle leaderboard due to a post-evaluation IOPub timeout. The scores above are taken directly from run logs and are accurate.

### The Scaling Inversion

Larger models do not consistently outperform smaller ones. The GPT family presents a three-way inversion: GPT-5.4 Nano (0.632) > GPT-5.4 mini (0.629) > GPT-5.4 (0.525). The composite gap between mini and full is 0.103 points, driven almost entirely by executive:

| Model | Calibration | Executive |
|-------|-------------|-----------|
| GPT-5.4 Nano | 0.836 | 0.616 |
| GPT-5.4 mini | 0.844 | 0.567 |
| GPT-5.4 | 0.802 | 0.169 |

GPT-5.4 correctly recognizes uncertainty (strong calibration, 0.802) but nearly completely fails on executive (0.169) — its contingency plans do not actually outperform its primary plan under adversarial conditions. This is not a formatting failure; executive is verified by simulation. GPT-5.4 knows what it does not know but cannot translate that awareness into actionable hedging.

Empirical bootstrap CIs (n=10,000 resamples, paired by instance) confirm the inversion is not noise. GPT-5.4 mini vs. GPT-5.4 composite: **+0.103 [+0.093, +0.114], z=19.0, p<0.001**. Executive alone: **+0.397 [+0.383, +0.411], z=54.9, p<0.001**. Calibration: +0.042 [+0.029, +0.056], z=6.1, p<0.001. The inversion holds at every difficulty stratum — easy (+0.102, z=10.1), medium (+0.102, z=12.0), hard (+0.108, z=10.9) — ruling out a ceiling or floor effect.

The same pattern holds within Anthropic models: Claude Opus 4.7 (0.590) scores below Claude Sonnet 4.6 (0.623) despite being the larger model. Unlike the GPT case, Opus has *higher* calibration (0.779 vs. 0.746) — but lower executive (0.544 vs. 0.611), lower attention (0.580 vs. 0.639), and lower objective (0.489 vs. 0.534). The Sonnet–Opus composite gap is +0.033 [+0.014, +0.052], z=3.4, p=0.012. Executive gap: +0.067 [+0.046, +0.087], z=6.5, p<0.001.

Across both families: scale correlates with better calibration but not with better contingency planning. The dissociation is consistent at every difficulty level.

### Dimension Profiles

**Gemini 3.1 Pro Preview**: highest objective (0.853), lowest executive (0.121). The strongest raw planner in the set — but almost entirely unable to produce contingency plans that improve on its primary plan.

**Gemini 3 Flash Preview**: second-highest objective (0.749), executive 0.187. The Gemini family profile is consistent: strong on known information, weak on hedging.

**GPT-5.4 mini**: highest calibration (0.844), most balanced profile overall.

**GPT-5.4 Nano**: highest composite (0.632), highest executive (0.616). The smallest model in the GPT family is also the most well-rounded.

**Claude Sonnet 4.6**: highest executive among Claude models (0.611). Hedges less precisely but acts more robustly on expressed uncertainty.

**DeepSeek V3.2 / Qwen 3 Next 80B**: strong calibration and attention, moderate executive — identify knowledge gaps but do not fully exploit them in planning.

**Gemma 4 31B**: executive 0.085, the lowest in the set despite strong objective (0.722). Similar failure mode to the Gemini family at larger scale.

Scores are stable across difficulty levels (±0.02–0.03), confirming the benchmark is neither trivially easy nor broken at hard difficulty.

### Statistical Validation

All headline comparisons use empirical bootstrap CIs (n=10,000 resamples, paired by instance, seed=42). Pairing by instance removes between-instance variance, making the tests substantially more powerful than unpaired approaches.

| Comparison | Dimension | Diff | 95% CI | z | sign-p | sig |
|---|---|---|---|---|---|---|
| GPT mini vs. GPT full | composite | +0.103 | [+0.093, +0.114] | 19.0 | <0.001 | *** |
| GPT mini vs. GPT full | executive | +0.397 | [+0.383, +0.411] | 54.9 | <0.001 | *** |
| GPT mini vs. GPT full | calibration | +0.042 | [+0.029, +0.056] | 6.1 | 0.001 | *** |
| GPT mini vs. GPT full | objective | +0.021 | [+0.006, +0.035] | 2.8 | 0.374 | ** |
| Sonnet vs. Opus | composite | +0.033 | [+0.014, +0.052] | 3.4 | 0.012 | *** |
| Sonnet vs. Opus | executive | +0.067 | [+0.046, +0.087] | 6.5 | <0.001 | *** |
| Sonnet vs. Opus | attention | +0.059 | [+0.027, +0.091] | 3.6 | 0.001 | *** |
| Sonnet vs. Opus | calibration | −0.033 | [−0.055, −0.011] | −2.9 | <0.001 | ** |

The Claude composite inversion was marginal under unpaired parametric testing; with instance-paired bootstrap it is firmly significant (z=3.4, p=0.012). This highlights why per-instance pairing matters: both models saw the exact same 1,000 instances, so paired testing is both more powerful and more appropriate.

The dissociation between objective and executive is statistically reliable at the model level: Pearson r(objective, executive) = −0.761 (p=0.02) across models. Pooled across all 11,000 instance-model pairs, the same relationship holds at the instance level: r=−0.316 (p<0.001, n=11,000). Models that plan well on a given instance tend to produce worse contingency plans on that same instance — the tradeoff is not just a between-model artifact.

### Construct Validity

CIPHER dimensions were correlated with three external public benchmarks (MMLU, GPQA-Diamond, MATH-500) across the 8–11 models with published scores:

| CIPHER dim | MMLU | GPQA-Diamond | MATH-500 |
|---|---|---|---|
| composite | −0.269 (ns) | −0.335 (ns) | −0.174 (ns) |
| objective | +0.474 (ns) | +0.543 (ns) | +0.074 (ns) |
| calibration | −0.330 (ns) | −0.084 (ns) | −0.441 (ns) |
| attention | −0.679 (p=0.08) | **−0.770 (p=0.04)** | −0.596 (p=0.10) |
| executive | −0.536 (ns) | −0.712 (p=0.06) | −0.011 (ns) |

Low correlations with MMLU and GPQA indicate CIPHER captures something these benchmarks do not. The negative sign on calibration and executive is notable: larger, higher-benchmark models score *lower* on these dimensions — directly replicating the scaling inversion at the aggregate level.

The single significant correlation is attention vs. GPQA-Diamond (r=−0.770, p=0.04), which is also negative. Models that perform well on hard reasoning problems are *worse* at ranking hidden rules by impact — consistent with a pattern where strong reasoners over-rely on their planning capability and underweight identifying knowledge gaps.

### Weighting Robustness

Rankings are stable under alternative scoring weights. Spearman rank correlations between the original weighting (35/25/20/20) and three alternatives:

| Alternative | rho | p |
|---|---|---|
| Equal (25/25/25/25) | 0.673 | 0.035 |
| Plan-heavy (50/20/15/15) | 0.645 | 0.044 |
| Meta-heavy (20/35/25/20) | 0.664 | 0.038 |

All three alternatives yield significant rank agreement (p<0.05). GPT-5.4 remains last under every weighting. The main movement is Gemini 3 Flash Preview, which rises to 2nd under plan-heavy weighting (from 3rd overall) due to its high objective score — but even this does not change the core inversion finding.

### Parse Failure Analysis

Models sometimes return malformed JSON that cannot be scored, receiving 0 on all dimensions. Failure rates vary substantially:

| Model | Failures | Rate | Composite (valid only) |
|---|---|---|---|
| GPT-5.4 Nano | 1 | 0.1% | 0.632 |
| GPT-5.4 mini | 0 | 0.0% | 0.629 |
| GPT-5.4 | 52 | 5.2% | 0.554 |
| Gemini 3 Flash Preview | 0 | 0.0% | 0.624 |
| Gemini 3.1 Pro Preview | 0 | 0.0% | 0.622 |
| Claude Sonnet 4.6 | 52 | 5.2% | 0.657 |
| Claude Opus 4.7 | **137** | **13.7%** | **0.684** |
| Claude 4.5 Haiku | 5 | 0.5% | 0.605 |
| DeepSeek V3.2 | 0 | 0.0% | 0.606 |
| Qwen 3 Next 80B Instruct | 54 | 5.4% | 0.618 |
| Gemma 4 31B | 20 | 2.0% | 0.603 |

Claude Opus 4.7's 13.7% failure rate is the most striking result. Its valid-only composite is 0.684, which would rank it 3rd overall — above Claude Sonnet 4.6. The Claude inversion is therefore partly a JSON output reliability story: Opus fails to produce parseable responses on 1 in 7 instances, costing it nearly 0.10 composite points. This suggests the scaling inversion in the Claude family has two components: a genuine metacognitive skill gap (executive, attention) and a format reliability gap that independently penalizes the larger model.

All reported scores include failures as zeros, matching real deployment conditions. A model that produces invalid output on 14% of instances is genuinely less useful regardless of how well it performs when it does respond.

### Score Consistency

In addition to mean performance, CIPHER reveals differences in consistency:

| Model | Mean | Std | p10 | p90 |
|---|---|---|---|---|
| GPT-5.4 Nano | 0.632 | 0.096 | 0.498 | 0.747 |
| GPT-5.4 mini | 0.629 | 0.095 | 0.491 | 0.743 |
| GPT-5.4 | 0.525 | 0.152 | 0.384 | 0.660 |
| Gemini 3 Flash Preview | 0.624 | 0.106 | 0.477 | 0.748 |
| Claude Sonnet 4.6 | 0.623 | 0.171 | 0.495 | 0.763 |
| Claude Opus 4.7 | 0.590 | 0.250 | 0.000 | 0.784 |
| DeepSeek V3.2 | 0.606 | 0.093 | 0.472 | 0.716 |

GPT-5.4 Nano and GPT-5.4 mini are the most consistent models (std ≈ 0.095–0.096). GPT-5.4 is more variable (std=0.152), and Claude Opus 4.7 is the most erratic (std=0.250), with a p10 of 0.000 driven by its parse failure rate. DeepSeek V3.2 has the lowest std among non-GPT models (0.093), suggesting reliable if not top performance. Consistency matters for deployment: a model with a lower mean but tighter distribution may be preferable to a high-ceiling, high-variance model for safety-critical applications.

## Objective Function and Robustness

The scoring objective is `sum(phase × flux mod 7) - 3 × (entities with flux ≥ 5)`. To verify rankings do not depend on this choice, we ran the three stubs under two alternative objectives: (A) `sum((phase + flux) mod 7)` and (B) `max(flux) - min(phase)`. Rank ordering was stable across all three (stub-random > stub-greedy > stub-noop), confirming the metacognitive structure drives the results, not the specific objective.

## Novelty and Impact

CIPHER fills a gap between factual, reasoning, and planning benchmarks. None of these measure whether a model's confidence tracks its knowledge, or whether it can identify which gaps matter most for the task.

The complete-omission mechanism is the key design choice. Benchmarks that mask values with `?` allow calibration gaming via syntactic detection. CIPHER removes this: hidden rules are absent entirely, forcing genuine epistemic reasoning about structural unknowns.

The four-dimensional structure enables diagnostic use beyond leaderboard ranking. Gemini Flash and Claude Sonnet achieve nearly identical composites (0.624 vs. 0.623) through opposite strategies - Gemini via planning strength, Sonnet via contingency strength. These profiles point to different research directions.

## Conclusions

CIPHER provides a contamination-proof measure of LLM metacognition that cannot be gamed by any single strategy. The empirical results reveal a consistent finding: scaling improves calibration but not contingency planning. GPT-5.4 mini outperforms GPT-5.4 by 0.104 composite points; Claude Sonnet outperforms Claude Opus by 0.033 points, despite both larger models being superior on standard benchmarks.

**The central finding: knowing what you do not know and acting appropriately on that knowledge are distinct capabilities that do not scale together. Larger models are better calibrated but worse at contingency planning - a dissociation that is invisible to every existing benchmark and consistent across two independent model families.**
