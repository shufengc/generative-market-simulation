# My Analysis: Current Problems and What They Mean
*Shufeng Chen — personal notes before handing off to Claude Code*

---

## The Big Picture: What Actually Happened

After carefully comparing all branches, here is what the data shows — no exaggeration:

### Finding 1: You were training on 5× less data than everyone else

Our shufeng branch uses `stride=5` in `run_pipeline.py`. Every other branch (main, yixuan, yizheng, yuxia) uses `stride=1`. This was never explicitly decided — it was a default value that diverged between branches.

With stride=5 on 60-day windows over 14 years of data: ~1,059 training windows.
With stride=1 (what everyone else uses): ~5,200+ windows.

This is a **5× difference in training data size**. Our DDPM being competitive with NormFlow (5.0/6 SF in Phase 5) despite less data is arguably a positive sign for the model. But it means the comparison was not fair, and we cannot claim our Phase 1-5 results as definitive.

### Finding 2: The evaluation tests themselves have changed

Yixuan rewrote the stylized facts tests. The changes are not minor:
- Fat tails: The Hill estimator is more statistically rigorous than kurtosis. Under the new test, a model needs to match the *real* tail index, not just exceed kurtosis 3.
- Long memory: Hurst R/S analysis is a better test for long-range dependence than ACF count. The new test also requires matching the *real* Hurst exponent.
- Volatility clustering: GARCH(1,1) is stronger than ARCH-LM.

Under the new tests, all our "4.7/6" and "5.0/6" numbers are meaningless. We do not know if our DDPM vpred still passes these tests or not. **This is the first thing we need to find out.**

### Finding 3: Our core innovation (v-prediction) is unaffected

The good news is that the DDPM model architecture changes (vpred, Student-t noise, temporal attention, etc.) are independent of the data pipeline and evaluation framework. If v-prediction was fundamentally better at capturing financial dynamics, it will still be better under the new evaluation — we just need to re-measure it.

The critical insight from Phase 3 still holds: **sigmoid schedule suppresses fat-tail generation in vpred models**. This is a real algorithmic finding. The new Hill estimator test directly measures fat tails, so this finding is even more relevant now.

---

## What I Think Will Happen When We Re-Run

**My honest prediction (not a guarantee):**

1. With stride=1 data (~5× more training windows), the DDPM model should train better. More data helps generative models avoid overfitting and capture more diverse patterns. I expect SF scores to go **up** not down when we re-run.

2. The new Hurst R/S test for SF4 may be harder to pass than the old ACF count. Generating long-memory in synthetic data is notoriously difficult. SF4 may fail.

3. The new Hill estimator for SF1 (fat tails) — this is where v-prediction has shown the most promise. If the real Hill α is around 2-4 (which is typical for equity returns), our model needs to match this. V-prediction + Student-t noise is our best bet.

4. SF6 (No Autocorrelation) — the MAA test is stricter. DDPM may struggle here.

**Rough expectation:** DDPM vpred baseline might score 3-4/6 under new eval + stride=1. Student-t variant might reach 4-5/6. We will not know until we run it.

---

## What the TA Feedback Means for Our Strategy

Gaurav's advice to prioritize DDPM over everything else was exactly right. The team should focus on getting DDPM results that are convincing, not breadth of models. 

His specific mention of the **leverage effect** is important — SF3 tests whether large drops predict higher future volatility (asymmetric correlation). Our current DDPM model does not have any special mechanism for this. The correlation-based test might pass by coincidence, but it is not something we designed for.

For Phase 5 DDPM refinement (after the re-baseline), if leverage effect is failing:
- Consider conditioning on past realized variance
- The GJR-GARCH model (asymmetric GARCH) that Yizheng added to the regime labels could be leveraged

---

## Advice for the New Claude Code Session

**Do these first, in this order:**
1. Sync `stylized_facts.py` from main (5 minutes)
2. Fix stride in `run_pipeline.py` (2 minutes)
3. Run the data pipeline to regenerate windows with stride=1
4. Run ONE experiment: `vpred=True, cosine schedule, 128ch, 400 epochs`
5. Record the new SF numbers under the new eval

**Do not do these yet:**
- Do not start Phase 6 innovations
- Do not commit
- Do not touch other people's code

**The key question you need answered before Thursday:**
> Under stride=1 data + new eval framework, how many stylized facts does DDPM vpred (cosine, 128ch) pass? And which specific tests fail?

That answer will determine what DDPM refinement we pursue on Thursday.

---

## On the Screenshots You Should Upload to Claude Code

**Yes, upload these screenshots** to the Claude Code session:
1. The WeChat screenshot showing Yixuan's data changes summary (the table with SF1-SF6 changes)
2. The screenshot showing Ye's message about data.4.21.docx and the evaluation change summary

These give Claude Code direct visual confirmation of what changed. The `TEAM_SYNC_STATUS.md` file already captures this textually, but the screenshots provide the original source.

**Do not need to upload:**
- The screenshot with team discussion about merging branches (already summarized)
- The first screenshot showing the Claude project setup (that's for you, not Claude Code)

---

## One More Thing: The Claude Code Project Setup

In the screenshot showing "Use an existing folder" → `EECS4904 Project`:

For the **Instructions** field, you should put something like:

```
This is a financial generative model research project. My branch is 'shufeng' and I am 
working on DDPM improvements. DO NOT commit or push anything to any branch. 
DO NOT modify the main branch. Work locally only. 

Key context: read docs/CLAUDE_CODE_CONTEXT.md and docs/TEAM_SYNC_STATUS.md first 
before doing anything. The team is using stride=1 in the data pipeline but my branch 
accidentally uses stride=5 — this needs to be fixed. The evaluation framework was also 
updated by a teammate and needs to be synced from origin/main.
```

For **Add files**, upload:
- `docs/CLAUDE_CODE_CONTEXT.md`
- `docs/TEAM_SYNC_STATUS.md`
- This file (`docs/SHUFENG_ANALYSIS.md`)
