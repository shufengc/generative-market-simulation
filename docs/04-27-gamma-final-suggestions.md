# Gamma Deck — Final Polish Suggestions
**Date:** Apr 27, 2026  
**From:** Shufeng  
**To:** Team — whoever is editing Gamma before tomorrow

Hey everyone, great work getting the deck to 48 slides with all the L3/L4 updates applied. I've done a full read-through of the exported PDF and everything substantive looks correct. There are a few small things left that would make it cleaner for the audience — writing them up here so whoever has edit access can work through them. None of these are blockers, but together they add up to a noticeably more polished final deck.

---

## Priority 1 — Content Accuracy (4 quick text fixes)

These are the only places where the slide text contradicts what we've actually done. Worth fixing before the presentation since an instructor might catch them.

---

### Fix 1 — Agenda slide (Slide 2)

**Problem:** Section 04 on the Agenda still reads:
> "Future work: L3 conditional control + L4 downstream utility"

This directly contradicts our conclusion slide (slide 43) which correctly says we delivered L3 and L4 partial.

**Gamma prompt:**
> On the Agenda slide, in section 04 (Conclusion), change the line that says "Future work: L3 conditional control + L4 downstream utility" to "Delivered: L1 + L2 + L3 + L4 (partial). Open problems: 99% Kupiec, calm regime, SF6."

---

### Fix 2 — Four-Layer Framework slide (Slide 5)

**Problem:** The body paragraph says:
> "Our architecture already supports L3 through FRED-based macro conditioning (crisis/normal/calm), and L4 is the next step..."

L3 is now implemented and measured. L4 is partially delivered. The "next step" framing is stale.

**Gamma prompt:**
> On the Four-Layer Utility Framework slide, find the paragraph that says "our architecture already supports L3 through FRED-based macro conditioning (crisis/normal/calm), and L4 is the next step for full use in risk tasks." Replace it with: "We have delivered L1 and L2 fully. L3 is implemented — regime-conditioned generation with confirmed vol ordering (crisis > normal > calm). L4 is partially validated — 95% VaR Kupiec PASS achieved with quantile moment mapping; 99% calibration is the remaining open problem."

---

### Fix 3 — DDPM Reverse Process slide (Slide 16)

**Problem:** The reverse process equations and network description appear **twice** at the bottom of the slide (duplicated block). The "Reverse: p_θ(x_{t−1}|x_t)..." and "Network: 1-D U-Net with sinusoidal time embedding..." lines are repeated verbatim.

**Gamma prompt:**
> On the DDPM Reverse Process slide, remove the duplicate block at the bottom. The slide currently shows the reverse process equation and network description twice. Keep only the first occurrence (the one in the main content area) and delete the second repeated block.

---

### Fix 4 — Bottom Line slide (Slide 45)

**Problem:** The callout at the bottom reads:
> "L3 + L4 are not aspirational — the architecture already supports them"

This is now outdated — they're not just "supported," they're implemented and measured.

**Gamma prompt:**
> On the Bottom Line slide, change the sentence "L3 + L4 are not aspirational — the architecture already supports them" to "L3 + L4 are not aspirational — they are implemented and measured. L3 conditioning is confirmed; L4 reaches 95% Kupiec PASS."

---

## Priority 2 — Table Rendering (2 slides)

Both of these have the same root cause: Gamma set uniform column widths that are too narrow, so numbers like "0.819" or "PASS (p=0.069)" wrap and stack vertically inside cells. The fix is to explicitly tell Gamma to auto-fit column widths and use a smaller table font.

---

### Fix 5 — L3 Regime Results table (Slide 36)

**Problem:** The 7-column regime table has digits stacking vertically in the SF, MMD, Disc, SynVol, and RealVol columns. Every number should fit on a single horizontal line.

**Gamma prompt:**
> On the L3 Conditional Control Results slide, the data table has a formatting problem — column widths are too narrow and numbers are stacking vertically. Rebuild the table with these specifications: use 12pt font for all table content (header and data rows). Auto-fit column widths to content — "Regime" and "n_real" should be wider, shorter columns like "SF" and "MMD" should be narrower. Every number must appear on a single horizontal line within its cell with no line breaks. The table should span about 55% of the slide width on the left. The data is: Regime | n_real | SF | MMD | Disc | SynVol | RealVol. Crisis | 724 | 2/6 | 0.009 | 0.819 | 1.31 | 1.68. Calm | 2,112 | 3/6 | 0.374 | 1.000 | 0.44 | 0.64. Normal | 2,457 | 4/6 | 0.027 | 0.722 | 0.62 | 0.95. Keep the footer note and the Key Findings column on the right exactly as they are.

---

### Fix 6 — L4 VaR Progression table (Slide 40)

**Problem:** The column headers "95% VaR Err", "95% Kupiec", "99% VaR Err", "99% Kupiec" and the cell "PASS (p=0.069)" are wrapping mid-word or mid-number.

**Gamma prompt:**
> On the "L4 — Raw Model VaR Underestimates Risk" slide, the VaR progression table has the same formatting problem — column headers and cell values are wrapping across multiple lines. Rebuild the table with 12pt font, auto-fit column widths, and ensure every cell's content appears on a single line. The table should span the full width of the slide content area. Data: Config | 95% VaR Err | 95% Kupiec | 99% VaR Err | 99% Kupiec. Row 1: Raw model (expF_balanced) | 35.1% | FAIL | 33.3% | FAIL. Row 2: expF + Flat Quantile Mapping | 6.3% | PASS (p=0.069) | 17.8% | FAIL. Keep the footer note "Kupiec LR chi-squared test, α=0.05. All results 3-seed avg." and the chart reference below.

---

## Priority 3 — Slide Titles (7 slides)

These slides currently use generic "(1/2)" / "(2/2)" continuation markers. Since we agreed every word in a title should be meaningful, here are suggested replacements. Each new title tells the audience exactly what the slide is about — helpful especially when the deck is viewed as a PDF or exported to slides.

| Slide | Current Title | Suggested Title |
|-------|--------------|-----------------|
| 6 | Six Stylized Facts of Financial Returns (1/2) | Stylized Facts: Distribution Shape and Persistence (SF1–SF3) |
| 7 | Six Stylized Facts of Financial Returns (2/2) | Stylized Facts: Temporal Dependence and Autocorrelation (SF4–SF6) |
| 26 | DDPM Phase 7 Ablation Study (1/2) — Results Table & Key Findings | Phase 7 Ablation: Results Table and Key Findings |
| 27 | DDPM Phase 7 Ablation Study (2/2) — Visual Analysis | Phase 7 Ablation: SF Coverage and MMD Visual Comparison |
| 32 | Training Details and Convergence (1/2) — Model Configurations | Training Configurations and Loss Convergence |
| 33 | Diagnostic Visualizations (1/2) — Return Distributions & QQ Plots | Diagnostics: Return Distributions and QQ Plots |
| 34 | Diagnostic Visualizations (2/2) — ACF & Correlation Structure | Diagnostics: ACF Decay and Cross-Asset Correlation |

Note on Slide 32: it currently says "(1/2)" but there is no matching "(2/2)" slide. The marker can simply be removed, or the slide can be split if the content is too dense (see Priority 4 below).

**Single Gamma prompt to rename all 7 at once:**
> Rename the following slides — change only the title text, keep all content on each slide exactly as is. (1) "Six Stylized Facts of Financial Returns (1/2)" → "Stylized Facts: Distribution Shape and Persistence (SF1–SF3)". (2) "Six Stylized Facts of Financial Returns (2/2)" → "Stylized Facts: Temporal Dependence and Autocorrelation (SF4–SF6)". (3) The Phase 7 Ablation slide with the results table → "Phase 7 Ablation: Results Table and Key Findings". (4) The Phase 7 Ablation slide with the visual charts → "Phase 7 Ablation: SF Coverage and MMD Visual Comparison". (5) "Training Details and Convergence (1/2)" → "Training Configurations and Loss Convergence". (6) "Diagnostic Visualizations (1/2)" → "Diagnostics: Return Distributions and QQ Plots". (7) "Diagnostic Visualizations (2/2)" → "Diagnostics: ACF Decay and Cross-Asset Correlation".

---

## Priority 4 — Density Check (3 slides to review)

These slides have a lot of content and may need splitting if the font is getting small in Gamma. Please check each one in the editor and split if the body text is below 14pt.

**Slide 22 (Evaluation Framework — Six SF Tests):** All 6 SF tests with formulas and pass criteria on a single slide. If this is squeezed, consider splitting: SF1–SF3 on one slide ("Evaluation Framework: Fat Tails, Clustering, Leverage") and SF4–SF6 on another ("Evaluation Framework: Long Memory, Correlations, Autocorrelation").

**Slide 28 (Decorrelation Regularizer):** Has problem statement, formula, results, insight paragraph, and a heatmap reference all together. If it feels dense, put the formula + results on one slide and the insight + heatmap on the next.

**Slide 42 (L4 QM Details):** Root cause, "what works," honest caveat callout, and open problem section all together. If it feels cramped, put root cause + what works on one slide and the honest caveat + open problem on a second.

**If you need to split any of these, use this prompt as a template (fill in the slide):**

> This slide is too dense for 16:9. Split it into two slides. Do not delete, summarize, or paraphrase any content — every bullet, formula, and caption must survive across both slides. Give each new slide a distinct, descriptive title that tells the audience exactly what that slide covers — no "(1/2)" markers. Split logic: [group the first half of the content on slide 1, the second half on slide 2]. Layout: same template as rest of deck, title top-left, body text ≥ 14pt, no orphan lines.

---

## Deck-Wide 16:9 Formatting Prompt (apply to entire deck)

If there's time for one final full-deck pass in Gamma, this prompt covers all formatting consistency:

> **Format: strict 16:9 widescreen, PowerPoint-export-safe.**
>
> THE SINGLE MOST IMPORTANT RULE: Never delete, summarize, or paraphrase away any content to make a slide fit. If a slide is too dense, split it into 2+ slides with descriptive titles (no "(1/2)" markers — every word in the title should be meaningful).
>
> Splitting thresholds: more than 6 body bullets → split. Table with 4+ columns AND body text below it → split. Figure AND more than 3 bullets of commentary → split (figure first, commentary second). Body text font would need to go below 14pt to fit → split instead.
>
> Layout rules for every slide: title top-left, same size/weight/color throughout. Visual hierarchy: title → key finding (bold or accent) → supporting details → figure/table. Tables must fit fully within slide bounds with auto-fit column widths and ≥ 12pt font — no vertical stacking of digits. Figures proportionally sized (≥ 40% slide width), never stretched, no "reserve space for" placeholders.
>
> No Gamma-only effects: no parallax, scroll, floating cards, overlapping layers. Only standard text boxes, images, tables, and simple shapes (PowerPoint-export-safe).
>
> Consistency: same font (Inter or Arial), body text ≥ 14pt always. Green for PASS/Delivered, red for FAIL/Limitation, amber for caveats. Same bullet style, spacing, indentation throughout. Same table header style everywhere. Slide number footer on every slide.

---

## Summary Checklist

| # | Slide | Issue | Priority |
|---|-------|-------|----------|
| 1 | 2 | Agenda says "Future work: L3+L4" | High |
| 2 | 5 | Body says L3 "supported," L4 "next step" | High |
| 3 | 16 | Duplicate equation/network text block | Medium |
| 4 | 45 | "Architecture supports L3+L4" (stale) | Medium |
| 5 | 36 | Table columns too narrow, digits stack vertically | High |
| 6 | 40 | Same table rendering issue | High |
| 7–13 | 6,7,26,27,32,33,34 | Generic "(1/2)" titles | Medium |
| 14 | 22 | Potentially too dense for 16:9 | Low |
| 15 | 28 | Potentially too dense for 16:9 | Low |
| 16 | 42 | Potentially too dense for 16:9 | Low |
| 17 | 32 | "(1/2)" with no matching "(2/2)" | Low |

Thank you all — the deck is in great shape and these are genuinely small things. Let me know if anything is unclear. 加油！
