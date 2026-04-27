# WeChat Team Summary — Apr 27, 2026

*Ready to copy-paste into group chat. Send before starting PPT work.*

---

大家好，Shufeng 这里 — 明天 presentation 前快速同步一下进展 📋

---

**【Shufeng 已完成的工作】**

1. **DDPM baseline + improved** (v-prediction + Student-t forward process)
   → SF = 5/6，MMD = 0.006 ← 5个模型中最好成绩

2. **7阶段消融实验 (Phase 1–7)**，3 seeds 可重复
   关键发现：sigmoid schedule 和 v-prediction 会相互抑制（Phase 3），cosine schedule 才是正确搭配

3. **L3 条件生成**：Crisis / Calm / Normal 三个 regime，classifier-free guidance
   - Crisis: SF=4/6, Disc=0.814
   - Normal: SF=4/6, Disc=0.782
   - Calm: SF=3/6, Disc=1.000（已知局限，结构性问题）
   - conditioning 方向验证通过：vol 排序 crisis > normal > calm ✓

4. **L4 下游可用性** (VaR/CVaR backtest, Kupiec coverage test)
   - 95% VaR：误差 6.3%，Kupiec **PASS** (p=0.071) ← 项目首次通过
   - 99% VaR：误差 17.8%，Kupiec FAIL（根因：vol compression，已诊断）

5. **5模型统一对比**（stride=1，新 eval，3 seeds）：
   DDPM > NormFlow > TimeGAN > GARCH ≈ VAE

6. README、docs/、所有实验报告已更新，shufeng branch 已 fast-forward merge 到 main ✓

---

**【Cross-Model Final Results (3-seed, unified settings)】**

| Model | SF | MMD | Disc |
|---|---|---|---|
| DDPM vpred+Student-t | **5/6** | **0.006** | 0.85 |
| NormFlow RealNVP | 5/6 | 0.027 | 0.73 |
| TimeGAN | 4/6 | 0.110 | 1.00 |
| VAE improved | 1/6 | 0.020 | 0.75 |
| GARCH baseline | 1.3/6 | 0.042 | 1.00 |

---

**【Branch 状态】**

- `shufeng` → 已 merge 到 `main` ✓（fast-forward，无冲突）
- 没有未提交的改动
- 其他 branch 的实验代码暂时不需要合并（明天前不影响 presentation）

---

**【明天 Presentation — 建议分工】**

我现在开始做 PPT slides + script，会 cover：
- DDPM 改进故事（Phase 1→7，SF 1.7→5.0）
- 5模型对比
- L3/L4（条件生成 + VaR backtest）
- KEY findings（见下）

**请大家各自准备自己模型的 1–2 张 slide（介绍 + 结果表）：**
- Yixuan：TimeGAN + 评估框架介绍
- Yizheng：VAE（改进版结果，注意用 unified eval 的数字）
- Yuxia：NormFlow + 跨模型雷达图
- Kevin：GARCH baseline + 为什么需要 deep generative models

---

**【PPT 中可以 highlight 的 KEY findings】**

1. **v-prediction + Student-t noise 是关键创新**：SF 从 1.7/6 提升到 5.0/6
2. **Sigmoid schedule 和 v-prediction 相互抑制**（Phase 3 发现）— 一个 non-obvious insight
3. **真实数据只通过 3/6 SF** → 5/6 是经验上限（calibration discovery，Yixuan 贡献）
4. **SF6 在数学上极难通过**：Ljung-Box 20 lag 联合检验，即使真正的白噪声也只有 36% 概率全部通过（0.95^20）—— 不是模型失败，是测试本身的问题
5. **L3 conditioning 方向正确**：vol 排序完全正确 crisis > normal > calm（3.93x 比率）
6. **L4 首次 95% Kupiec PASS**，根因 vol compression 已诊断，有清晰的 next-step roadmap

---

*docs 里有完整的实验报告和结果，有问题随时问！加油大家！🚀*
