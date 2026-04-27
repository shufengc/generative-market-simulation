"""
run_multiseed_eval.py
=====================
Multi-seed evaluation for robustness testing.

Runs full regime evaluation + VaR backtest for a given checkpoint across
multiple seeds to confirm results are not a lucky single draw.

Usage:
    python3 experiments/run_multiseed_eval.py --tag expF_balanced --seeds 42 123 456
    python3 experiments/run_multiseed_eval.py --tag expF_balanced_ftcalm --seeds 42 123 456

Outputs:
    experiments/results/conditional_ddpm_v2/{tag}/multiseed/
        multiseed_summary.json   -- per-seed and aggregated mean±std metrics
        multiseed_var.json       -- per-seed VaR/Kupiec results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.ddpm_improved import ImprovedDDPM                   # noqa: E402
from src.data.regime_labels import get_regime_conditioning_vectors  # noqa: E402
from src.evaluation.metrics import (                                 # noqa: E402
    maximum_mean_discrepancy,
    discriminative_score,
)
from src.evaluation.stylized_facts import run_all_tests             # noqa: E402
from scipy.stats import chi2 as _chi2, kurtosis as sp_kurt         # noqa: E402

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(ROOT, "data")
REGIMES  = ["crisis", "calm", "normal"]
REGIME_INT = {"crisis": 1, "calm": 2, "normal": 0}
CONFIDENCE_LEVELS = [0.95, 0.99]


def kupiec_lr(real_pnl: np.ndarray, var_syn: float, conf: float) -> dict:
    n = len(real_pnl)
    n_exc = int((-real_pnl > var_syn).sum())
    p_hat = n_exc / n
    p_0   = 1.0 - conf
    if n_exc == 0 or n_exc == n:
        lr_stat, p_value = 0.0, 1.0
    else:
        lr_stat = 2.0 * (
            n_exc * np.log(p_hat / p_0)
            + (n - n_exc) * np.log((1.0 - p_hat) / (1.0 - p_0))
        )
        p_value = float(1.0 - _chi2.cdf(lr_stat, df=1))
    return {
        "hit_rate":    round(p_hat, 4),
        "nominal":     p_0,
        "kupiec_pass": p_value > 0.05,
        "p_value":     round(p_value, 4),
    }


def load_model(ckpt_path: str, n_features: int,
               use_aux_sf: bool = True, student_t_df: float = 5.0) -> ImprovedDDPM:
    model = ImprovedDDPM(
        n_features=n_features, seq_len=60, cond_dim=5, T=1000,
        base_channels=128, channel_mults=(1, 2, 4),
        use_vpred=True, use_student_t_noise=True,
        student_t_df=student_t_df,
        use_aux_sf_loss=use_aux_sf,
        cfg_drop_prob=0.1,
        device=DEVICE,
    )
    model.load(ckpt_path)
    return model


def eval_seed(model: ImprovedDDPM, windows: np.ndarray, window_regimes: np.ndarray,
              seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    regime_vecs = get_regime_conditioning_vectors()
    regime_results: dict = {}

    for regime_name in REGIMES:
        syn = model.generate(
            n_samples=1000, use_ddim=True, ddim_steps=50,
            guidance_scale=2.0, ddim_eta=0.3,
            cond=regime_vecs[regime_name],
        )
        mask = window_regimes == REGIME_INT[regime_name]
        real = windows[mask]
        n = min(len(real), 1000)
        r, s = real[:n], syn[:n]
        mmd  = float(maximum_mean_discrepancy(r.reshape(n, -1), s.reshape(n, -1)))
        disc = float(discriminative_score(r, s))
        sf_list = run_all_tests(s)
        sf_count = sum(1 for x in sf_list if x.get("pass", False))
        syn_flat = s.reshape(-1, s.shape[-1])
        real_flat = r.reshape(-1, r.shape[-1])
        regime_results[regime_name] = {
            "sf_count": sf_count,
            "mmd": round(mmd, 5),
            "disc": round(disc, 4),
            "syn_vol": round(float(np.std(syn_flat, axis=0).mean()), 4),
            "real_vol": round(float(np.std(real_flat, axis=0).mean()), 4),
            "syn_kurt": round(float(sp_kurt(s.flatten(), fisher=True)), 3),
        }

    # VaR backtest (unconditional)
    syn_uncond = model.generate(
        n_samples=5000, use_ddim=True, ddim_steps=50,
        guidance_scale=1.0, ddim_eta=0.0, cond=None,
    )
    weights = np.ones(windows.shape[-1]) / windows.shape[-1]
    real_pnl = (windows * weights[None, None, :]).sum(-1).sum(-1)
    syn_pnl  = (syn_uncond * weights[None, None, :]).sum(-1).sum(-1)
    var_results = {}
    for conf in CONFIDENCE_LEVELS:
        var_r = float(-np.quantile(real_pnl, 1 - conf))
        var_s = float(-np.quantile(syn_pnl,  1 - conf))
        err   = abs(var_s - var_r) / (abs(var_r) + 1e-8) * 100
        var_results[f"{conf:.0%}"] = {
            "VaR_real": round(var_r, 4),
            "VaR_syn":  round(var_s, 4),
            "err_pct":  round(err, 2),
            "kupiec":   kupiec_lr(real_pnl, var_s, conf),
        }

    return {"regime": regime_results, "var": var_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",   type=str, required=True,
                        help="Experiment tag (e.g. expF_balanced)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--use-aux-sf", action="store_true", default=True,
                        help="Load model with aux_sf_loss (default True)")
    args = parser.parse_args()

    ckpt_path = os.path.join(ROOT, "checkpoints", f"ddpm_conditional_{args.tag}.pt")
    if not os.path.exists(ckpt_path):
        # Fallback to v1 for baseline comparisons
        ckpt_path = os.path.join(ROOT, "checkpoints", "ddpm_conditional.pt")
        print(f"Checkpoint for tag '{args.tag}' not found; falling back to v1: {ckpt_path}")

    windows = np.load(os.path.join(DATA_DIR, "windows.npy"))
    window_regimes = np.load(os.path.join(DATA_DIR, "window_regimes.npy"))
    n_features = windows.shape[2]

    print(f"Loading model from: {ckpt_path}")
    print(f"Device: {DEVICE}  Seeds: {args.seeds}")
    model = load_model(ckpt_path, n_features, use_aux_sf=args.use_aux_sf)

    all_results = {}
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        all_results[str(seed)] = eval_seed(model, windows, window_regimes, seed)
        for rn, rv in all_results[str(seed)]["regime"].items():
            print(f"  {rn}: SF={rv['sf_count']}/6  Disc={rv['disc']:.3f}  Vol={rv['syn_vol']:.4f}")
        for ck, cv in all_results[str(seed)]["var"].items():
            kp = cv["kupiec"]
            print(f"  VaR@{ck}: err={cv['err_pct']:.1f}%  hit={kp['hit_rate']:.4f}  "
                  f"p={kp['p_value']:.4f}  {'PASS' if kp['kupiec_pass'] else 'FAIL'}")

    # Aggregate across seeds
    def _mean_std(key_path: list, is_pass: bool = False):
        vals = []
        for seed_res in all_results.values():
            cur = seed_res
            for k in key_path:
                cur = cur[k]
            vals.append(float(cur))
        if is_pass:
            return {"pass_rate": round(sum(vals) / len(vals), 3)}
        return {"mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4)}

    summary = {
        "tag": args.tag, "seeds": args.seeds, "n_seeds": len(args.seeds),
        "per_seed": all_results,
        "aggregate": {
            rn: {
                "sf_count":  _mean_std(["regime", rn, "sf_count"]),
                "disc":      _mean_std(["regime", rn, "disc"]),
                "syn_vol":   _mean_std(["regime", rn, "syn_vol"]),
            } for rn in REGIMES
        } | {
            f"var_{ck}": {
                "err_pct":     _mean_std(["var", ck, "err_pct"]),
                "kupiec_pass": _mean_std(["var", ck, "kupiec", "kupiec_pass"], is_pass=True),
            } for ck in ["95%", "99%"]
        },
    }

    out_dir = os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2",
                           args.tag, "multiseed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multiseed_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print aggregated summary
    print("\n" + "=" * 70)
    print(f"MULTI-SEED SUMMARY  tag={args.tag}  seeds={args.seeds}")
    print("=" * 70)
    for rn in REGIMES:
        a = summary["aggregate"][rn]
        print(f"  {rn:8s}: SF={a['sf_count']['mean']:.2f}±{a['sf_count']['std']:.2f}  "
              f"Disc={a['disc']['mean']:.3f}±{a['disc']['std']:.3f}  "
              f"Vol={a['syn_vol']['mean']:.4f}±{a['syn_vol']['std']:.4f}")
    for ck in ["95%", "99%"]:
        ak = summary["aggregate"][f"var_{ck}"]
        print(f"  VaR@{ck}: err={ak['err_pct']['mean']:.1f}%±{ak['err_pct']['std']:.1f}%  "
              f"Kupiec pass_rate={ak['kupiec_pass']['pass_rate']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
