"""Figures corresponding to disclosure figures 10, 12-15, plus the honest-evaluation charts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CATEGORIES
from . import dataset as D
from .features import ALL_FEATURES

NAMES = {"sev_sum_open": "Open vulnerability severity", "open_vulns": "Unresolved vulnerabilities", "patch_delay_max": "Patch delay (max)", "patch_delay_mean": "Patch delay (mean)",
         "max_sev_open": "Max open severity", "failed_auth": "Failed authentications", "suspicious_access": "Suspicious access", "unauth_changes_since_audit": "Configuration drift",
         "backup_failed": "Backup failure", "windows_since_backup_ok": "Windows since good backup", "compliance_violations": "Compliance violations", "security_alerts": "Security alerts",
         "priv_esc": "Privilege escalation", "windows_since_audit": "Windows since audit", "ts": "Current trust score", "ts_node": "Own-state trust score", "high_risk_connections": "High-risk connections"}
COL = ["#16a34a", "#eab308", "#f97316", "#dc2626"]


def _save(fig, p):
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)


def make_all(m, tkg, base, sp, engine, out):
    out.mkdir(parents=True, exist_ok=True)
    # Fig 12: confusion matrix, test
    cm = np.array(m["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4), CATEGORIES, rotation=30); ax.set_yticks(range(4), CATEGORIES)
    ax.set(xlabel="Predicted", ylabel="Observed", title=f"Fig 12. Confusion matrix, test (n={m['test']['n']})")
    _save(fig, out / "fig12_confusion_matrix.png")

    # Fig 13: feature importance
    imp = list(m["feature_importance"].items())[:14][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.barh([NAMES.get(k, k) for k, _ in imp], [v for _, v in imp], color="#2563eb")
    ax.set(xlabel="Importance (Random Forest, impurity)", title="Fig 13. Feature importance")
    _save(fig, out / "fig13_feature_importance.png")

    # Fig 14: trust evolution of the assets with the largest predicted decay at the last test window
    last = sp.test["window"].max()
    row = sp.test[sp.test["window"] == last]
    P = engine.predict(row)
    top = row.assign(p=P["expected_decay"].values).nlargest(5, "p")["asset"].tolist()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, a in enumerate(top):
        h = base[base["asset"] == a].sort_values("window")
        ax.plot(h["window"], h["ts"], lw=1.4, label=a)
    ax.axvline(sp.bounds["test"][0], color="#9ca3af", ls="--"); ax.text(sp.bounds["test"][0] + .3, 4, "test period", fontsize=8, color="#6b7280")
    ax.set(xlabel="Window (day)", ylabel="Trust score", ylim=(0, 100), title="Fig 14. Trust evolution, assets with highest predicted decay")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    _save(fig, out / "fig14_trust_evolution.png")

    # Fig 15: predicted vs observed decay distribution
    X = sp.test[ALL_FEATURES]
    Pt = engine.predict(sp.test)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    bins = np.linspace(0, 45, 24)
    ax.hist(sp.test["decay"], bins=bins, alpha=.55, label="Observed decay", color="#dc2626")
    ax.hist(Pt["pred_decay"], bins=bins, alpha=.55, label="Predicted decay", color="#2563eb")
    ax.set(xlabel=f"Trust decay over {m['horizon_windows']} window(s), points", ylabel="Asset-windows", title="Fig 15. Distribution of decay, test period")
    ax.legend()
    _save(fig, out / "fig15_decay_distribution.png")

    # Horizon sweep
    rows = m["horizon_sweep"]["rows"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([r["horizon"] for r in rows], [r["val_f1_macro"] for r in rows], "o-", label="Validation macro-F1 (used to choose)")
    ax.plot([r["horizon"] for r in rows], [r["val_auc_high"] for r in rows], "s--", label="Validation AUC (High/Critical)")
    ax.axvline(m["horizon_windows"], color="#9ca3af", ls=":")
    ax.set(xlabel="Prediction horizon (windows)", title="Horizon sweep", ylim=(0, 1)); ax.legend(fontsize=8)
    _save(fig, out / "fig_horizon_sweep.png")

    # Ablation + ranking
    ab = m["ablation"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    names = list(ab)
    axs[0].bar(range(3), [ab[n]["auc_high_or_critical"] for n in names], color=["#93c5fd", "#60a5fa", "#2563eb"])
    axs[0].axhline(0.5, color="#dc2626", ls="--", lw=1); axs[0].text(2.45, 0.51, "chance", color="#dc2626", fontsize=8, ha="right")
    for i, n in enumerate(names):
        axs[0].text(i, ab[n]["auc_high_or_critical"] + .01, f'{ab[n]["auc_high_or_critical"]:.3f}', ha="center", fontsize=8)
    axs[0].set_xticks(range(3), [n.replace("+", "+\n") for n in names], fontsize=8); axs[0].set(ylim=(0, 1), ylabel="AUC (High/Critical)", title="Feature-set ablation (axis starts at 0)")
    rk = {k: v for k, v in m["ranking"].items() if not k.startswith("_")}
    axs[1].barh(list(rk)[::-1], [v["precision_at_k"] for v in rk.values()][::-1], color="#2563eb")
    axs[1].axvline(m["ranking"]["_meta"]["high_or_critical_base_rate"], color="#dc2626", ls="--")
    axs[1].set(xlabel=f"Precision of top {int(m['ranking']['_meta']['k_fraction'] * 100)}% audit list", title="Audit prioritisation")
    axs[1].tick_params(axis="y", labelsize=7)
    _save(fig, out / "fig_ablation_ranking.png")
