"""python train.py [--quick]  trains, evaluates honestly, writes artifacts/ (model, metrics.json, enterprise.json.gz, figures)."""
import argparse

from trustdecay import evaluation

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="smaller forests and fewer seeds (a few minutes less)")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
m = evaluation.run(quick=a.quick, seed=a.seed)
t, d = m["test"], m["data"]
print(f"\nModel {m['version']}  ({m['seconds']} s)   SIMULATED data: {d['assets']} assets x {d['windows']} windows, {d['events']:,} events")
print(f"Horizon chosen on validation: {m['horizon_windows']} window(s) = {m['horizon_windows'] * m['window_hours']} h")
print(f"Class mix: " + ", ".join(f"{k} {v:.0%}" for k, v in d["class_mix"].items()))
print(f"\nTest (chronological, n={t['n']}):  accuracy {t['accuracy']:.3f}  (majority class {t['majority_class_accuracy']:.3f})   "
      f"weighted P/R/F1 {t['precision_weighted']:.3f}/{t['recall_weighted']:.3f}/{t['f1_weighted']:.3f}   macro-F1 {t['f1_macro']:.3f}   AUC(High+Critical) {t['auc_high_or_critical']:.3f}")
print(f"Errors that are one category off: {t['adjacent_error_share']:.0%}")
print("\nAudit list quality (precision of top 10%% per window; base rate of High/Critical %.2f):" % m["ranking"]["_meta"]["high_or_critical_base_rate"])
for k, v in m["ranking"].items():
    if not k.startswith("_"):
        print(f"  {k:52s} {v['precision_at_k']:.3f}")
print("\nFeature-set ablation (AUC High+Critical):", {k: round(v["auc_high_or_critical"], 3) for k, v in m["ablation"].items()})
c = m["graph_signal_control"]
print("Graph-feature AUC gain: with relationship signal %+.3f, control world without %+.3f" % (
    c["with_relationship_signal"]["auc_gain"]["mean"], c["control_no_relationship_signal"]["auc_gain"]["mean"]))
print("\n" + m["disclosure_comparison"]["note"])
n = m["non_predictive_control"]
print(f"Control: a NON-forward-looking label ({n['label']}) is reproduced at {n['accuracy']:.1%} accuracy. High accuracy is easy when the label is computed from data the model sees.")
