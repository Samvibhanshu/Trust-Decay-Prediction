# IS Trust Decay Prediction with a Temporal Knowledge Graph

[![tests](https://github.com/Samvibhanshu/Trust-Decay-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Samvibhanshu/Trust-Decay-Prediction/actions/workflows/ci.yml)

A runnable prototype of the VIT IPR disclosure *"AI-Driven Information System Trust Decay Prediction Using Temporal Knowledge
Graphs for Continuous Audit Intelligence"* (TRL 4): audit events -> Temporal Knowledge Graph -> trust features -> Random Forest
trust-decay prediction -> explainable audit prioritisation -> historical repository, with a web dashboard and JSON API.

**Everything here runs on a simulated enterprise.** The assets, audit events and trust scores are synthetic (as in the disclosure).
Results describe how predictable *this simulator* is. They are not evidence about real organisations.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q          # 68 tests, ~25 s (trains a small model itself)
python train.py              # optional: retrain + re-evaluate (~2.5 min); artifacts/ already contains a trained model
python run.py                # dashboard on http://127.0.0.1:5000
```

Set `TRUSTDECAY_API_KEY` to require an `X-API-Key` header on POST endpoints. `TRUSTDECAY_DB` sets the SQLite repository path.
(`node` is only needed for the test that syntax-checks the pages' JavaScript.)

## What was measured

Simulated enterprise: 48 assets x 64 daily windows, ~19k events. Chronological split by window (train 0-29, validation 35-41,
test 47-58, with purged windows at each boundary). Prediction horizon 5 days, chosen on **validation** macro-F1 only.
The disclosure's 80.95% (168 records) uses its own simulated data and labelling and is **not comparable**.

| Test result (n = 576) | Value |
|---|---|
| Accuracy | **0.477**, below the 0.526 majority-class baseline |
| Macro-F1 / weighted F1 | 0.290 / 0.436 |
| AUC for High or Critical decay | 0.682 (rolling-origin CV: 0.705 +/- 0.045) |
| Per-class recall Low / Moderate / High / Critical | 0.78 / 0.03 / 0.15 / 0.28 |
| Errors that are exactly one category off | 41% (the disclosure says most; here they are not) |
| Future trust score, R^2 | 0.171 (persistence baseline: -0.153) |

**Where it helps: ranking assets for audit.** Audit the top 10% (5 assets) each day; share that truly showed High/Critical decay:

| Ranking | Precision@5 | 95% interval over test days |
|---|---|---|
| Random (expected) | 0.306 | |
| Lowest current trust first | 0.033 | 0.00-0.08 |
| Most open vulnerabilities first | 0.083 | 0.00-0.17 |
| Steepest recent decline first | 0.167 | 0.08-0.27 |
| Model P(High or Critical) | 0.433 | 0.32-0.53 |
| Model rank score (adds criticality, audit age) | 0.467 | 0.33-0.58 |
| Dashboard order (priority floors, then rank score) | 0.433 | 0.28-0.57 |

The model lists beat chance and the naive rankers. The dashboard order's interval includes chance, so treat that as unproven.

**Other checks**

- Feature-set ablation (AUC): asset's own state 0.672, + temporal history 0.659, + graph 0.682. Not a clear gain.
- Graph-feature sanity check (3 simulated enterprises): AUC gain +0.013 (sd 0.010) when the simulator links connected assets vs -0.004 (sd 0.015) when it does not. Directionally right, within noise.
- Logistic regression is competitive (AUC 0.652, macro-F1 0.306): the Random Forest's extra flexibility is not shown to matter.
- Unseen assets (12 held out of training): AUC 0.696 vs 0.673 on seen assets, so it transfers across assets in this simulator.
- Control: a label that is **not** forward-looking (the change over the *past* window) is reproduced at 99.0% accuracy. High accuracy is easy when the label is computed from data the model sees; it means little unless the label is in the future.

`artifacts/metrics.json` holds every number; the **Model & evidence** page reads it live and generates its conclusions from it.

## Architecture: disclosure -> code

| Disclosure element | Module |
|---|---|
| Step 1 Audit data collection; claim 2 continuous updates | `simulator.py` (synthetic stream), `service.ingest`, `POST /api/events` |
| Step 2 Preprocessing (validation, timestamp sync, dedup, ordering) | `schema.py` (validation, UTC normalisation), `tkg.add_event` (dedup, late-arrival ordering) |
| Step 3 Temporal Knowledge Graph, (S, R, O, T) | `tkg.py` (networkx; validity intervals on structural edges; hashed snapshots) |
| Step 4 / claim 3 Trust features | `features.py` (node, temporal-history and graph/relationship feature groups) |
| Trust score, TDS = max(0, TS_t - TS_t+h), categories 5/15/30 | `trust.py`, `config.py` |
| Step 5 / claim 4 Prediction engine | `models.py` (RF classifier + RF regressor), `evaluation.py` |
| Step 6 / claim 5 Audit intelligence | `prioritize.py`, `webapp.py`, `templates/` |
| Claim 6 Explainability | `explain.py` (observed rubric penalties + model attribution by counterfactual substitution) |
| Step 7 / claim 7 Repository, predicted-vs-observed | `store.py` (SQLite), `service.reconcile`, `/repository` |
| Figures 10, 12-15 | `figures.py` -> `artifacts/figures/` |

Deviations from the disclosure: Flask instead of Streamlit/Dash; networkx only (no Neo4j); SQLite instead of PostgreSQL/MongoDB.
The disclosure itself states it uses graph-derived features with a Random Forest, not a temporal GNN; so does this.

## Design choices that matter

- **No look-ahead.** The vector for window t uses only events up to t (tested by rebuilding the graph truncated at t and comparing).
  Labels come from window t+h. Splits are by window, with purging so no training label reaches a later split's inputs.
- **Honest selection.** Horizon and hyperparameters are chosen on validation; test columns in the horizon sweep are informational.
  The deployed model is fitted on train+validation only.
- **Dashboard labels each window** as train / validation / test / live, since in-sample windows look much better than new data.
- **Two views of "why".** *Observed* penalties come straight from the rubric (evidence). The *model attribution* shows what the
  prediction reacts to (not proven cause).
- **Priority floors.** An asset already near the bottom of the trust scale can't decay much, so a decay forecast alone ignores it
  (ranking by lowest current trust scores 0.033). Assets with trust < 15 get at least High, < 30 at least Medium. This is a
  design decision, not a fitted one, and it lowers ranking precision slightly (0.467 -> 0.433).

## Known limitations

- **Synthetic data.** Event rates, degradation episodes, the contagion effect and the trust rubric are all my assumptions. The
  rubric is not a real audit standard; replace it with an organisation's policy. Class mix (56/15/15/15%) came out of those
  assumptions and was tuned only to have enough rare-class examples.
- **Weak classifier.** Moderate decay is almost never identified (recall 3%); Critical is right about a quarter of the time.
  Do not present the category as accurate.
- **Volatile scores.** Simulated trust scores swing by tens of points day to day (see Fig 14), which limits predictability at short
  horizons. Real audit data may be smoother or not.
- **Graph value unproven.** Temporal and graph features add little here; the contagion effect was put into the simulator by me,
  so detecting it shows the pipeline can, not that real enterprises have it.
- **Small test set.** 12 test days x 48 assets; intervals are wide (the ranking interval spans ~20 percentage points).
- **Same-asset deployment assumed.** The rolling-origin and new-asset checks help, but nothing here tests drift or a changed enterprise.
- **Not production.** No authentication beyond an optional API key, single process, in-memory caches, no live SIEM integration.

## API

`GET /api/assess?window=&type=&category=&priority=&explain=` - prioritised audit list (default: latest window)
`POST /api/assess {window?}` - assess and log a run to the repository
`GET /api/assets/<id>` - history, forecasts, observed penalties, model factors, recent events
`GET /api/graph/<id>?window=&back=` - temporal graph neighbourhood
`POST /api/events {events:[{s,r,o,t,attrs}]}` - ingest audit events (validated, deduplicated; rejects reported per event)
`GET /api/reconcile`, `/api/repository/{runs,snapshots,predictions,counts}`, `/api/model/metrics`, `/health`

Example event: `{"s":"VULN-901","r":"detected_on","o":"FW-01","t":"2026-03-06T03:00:00Z","attrs":{"severity":9}}`
