"""Report rendering: machine-readable JSON, human-readable HTML, and a terse
Markdown block for the GitHub Actions job summary / alert issue.

The HTML report is intentionally a single self-contained file with no CDN calls —
it has to open from a workflow artifact download on someone's laptop with no
network, which is the moment a monitoring report is most needed.
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

LEVEL_COLORS = {"OK": "#1a7f4b", "WARN": "#b7791f", "ALERT": "#c0392b", "UNKNOWN": "#6b7280"}
ACTION_BLURB = {
    "NO_ACTION": "All monitored signals are within tolerance.",
    "WATCH": "Something moved, but not enough (or not for long enough) to act on.",
    "RECALIBRATE_THRESHOLD": "Re-cut the operating threshold; the model still ranks well.",
    "RETRAIN": "Confirmed degradation. Build and gate a challenger.",
    "HALT": "Data quality failure — do not trust scores from this batch.",
}


def _num(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _json_safe(node):
    """NaN/Infinity are valid Python json output but invalid JSON — and the report is
    parsed by github-script in the alert workflow, which would throw on them."""
    if isinstance(node, dict):
        return {k: _json_safe(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_json_safe(v) for v in node]
    if isinstance(node, float) and not math.isfinite(node):
        return None
    return node


def save_json_report(report: dict, reports_dir: Path, batch_id: str) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{batch_id}.json"
    with open(path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, default=str, allow_nan=False)
    return path


def to_markdown(report: dict) -> str:
    """What lands in the Actions job summary and the alert issue body."""
    decision = report.get("decision", {})
    action = decision.get("action", "UNKNOWN")
    drift = report.get("drift", {})
    prediction = report.get("prediction", {})
    performance = report.get("performance", {})

    lines = [
        f"## Monitoring report — `{report.get('batch_id')}`",
        "",
        f"**Decision: `{action}`** — {ACTION_BLURB.get(action, '')}",
        "",
        f"- Environment: `{report.get('environment')}`",
        f"- Model version: `{report.get('model_version')}`",
        f"- Rows scored: {report.get('n_rows'):,}" if report.get("n_rows") else "- Rows scored: —",
        f"- Evaluated at: {report.get('evaluated_at')}",
        f"- Confirmation streak: {decision.get('confirmed_streak', 0)}"
        + (" (cooldown active)" if decision.get("cooldown_active") else ""),
        "",
        "### Drift",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Features alerting | {drift.get('n_features_alert', 0)} / {drift.get('n_features_tested', 0)} |",
        f"| Max feature PSI | {_num(drift.get('max_psi'))} |",
        f"| Top movers | {', '.join(drift.get('top_drifted', [])[:5]) or '—'} |",
        f"| Score PSI | {_num(prediction.get('score_psi'))} |",
        f"| Flag rate | {_num(prediction.get('flag_rate'), 5)} (ref {_num(prediction.get('reference_flag_rate'), 5)}) |",
        "",
    ]

    lines += ["### Performance", ""]
    if performance.get("status") == "OK":
        metrics = performance.get("metrics", {})
        comparison = performance.get("comparison", {})
        lines += [
            "| Metric | Current | Baseline |",
            "|---|---|---|",
            f"| PR-AUC | {_num(metrics.get('pr_auc'))} | {_num(comparison.get('baseline', {}).get('pr_auc'))} |",
            f"| Recall | {_num(metrics.get('recall'))} | {_num(comparison.get('baseline', {}).get('recall'))} |",
            f"| Precision | {_num(metrics.get('precision'))} | {_num(comparison.get('baseline', {}).get('precision'))} |",
            f"| Frauds missed | {metrics.get('missed_fraud_count', '—')} | — |",
            "",
        ]
    else:
        lines += [
            f"_{performance.get('status')}: {performance.get('reason', 'labels not yet mature')}_",
            "",
        ]

    reasons = decision.get("reasons", [])
    if reasons:
        lines += ["### Why", ""] + [f"- {r}" for r in reasons] + [""]
    return "\n".join(lines)


def to_html(report: dict) -> str:
    decision = report.get("decision", {})
    action = decision.get("action", "UNKNOWN")
    severity = decision.get("severity", "OK")
    drift = report.get("drift", {})
    prediction = report.get("prediction", {})
    performance = report.get("performance", {})
    color = LEVEL_COLORS.get(severity, "#6b7280")

    feature_rows = "".join(
        f"<tr class='{f.get('psi_level','OK').lower()}'>"
        f"<td>{html.escape(str(f.get('feature')))}</td>"
        f"<td>{_num(f.get('psi'))}</td>"
        f"<td><span class='pill' style='background:{LEVEL_COLORS.get(f.get('psi_level'),'#6b7280')}'>"
        f"{html.escape(str(f.get('psi_level')))}</span></td>"
        f"<td>{_num(f.get('ks_statistic'))}</td>"
        f"<td>{'yes' if f.get('ks_significant') else 'no'}</td>"
        f"<td>{_num(f.get('mean_shift_in_ref_sds'), 2)}</td></tr>"
        for f in sorted(
            drift.get("features", []),
            key=lambda f: (-(f.get("psi") or 0) if isinstance(f.get("psi"), (int, float)) else 0),
        )
    )

    reasons = "".join(f"<li>{html.escape(str(r))}</li>" for r in decision.get("reasons", []))
    if performance.get("status") == "OK":
        metrics = performance.get("metrics", {})
        base = performance.get("comparison", {}).get("baseline", {})
        perf_html = f"""
        <table>
          <tr><th>Metric</th><th>Current</th><th>Baseline</th></tr>
          <tr><td>PR-AUC</td><td>{_num(metrics.get('pr_auc'))}</td><td>{_num(base.get('pr_auc'))}</td></tr>
          <tr><td>Recall</td><td>{_num(metrics.get('recall'))}</td><td>{_num(base.get('recall'))}</td></tr>
          <tr><td>Precision</td><td>{_num(metrics.get('precision'))}</td><td>{_num(base.get('precision'))}</td></tr>
          <tr><td>Alerts raised</td><td>{metrics.get('alert_volume','—')}</td><td>—</td></tr>
          <tr><td>Frauds missed</td><td>{metrics.get('missed_fraud_count','—')}</td><td>—</td></tr>
        </table>"""
    else:
        perf_html = (
            f"<p class='muted'>{html.escape(str(performance.get('status','')))}: "
            f"{html.escape(str(performance.get('reason','labels not yet mature')))}</p>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Fraud monitoring — {html.escape(str(report.get('batch_id')))}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 32px; background: #f7f8fa; color: #16181d; }}
  @media (prefers-color-scheme: dark) {{ body {{ background:#111317; color:#e6e8eb; }}
    .card {{ background:#1a1d23 !important; border-color:#2a2f38 !important; }}
    th {{ background:#22262e !important; }} td, th {{ border-color:#2a2f38 !important; }} }}
  .wrap {{ max-width: 1080px; margin: 0 auto; }}
  .card {{ background:#fff; border:1px solid #e3e6ea; border-radius:10px; padding:20px 24px;
           margin-bottom:20px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:0 0 12px; }}
  .muted {{ color:#6b7280; font-size:13px; margin:0; }}
  .banner {{ border-left:5px solid {color}; }}
  .action {{ font-size:26px; font-weight:650; color:{color}; letter-spacing:-0.3px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid #e9ecef; }}
  th {{ background:#f2f4f7; font-weight:600; }}
  .pill {{ color:#fff; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }}
  .kpis {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .kpi {{ flex:1 1 150px; }} .kpi .v {{ font-size:20px; font-weight:600; }}
  ul {{ margin:8px 0 0; padding-left:20px; }} li {{ margin-bottom:6px; }}
  .scroll {{ max-height:520px; overflow:auto; }}
</style></head>
<body><div class="wrap">
  <div class="card banner">
    <p class="muted">{html.escape(str(report.get('environment')))} ·
      model <code>{html.escape(str(report.get('model_version')))}</code> ·
      {html.escape(str(report.get('evaluated_at')))}</p>
    <h1>Monitoring report — {html.escape(str(report.get('batch_id')))}</h1>
    <p class="action">{html.escape(action)}</p>
    <p class="muted">{html.escape(ACTION_BLURB.get(action, ''))}
      Confirmation streak {decision.get('confirmed_streak', 0)}
      {'· cooldown active' if decision.get('cooldown_active') else ''}</p>
  </div>

  <div class="card"><h2>Headline</h2><div class="kpis">
    <div class="kpi"><p class="muted">Rows scored</p><div class="v">{report.get('n_rows', 0):,}</div></div>
    <div class="kpi"><p class="muted">Features alerting</p><div class="v">{drift.get('n_features_alert', 0)}/{drift.get('n_features_tested', 0)}</div></div>
    <div class="kpi"><p class="muted">Max feature PSI</p><div class="v">{_num(drift.get('max_psi'), 3)}</div></div>
    <div class="kpi"><p class="muted">Score PSI</p><div class="v">{_num(prediction.get('score_psi'), 3)}</div></div>
    <div class="kpi"><p class="muted">Flag rate</p><div class="v">{_num(prediction.get('flag_rate'), 4)}</div></div>
  </div></div>

  <div class="card"><h2>Why this decision</h2><ul>{reasons or '<li>All signals within tolerance.</li>'}</ul></div>

  <div class="card"><h2>Performance (labelled window)</h2>{perf_html}</div>

  <div class="card"><h2>Feature drift</h2><div class="scroll"><table>
    <tr><th>Feature</th><th>PSI</th><th>Level</th><th>KS</th><th>KS sig.</th><th>Mean shift (ref SDs)</th></tr>
    {feature_rows}
  </table></div>
  <p class="muted" style="margin-top:10px">PSI bands: &lt;0.10 stable · 0.10–0.25 moderate · &gt;0.25 significant.
  KS p-values Bonferroni-corrected across {drift.get('n_features_tested', 0)} simultaneous tests.</p></div>

  <p class="muted">Generated {datetime.now(timezone.utc).isoformat()} by fraud-monitoring.</p>
</div></body></html>"""


def save_html_report(report: dict, reports_dir: Path, batch_id: str) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{batch_id}.html"
    path.write_text(to_html(report), encoding="utf-8")
    return path
