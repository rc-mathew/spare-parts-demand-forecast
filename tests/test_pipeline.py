import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import make_sample_data as gen  # noqa: E402
from src import pipeline as P  # noqa: E402


def test_end_to_end_and_no_leakage(tmp_path):
    df = P.clean(gen.make(n_weeks=80, lines_per_week=120))
    m = P.run(df, out=str(tmp_path / "o"), models=str(tmp_path / "m"), top_items=6)
    fc = __import__("pandas").read_csv(tmp_path / "o" / "forecast.csv")
    assert len(fc) == 6 * P.H and fc["predicted_demand"].ge(0).all() and fc.notna().all().all()
    assert m["n_test_rows"] > 0

    # leakage: every feature of a week must be unchanged if the last H weeks of demand are altered
    panel, last = P.weekly_panel(df, top_items=6)
    base = P.add_features(panel)
    edited = panel.copy()
    edited.loc[edited["week"] > last - __import__("pandas").Timedelta(weeks=P.H - 1), "demand"] += 99
    ed = P.add_features(edited)
    t = last - __import__("pandas").Timedelta(weeks=P.H)  # week t only sees demand up to t-H
    cols = [c for c in P.FEATURES if c not in ("item_c", "centre_c")]
    a, b = base[base["week"] <= t][cols].fillna(-1), ed[ed["week"] <= t][cols].fillna(-1)
    assert a.equals(b)
