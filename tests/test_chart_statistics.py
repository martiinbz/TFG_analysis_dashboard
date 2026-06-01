import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit" / "app.py"


def load_app_module():
    spec = spec_from_file_location("dashboard_app", APP_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bootstrap_confidence_interval_returns_bounds():
    app = load_app_module()
    df = pd.DataFrame(
        {
            "decade": ["1960s"] * 4 + ["1970s"] * 4,
            "metric": [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    out = app.bootstrap_ci_by_decade(df, "metric", value_mode="mean", iterations=50, random_state=7)

    assert {"decade", "value", "ci_low", "ci_high"} <= set(out.columns)
    assert out["decade"].tolist() == ["1960s", "1970s"]
    assert (out["ci_low"] <= out["value"]).all()
    assert (out["value"] <= out["ci_high"]).all()


def test_stacked_percent_uses_visible_categories_as_denominator():
    app = load_app_module()
    df = pd.DataFrame(
        {
            "title": ["A", "B", "C", "D"],
            "decade": ["1960s", "1960s", "1960s", "1960s"],
            "class_representation": ["working", "wealthy", "unknown", "unknown"],
        }
    )
    controls = app.Controls("mean", 1, "decade", 12, 1, True, False, False, (1960, 1969))

    fig = app.stacked_percent(df, "decade", "class_representation", "Class Representation", controls)
    visible_share = sum(float(y) for trace in fig.data for y in trace.y)

    assert visible_share == 100.0
    assert fig.layout.barmode == "stack"


def test_heatmap_percent_uses_visible_categories_as_denominator():
    app = load_app_module()
    df = pd.DataFrame(
        {
            "primary_genre": ["Drama", "Drama", "Comedy", "Comedy"],
            "class_representation": ["working", "unknown", "wealthy", "unknown"],
        }
    )
    controls = app.Controls("mean", 1, "primary_genre", 12, 1, True, False, False, (1960, 1969))

    fig = app.heatmap_percent(df, "primary_genre", "class_representation", "Class Representation", controls)

    assert list(fig.data[0].z) == [100.0, 100.0]


def test_stacked_percent_keeps_none_as_a_real_category():
    app = load_app_module()
    df = pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "decade": ["1960s", "1960s", "1960s"],
            "strong_language_frequency": ["none", "occasional", "frequent"],
        }
    )
    controls = app.Controls("mean", 1, "decade", 12, 1, True, False, False, (1960, 1969))

    fig = app.stacked_percent(df, "decade", "strong_language_frequency", "Strong Language", controls)
    categories = {trace.name for trace in fig.data}

    assert categories == {"none", "occasional", "frequent"}


def test_bootstrap_intervals_use_error_bars_without_extra_traces():
    app = load_app_module()
    fig = app.rate_line(
        pd.DataFrame({"title": ["A"], "decade": ["1940s"], "drug_culture_presence": [True]}),
        "drug_culture_presence",
        True,
        "Drug Culture",
        app.Controls("mean", 1, "decade", 12, 1, True, False, False, (1940, 1949)),
    )

    assert len(fig.data) == 1
    assert fig.data[0].error_y.visible is None
    assert tuple(fig.data[0].error_y.array) == (0.0,)
    assert tuple(fig.data[0].error_y.arrayminus) == (0.0,)


def test_violin_percentage_axis_stays_in_percent_range():
    app = load_app_module()
    fig = app.violin(
        pd.DataFrame(
            {
                "bechdel_test": [True, True, False, False],
                "female_dialogue_share_pct": [20, 80, 5, 95],
            }
        ),
        "bechdel_test",
        "female_dialogue_share_pct",
        "Bechdel vs Female Dialogue",
        app.Controls("mean", 1, "decade", 12, 1, True, False, False, (1900, 2026)),
    )

    assert fig.layout.yaxis.range == (0, 100)
    assert all(trace.spanmode == "hard" for trace in fig.data)
