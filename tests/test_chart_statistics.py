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


def test_bechdel_by_decade_uses_confidence_intervals():
    app = load_app_module()
    fig = app.bechdel_by_decade(
        pd.DataFrame(
            {
                "title": [f"F{i}" for i in range(8)],
                "decade": ["1980s"] * 4 + ["1990s"] * 4,
                "bechdel_test": [True, False, True, False, True, True, False, False],
            }
        ),
        app.Controls("mean", 1, "decade", 12, 1, True, False, False, (1980, 1999)),
    )

    assert len(fig.data) == 1
    assert tuple(fig.data[0].error_y.array) == tuple(fig.data[0].error_y.arrayminus)
    assert all(value >= 0 for value in fig.data[0].error_y.array)


def test_protagonist_gender_stacks_female_ensemble_male():
    app = load_app_module()
    fig = app.stacked_percent(
        pd.DataFrame(
            {
                "title": ["A", "B", "C"],
                "decade": ["2000s", "2000s", "2000s"],
                "protagonist_gender": ["male", "female", "ensemble"],
            }
        ),
        "decade",
        "protagonist_gender",
        "Protagonist Gender",
        app.Controls("mean", 1, "decade", 12, 1, True, False, False, (2000, 2009)),
    )

    assert [trace.name for trace in fig.data] == ["female", "ensemble", "male"]


def test_female_dialogue_reference_metrics_include_bias_and_p_values():
    app = load_app_module()
    matched = pd.DataFrame(
        {
            "local_female_dialogue_share_pct": [10, 30, 60, 90, 95],
            "reference_female_dialogue_share_pct": [12, 35, 55, 85, 100],
            "female_dialogue_error_pct": [-2, -5, 5, 5, -5],
        }
    )

    metrics = app.female_dialogue_reference_metrics(matched)

    assert metrics["mean_error"] == -0.4
    assert 0 <= metrics["pearson_p"] <= 1
    assert 0 <= metrics["spearman_p"] <= 1


def test_female_dialogue_majority_chart_counts_films_by_decade():
    app = load_app_module()
    fig = app.female_dialogue_majority_by_decade(
        pd.DataFrame(
            {
                "title": ["A", "B", "C", "D"],
                "decade": ["1990s", "1990s", "2000s", "2000s"],
                "female_dialogue_share_pct": [51, 50, 75, 20],
            }
        ),
        app.Controls("mean", 1, "decade", 12, 1, True, False, False, (1990, 2009)),
    )

    assert list(fig.data[0].y) == [1, 1]


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
