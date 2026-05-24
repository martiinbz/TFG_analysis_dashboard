from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit" / "app.py"


def function_body(source: str, name: str) -> str:
    marker = f"def {name}("
    start = source.index(marker)
    next_def = source.find("\ndef ", start + len(marker))
    return source[start:] if next_def == -1 else source[start:next_def]


def test_explorer_is_custom_chart_only():
    source = APP_PATH.read_text(encoding="utf-8")
    body = function_body(source, "explorer_dashboard")

    assert "custom_chart(df, controls)" in body
    assert "Recommended charts" not in body
    assert "Chart section" not in body
    assert "Search charts" not in body
    assert "evidence_views" not in body


def test_main_dashboards_render_related_chart_expanders():
    source = APP_PATH.read_text(encoding="utf-8")
    dashboards = [
        "overview_dashboard",
        "social_dashboard",
        "narrative_dashboard",
        "violence_dashboard",
        "institutions_dashboard",
        "technology_dashboard",
        "cross_analysis_dashboard",
    ]

    for dashboard in dashboards:
        body = function_body(source, dashboard)
        assert "render_related_chart_expanders(" in body, dashboard


def test_related_chart_expanders_render_charts_lazily():
    source = APP_PATH.read_text(encoding="utf-8")
    body = function_body(source, "render_related_chart_expanders")

    assert "key_prefix: str" in body
    assert 'st.toggle("Show chart"' in body
    assert "if show_chart_toggle:" in body
    assert "toggle_related_{key_prefix}_{key_slug}" in body


def test_related_chart_sections_stay_in_dashboard_category():
    source = APP_PATH.read_text(encoding="utf-8")
    expected_sections = {
        "overview_dashboard": ['["Dataset and Temporal Coverage"]'],
        "social_dashboard": [
            '"Gender Representation"',
            '"LGBTQ, Race and Minorities"',
            '"Class, Family and Disability"',
        ],
        "narrative_dashboard": ['"Narrative"', '"Technical Screenplay"'],
        "violence_dashboard": ['"Violence, Morality, Language, Sex and Drugs"', '"Tone, Emotion and Themes"'],
        "institutions_dashboard": ['"Institutions and Power"', '"History, War and Politics"'],
        "technology_dashboard": ['["Technology, Science and Environment"]'],
        "cross_analysis_dashboard": ['["Cross-analysis"]'],
    }
    disallowed_sections = {
        "overview_dashboard": ['"Cross-analysis"'],
        "social_dashboard": ['"History, War and Politics"', '"Tone, Emotion and Themes"', '"Cross-analysis"'],
        "narrative_dashboard": ['"Tone, Emotion and Themes"', '"Cross-analysis"'],
        "violence_dashboard": ['"Cross-analysis"'],
        "institutions_dashboard": ['"Cross-analysis"'],
        "technology_dashboard": ['"Cross-analysis"'],
    }

    for dashboard, snippets in expected_sections.items():
        body = function_body(source, dashboard)
        for snippet in snippets:
            assert snippet in body, f"{dashboard} missing {snippet}"
        for snippet in disallowed_sections.get(dashboard, []):
            assert snippet not in body, f"{dashboard} should not include {snippet}"
