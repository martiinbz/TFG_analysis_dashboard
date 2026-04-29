from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "RESULTADOS_ANALISIS" / "FINAL_DATASET" / "results"

PLOTLY_TEMPLATE = "plotly_white"
MISSING_STRINGS = {"", "unknown", "Unknown", "UNKNOWN", "n/a", "N/A", "null", "Null"}
NONE_STRINGS = {"none", "None", "NONE"}
NON_WHITE_RACE = {"black", "latino", "asian", "middle_eastern"}
BLOCKS = [
    "metadata",
    "narrative",
    "technical",
    "representation",
    "identity_and_social",
    "class_and_family",
    "violence_and_morality",
    "institutions",
    "technology_and_environment",
    "historical_context",
    "thematic",
]

BLOCK_LABELS = {
    "metadata": "Metadata",
    "narrative": "Narrative",
    "technical": "Technical",
    "representation": "Gender Representation",
    "identity_and_social": "LGBTQ, Race and Minorities",
    "class_and_family": "Class, Family and Disability",
    "violence_and_morality": "Violence and Morality",
    "institutions": "Institutions and Power",
    "technology_and_environment": "Technology, Science and Environment",
    "historical_context": "History, War and Politics",
    "thematic": "Tone, Emotion and Themes",
}

FEATURE_BLOCKS = {
    "metadata": ["title", "release_year", "decade", "script_source"],
    "narrative": [
        "primary_genre",
        "story_structure",
        "plot_complexity",
        "opening_type",
        "ending_type",
        "protagonist_age_group",
        "protagonist_gender",
        "protagonist_race_coding",
        "lead_structure",
        "protagonist_type",
        "character_arc_pattern",
        "protagonist_fate",
        "relationship_structure",
    ],
    "technical": [
        "word_count",
        "scene_count",
        "dialogue_density_pct",
        "int_ext_ratio",
        "day_night_ratio",
        "story_time_span",
        "geographic_setting",
        "non_linear_usage",
    ],
    "representation": [
        "female_named_character_count",
        "male_named_character_count",
        "female_dialogue_share_pct",
        "male_dialogue_share_pct",
        "gender_power_dynamics",
        "bechdel_test",
        "sexual_content_presence",
    ],
    "identity_and_social": [
        "lgbtq_presence",
        "lgbtq_character_importance",
        "coming_out_narrative",
        "same_sex_relationships",
        "racial_diversity_presence",
        "minority_portrayal_salience",
        "minority_portrayal_tone",
    ],
    "class_and_family": [
        "class_representation",
        "economic_struggle_presence",
        "wealth_disparity",
        "parental_figure_status",
        "family_dynamics",
        "divorce_presence",
        "physical_disability_representation",
    ],
    "violence_and_morality": [
        "violence_intensity",
        "violence_type",
        "violence_consequences",
        "strong_language_frequency",
        "explicit_scene_count",
        "moral_ambiguity",
        "drug_culture_presence",
    ],
    "institutions": [
        "law_enforcement_portrayal",
        "government_representation",
        "corporate_power_portrayal",
    ],
    "technology_and_environment": [
        "environmental_concerns",
        "ai_presence",
        "human_vs_technology_conflict",
        "scientific_progress_portrayal",
        "digital_revolution_presence",
    ],
    "historical_context": [
        "war_reference",
        "war_portrayal",
        "historical_event_reference",
        "historical_event_salience",
        "political_climate_reference",
        "social_movement_focus",
    ],
    "thematic": [
        "overall_tone",
        "tonal_shift_presence",
        "humor_type",
        "emotional_intensity",
        "ending_tone",
        "primary_universal_theme",
    ],
}

FEATURE_LABELS = {
    "title": "Title",
    "release_year": "Year",
    "decade": "Decade",
    "script_source": "Script Source",
    "primary_genre": "Primary Genre",
    "story_structure": "Story Structure",
    "plot_complexity": "Plot Complexity",
    "opening_type": "Opening Type",
    "ending_type": "Ending Type",
    "protagonist_age_group": "Protagonist Age Group",
    "protagonist_gender": "Protagonist Gender",
    "protagonist_race_coding": "Protagonist Race Coding",
    "lead_structure": "Lead Structure",
    "protagonist_type": "Protagonist Type",
    "character_arc_pattern": "Character Arc Pattern",
    "protagonist_fate": "Protagonist Fate",
    "relationship_structure": "Relationship Structure",
    "word_count": "Word Count",
    "scene_count": "Scene Count",
    "dialogue_density_pct": "Dialogue Density (%)",
    "int_ext_ratio": "Interior / Exterior Ratio",
    "day_night_ratio": "Day / Night Ratio",
    "story_time_span": "Story Time Span",
    "geographic_setting": "Geographic Setting",
    "non_linear_usage": "Technical Non-linear Usage",
    "female_named_character_count": "Named Female Characters",
    "male_named_character_count": "Named Male Characters",
    "female_dialogue_share_pct": "Female Dialogue Share (%)",
    "male_dialogue_share_pct": "Male Dialogue Share (%)",
    "gender_power_dynamics": "Gender Power Dynamics",
    "bechdel_test": "Bechdel Test",
    "sexual_content_presence": "Sexual Content",
    "lgbtq_presence": "LGBTQ+ Presence",
    "lgbtq_character_importance": "LGBTQ+ Character Importance",
    "coming_out_narrative": "Coming-out Narrative",
    "same_sex_relationships": "Same-sex Relationships",
    "racial_diversity_presence": "Explicit Racial Diversity",
    "minority_portrayal_salience": "Minority Portrayal Salience",
    "minority_portrayal_tone": "Minority Portrayal Tone",
    "class_representation": "Class Representation",
    "economic_struggle_presence": "Economic Struggle",
    "wealth_disparity": "Wealth Disparity",
    "parental_figure_status": "Parental Figure Status",
    "family_dynamics": "Family Dynamics",
    "divorce_presence": "Divorce",
    "physical_disability_representation": "Physical Disability",
    "violence_intensity": "Violence Intensity",
    "violence_type": "Violence Type",
    "violence_consequences": "Violence Consequences",
    "strong_language_frequency": "Strong Language",
    "explicit_scene_count": "Explicit Scene Count",
    "moral_ambiguity": "Moral Ambiguity",
    "drug_culture_presence": "Drug Culture",
    "law_enforcement_portrayal": "Law Enforcement Portrayal",
    "government_representation": "Government Representation",
    "corporate_power_portrayal": "Corporate Power Portrayal",
    "environmental_concerns": "Environmental Concerns",
    "ai_presence": "AI Presence",
    "human_vs_technology_conflict": "Human vs Technology Conflict",
    "scientific_progress_portrayal": "Scientific Progress Portrayal",
    "digital_revolution_presence": "Digital Revolution Presence",
    "war_reference": "War Reference",
    "war_portrayal": "War Portrayal",
    "historical_event_reference": "Historical Event Reference",
    "historical_event_salience": "Historical Event Salience",
    "political_climate_reference": "Political Climate Reference",
    "social_movement_focus": "Social Movement Focus",
    "overall_tone": "Overall Tone",
    "tonal_shift_presence": "Tonal Shift",
    "humor_type": "Humor Type",
    "emotional_intensity": "Emotional Intensity",
    "ending_tone": "Ending Tone",
    "primary_universal_theme": "Primary Universal Theme",
    "int_pct": "Interior (%)",
    "ext_pct": "Exterior (%)",
    "day_pct": "Day (%)",
    "night_pct": "Night (%)",
    "female_character_share": "Female Character Share",
    "dialogue_gender_gap": "Dialogue Gender Gap",
    "characters_total": "Named Characters",
    "female_protagonist": "Female Protagonist",
    "high_violence": "High Violence",
    "high_emotion": "High Emotional Intensity",
    "complex_plot": "Complex Plot",
    "dark_or_tense_tone": "Dark or Tense Tone",
    "tragic_or_ambiguous_ending": "Tragic or Ambiguous Ending",
    "institutional_negativity": "Institutional Negativity",
    "war_film": "War Reference",
    "technology_theme": "Technology Theme",
    "historical_or_political": "Historical or Political",
}

GENRE_MAP = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "comedy": "Comedy",
    "crime": "Crime",
    "drama": "Drama",
    "fantasy": "Fantasy",
    "horror": "Horror",
    "musical": "Musical",
    "mystery": "Mystery",
    "romance": "Romance",
    "sci-fi": "Sci-fi",
    "sci fi": "Sci-fi",
    "science fiction": "Sci-fi",
    "thriller": "Thriller",
    "war": "War",
    "western": "Western",
}

COLORWAY = [
    "#0f766e",
    "#b45309",
    "#4f46e5",
    "#be123c",
    "#64748b",
    "#15803d",
    "#9333ea",
    "#ca8a04",
    "#0369a1",
    "#9f1239",
]

CHART_COUNTER = itertools.count()


st.set_page_config(
    page_title="Film Narrative Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f5ef;
            --panel: rgba(255,255,255,.82);
            --ink: #171717;
            --muted: #68645d;
            --line: rgba(23,23,23,.12);
            --accent: #0f766e;
            --warm: #b45309;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(255,255,255,.55), rgba(247,245,239,.92)),
                #f7f5ef;
            color: var(--ink);
        }
        .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1480px; }
        [data-testid="stSidebar"] {
            background: rgba(250,248,243,.98);
            border-right: 1px solid var(--line);
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 12px 28px rgba(15,23,42,.045);
        }
        .hero {
            padding: .6rem 0 1.1rem;
            margin-bottom: .8rem;
            border-bottom: 1px solid var(--line);
        }
        .hero h1 {
            margin: 0 0 .55rem;
            font-size: clamp(2.4rem, 5vw, 5.2rem);
            line-height: .94;
            letter-spacing: 0;
        }
        .hero p {
            max-width: 980px;
            margin: 0;
            color: var(--muted);
            line-height: 1.62;
            font-size: 1.03rem;
        }
        .kicker {
            color: var(--accent);
            font-size: .76rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 800;
            margin: .65rem 0 .2rem;
        }
        .note {
            color: var(--muted);
            border-left: 3px solid var(--accent);
            padding-left: 12px;
            line-height: 1.55;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 14px 30px rgba(15,23,42,.045);
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
            gap: 10px;
        }
        .feature-card {
            background: rgba(255,255,255,.72);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
            min-height: 76px;
        }
        .feature-card .label {
            color: var(--muted);
            font-size: .74rem;
            text-transform: uppercase;
            letter-spacing: .04em;
            font-weight: 800;
        }
        .feature-card .value {
            margin-top: 8px;
            font-size: 1.02rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
        .pill {
            display:inline-flex;
            align-items:center;
            border:1px solid var(--line);
            background: rgba(255,255,255,.68);
            border-radius: 999px;
            padding: 6px 10px;
            color: #334155;
            font-size: .84rem;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 9px 14px; }
        div[data-testid="stExpander"] {
            background: rgba(255,255,255,.58);
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def label(name: str) -> str:
    base = name.split("__")[-1]
    return FEATURE_LABELS.get(base, base.replace("_", " ").title())


def chart_config() -> dict[str, bool]:
    return {"displayModeBar": False, "responsive": True}


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in MISSING_STRINGS:
        return True
    return False


def clean_categorical_value(value: Any, *, keep_none: bool = True) -> Any:
    if not isinstance(value, str):
        return value
    clean = value.strip()
    if clean in MISSING_STRINGS:
        return np.nan
    if clean in NONE_STRINGS:
        return "none" if keep_none else np.nan
    return clean


def parse_ratio(value: Any) -> tuple[float, float]:
    if not isinstance(value, str) or ":" not in value:
        return np.nan, np.nan
    left, right = value.split(":", 1)
    try:
        a = float(left.strip())
        b = float(right.strip())
    except ValueError:
        return np.nan, np.nan
    total = a + b
    if total <= 0:
        return np.nan, np.nan
    return round(a / total * 100, 2), round(b / total * 100, 2)


def decade_key(value: Any) -> float:
    match = re.search(r"(\d{4})", str(value))
    return float(match.group(1)) if match else np.inf


def decade_category_order(values: list[Any]) -> list[str]:
    clean = [str(v) for v in values if pd.notna(v)]
    decades = sorted({v for v in clean if re.search(r"\d{4}s", v)}, key=decade_key)
    return decades


def trace_values(fig: go.Figure, axis: str) -> list[Any]:
    values: list[Any] = []
    for trace in fig.data:
        raw = getattr(trace, axis, None)
        if raw is None:
            continue
        try:
            values.extend(np.ravel(raw).tolist())
        except TypeError:
            values.append(raw)
    return values


def sorted_by_decade(df: pd.DataFrame, col: str = "decade") -> pd.DataFrame:
    if col not in df:
        return df
    out = df.copy()
    out["_order"] = out[col].map(decade_key)
    return out.sort_values("_order").drop(columns="_order")


def stringify_evidence(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(stringify_evidence(v) for v in value if stringify_evidence(v))
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            text = stringify_evidence(item)
            if text:
                parts.append(f"{key}: {text}")
        return "\n\n".join(parts)
    return str(value)


def flatten_json(data: dict[str, Any], source_file: str) -> dict[str, Any]:
    row: dict[str, Any] = {"source_file": source_file}
    for section, values in data.items():
        if section == "evidence_payload":
            if isinstance(values, dict):
                for key, value in values.items():
                    row[f"evidence__{key}"] = stringify_evidence(value)
            continue
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            row[f"{section}__{key}"] = value
            if key not in row:
                row[key] = value
    return row


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        if not col.startswith("evidence__"):
            df[col] = df[col].map(clean_categorical_value)

    if "title" in df:
        df["title"] = df["title"].fillna("Untitled")

    if "primary_genre" in df:
        df["primary_genre"] = df["primary_genre"].map(
            lambda value: GENRE_MAP.get(str(value).strip().lower(), value)
            if pd.notna(value)
            else value
        )

    if "release_year" in df:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    if "decade" in df:
        df["decade"] = df["decade"].where(df["decade"].astype(str).str.match(r"^\d{4}s$"), np.nan)
    if {"release_year", "decade"}.issubset(df.columns):
        derived = (df["release_year"] // 10 * 10).astype("Int64").astype(str) + "s"
        df["decade"] = df["decade"].fillna(derived.where(df["release_year"].notna()))

    bool_cols = [
        "bechdel_test",
        "lgbtq_presence",
        "coming_out_narrative",
        "same_sex_relationships",
        "racial_diversity_presence",
        "economic_struggle_presence",
        "divorce_presence",
        "physical_disability_representation",
        "drug_culture_presence",
        "ai_presence",
        "human_vs_technology_conflict",
        "digital_revolution_presence",
    ]
    for col in bool_cols:
        if col in df:
            df[col] = df[col].astype("boolean")
    return df


@st.cache_data(show_spinner=False)
def load_dataset(data_dir: Path, signature: tuple[int, int, int]) -> pd.DataFrame:
    del signature
    rows = []
    for path in sorted(data_dir.glob("*.json")):
        try:
            rows.append(flatten_json(json.loads(path.read_text(encoding="utf-8")), path.name))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "release_year",
        "plot_complexity",
        "word_count",
        "scene_count",
        "dialogue_density_pct",
        "female_named_character_count",
        "male_named_character_count",
        "female_dialogue_share_pct",
        "male_dialogue_share_pct",
        "violence_intensity",
        "explicit_scene_count",
        "emotional_intensity",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = normalize_dataset(df).copy()

    if "int_ext_ratio" in df:
        ratios = df["int_ext_ratio"].map(parse_ratio)
        df["int_pct"] = ratios.map(lambda x: x[0])
        df["ext_pct"] = ratios.map(lambda x: x[1])
    if "day_night_ratio" in df:
        ratios = df["day_night_ratio"].map(parse_ratio)
        df["day_pct"] = ratios.map(lambda x: x[0])
        df["night_pct"] = ratios.map(lambda x: x[1])
    if {"female_named_character_count", "male_named_character_count"}.issubset(df.columns):
        total = df["female_named_character_count"].fillna(0) + df["male_named_character_count"].fillna(0)
        df["characters_total"] = total
        df["female_character_share"] = np.where(total > 0, df["female_named_character_count"] / total * 100, np.nan)
    if {"female_dialogue_share_pct", "male_dialogue_share_pct"}.issubset(df.columns):
        df["dialogue_gender_gap"] = df["female_dialogue_share_pct"] - df["male_dialogue_share_pct"]
    if "protagonist_race_coding" in df:
        df["explicit_non_white_protagonist"] = df["protagonist_race_coding"].isin(NON_WHITE_RACE)
    if {"family_dynamics", "ending_tone"}.issubset(df.columns):
        df["family_conflict_or_abuse"] = df["family_dynamics"].isin(["conflictual", "abusive"])
    if {"class_representation"}.issubset(df.columns):
        df["working_or_poverty"] = df["class_representation"].isin(["working class", "poverty"])
        df["wealthy_or_upper_middle"] = df["class_representation"].isin(["wealthy", "upper-middle"])
    if {"violence_consequences"}.issubset(df.columns):
        df["violence_minimal_or_rewarded"] = df["violence_consequences"].isin(["minimal", "rewarded"])
    if {"minority_portrayal_salience"}.issubset(df.columns):
        df["minority_salience_medium_high"] = df["minority_portrayal_salience"].isin(["medium", "high"])
    if {"law_enforcement_portrayal"}.issubset(df.columns):
        df["law_negative_or_corrupt"] = df["law_enforcement_portrayal"].isin(["negative", "corrupt"])
    if {"government_representation"}.issubset(df.columns):
        df["government_bad"] = df["government_representation"].isin(["oppressive", "incompetent"])
    if {"overall_tone"}.issubset(df.columns):
        df["dark_or_tense_tone"] = df["overall_tone"].isin(["dark", "tense"])
    if {"ending_tone"}.issubset(df.columns):
        df["tragic_or_disturbing_ending"] = df["ending_tone"].isin(["tragic", "disturbing"])
    if {"primary_universal_theme"}.issubset(df.columns):
        df["identity_theme"] = df["primary_universal_theme"].astype(str).str.lower().eq("identity")
        df["power_theme"] = df["primary_universal_theme"].astype(str).str.lower().eq("power")
        df["freedom_control_theme"] = df["primary_universal_theme"].astype(str).str.lower().isin(
            ["freedom vs control", "freedom/control", "control vs freedom"]
        )

    return df.sort_values(["release_year", "title"], na_position="last").reset_index(drop=True)


def data_signature(data_dir: Path) -> tuple[int, int, int]:
    files = list(data_dir.glob("*.json"))
    return (
        len(files),
        max((path.stat().st_mtime_ns for path in files), default=0),
        sum(path.stat().st_size for path in files),
    )


def available_options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df:
        return []
    values = [str(v) for v in df[column].dropna().unique() if str(v).strip()]
    return sorted(values, key=lambda x: (decade_key(x) if column == "decade" else 0, x))


@dataclass
class Controls:
    calc_mode: str
    min_films_per_decade: int
    group_by: str
    top_n: int
    smoothing: int
    exclude_unknown: bool
    show_evidence: bool
    year_range: tuple[int, int]


def apply_global_controls(df: pd.DataFrame) -> tuple[pd.DataFrame, Controls]:
    st.sidebar.markdown("### Global Controls")
    search = st.sidebar.text_input("Search film", placeholder="Alien, Titanic, Her...")

    year_min = int(df["release_year"].min()) if "release_year" in df and df["release_year"].notna().any() else 1900
    year_max = int(df["release_year"].max()) if "release_year" in df and df["release_year"].notna().any() else 2026
    year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

    decades = st.sidebar.multiselect("Decades", available_options(df, "decade"))
    genres = st.sidebar.multiselect("Primary genre", available_options(df, "primary_genre"))
    exclude_unknown = st.sidebar.checkbox("Exclude unknown values", value=True)
    calc_mode = st.sidebar.radio("Calculation mode", ["count", "% by decade", "% of total", "mean"], index=1)
    min_n = st.sidebar.slider("Minimum films per decade", 1, 50, 1)
    group_options = ["decade", "primary_genre", "protagonist_gender", "protagonist_race_coding", "overall_tone"]
    group_by = st.sidebar.selectbox("Group by", [g for g in group_options if g in df], format_func=label)
    top_n = st.sidebar.slider("Top N categories", 3, 30, 12)
    smoothing = st.sidebar.slider("Temporal smoothing", 1, 3, 1, help="Rolling mean across decades.")
    show_evidence = st.sidebar.checkbox("Show evidence snippets", value=False)

    filtered = df.copy()
    if search and "title" in filtered:
        filtered = filtered[filtered["title"].astype(str).str.contains(search, case=False, na=False)]
    if "release_year" in filtered:
        filtered = filtered[filtered["release_year"].between(year_range[0], year_range[1])]
    if decades:
        filtered = filtered[filtered["decade"].astype(str).isin(decades)]
    if genres:
        filtered = filtered[filtered["primary_genre"].astype(str).isin(genres)]

    st.sidebar.caption(f"{len(filtered):,} of {len(df):,} films visible")
    return filtered, Controls(calc_mode, min_n, group_by, top_n, smoothing, exclude_unknown, show_evidence, year_range)


def valid_df(df: pd.DataFrame, cols: list[str], controls: Controls | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out:
            return out.iloc[0:0]
        if controls and controls.exclude_unknown and not pd.api.types.is_numeric_dtype(out[col]):
            out = out[~out[col].astype(str).str.lower().isin(["unknown", "none", "nan"])]
    return out


def filter_min_decade(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    if "decade" not in df:
        return df
    counts = df["decade"].value_counts()
    return df[df["decade"].isin(counts[counts >= min_n].index)]


def apply_smoothing(data: pd.DataFrame, x: str, y: str, color: str | None, window: int) -> pd.DataFrame:
    if window <= 1 or x != "decade" or y not in data:
        return data
    out = sorted_by_decade(data, x)
    if color and color in out:
        out[y] = out.groupby(color, group_keys=False)[y].transform(lambda s: s.rolling(window, min_periods=1).mean())
    else:
        out[y] = out[y].rolling(window, min_periods=1).mean()
    return out


def finish_fig(fig: go.Figure, height: int = 430, percent_y: bool = False) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        colorway=COLORWAY,
        height=height,
        margin=dict(l=10, r=10, t=58, b=35),
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.42)",
    )
    if percent_y:
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
    x_decades = decade_category_order(trace_values(fig, "x"))
    if x_decades:
        fig.update_xaxes(categoryorder="array", categoryarray=x_decades)
    y_decades = decade_category_order(trace_values(fig, "y"))
    if y_decades:
        fig.update_yaxes(categoryorder="array", categoryarray=y_decades)
    return fig


def line_count(df: pd.DataFrame, x: str, title: str, controls: Controls, color: str | None = None) -> go.Figure:
    plot_df = valid_df(df, [x] + ([color] if color else []), controls)
    grouped = plot_df.groupby([x] + ([color] if color else []), dropna=False).size().reset_index(name="Films")
    if x == "decade":
        grouped = sorted_by_decade(grouped, x)
    fig = px.line(grouped, x=x, y="Films", color=color, markers=True, title=title, labels={x: label(x)})
    return finish_fig(fig)


def bar_count(df: pd.DataFrame, x: str, title: str, controls: Controls, color: str | None = None) -> go.Figure:
    plot_df = valid_df(df, [x] + ([color] if color else []), controls)
    grouped = plot_df.groupby([x] + ([color] if color else []), dropna=False).size().reset_index(name="Films")
    if x == "decade":
        grouped = sorted_by_decade(grouped, x)
    fig = px.bar(grouped, x=x, y="Films", color=color, title=title, labels={x: label(x)})
    return finish_fig(fig)


def stacked_percent(df: pd.DataFrame, x: str, category: str, title: str, controls: Controls, top_n: int | None = None) -> go.Figure:
    plot_df = valid_df(df, [x, category], controls)
    if x == "decade":
        plot_df = filter_min_decade(plot_df, controls.min_films_per_decade)
    if top_n:
        top = plot_df[category].value_counts().head(top_n).index
        plot_df = plot_df[plot_df[category].isin(top)]
    grouped = plot_df.groupby([x, category], dropna=False).size().reset_index(name="Films")
    totals = grouped.groupby(x)["Films"].transform("sum")
    grouped["Percent"] = np.where(totals > 0, grouped["Films"] / totals * 100, 0)
    if x == "decade":
        grouped = sorted_by_decade(grouped, x)
    fig = px.bar(
        grouped,
        x=x,
        y="Percent",
        color=category,
        title=title,
        labels={x: label(x), category: label(category), "Percent": "%"},
    )
    return finish_fig(fig, percent_y=True)


def heatmap_percent(df: pd.DataFrame, x: str, y: str, title: str, controls: Controls, top_n: int | None = None) -> go.Figure:
    plot_df = valid_df(df, [x, y], controls)
    if top_n:
        top = plot_df[y].value_counts().head(top_n).index
        plot_df = plot_df[plot_df[y].isin(top)]
    grouped = plot_df.groupby([x, y], dropna=False).size().reset_index(name="Films")
    totals = grouped.groupby(x)["Films"].transform("sum")
    grouped["Percent"] = np.where(totals > 0, grouped["Films"] / totals * 100, 0)
    fig = px.density_heatmap(
        grouped,
        x=x,
        y=y,
        z="Percent",
        histfunc="sum",
        title=title,
        labels={x: label(x), y: label(y), "Percent": "%"},
        color_continuous_scale=["#f4eee5", "#0f766e"],
    )
    fig.update_coloraxes(colorbar_ticksuffix="%")
    return finish_fig(fig, height=500)


def mean_line(df: pd.DataFrame, metric: str, title: str, controls: Controls, color: str | None = None) -> go.Figure:
    cols = ["decade", metric] + ([color] if color else [])
    plot_df = valid_df(df, cols, controls).dropna(subset=[metric, "decade"])
    plot_df = filter_min_decade(plot_df, controls.min_films_per_decade)
    group_cols = ["decade"] + ([color] if color else [])
    grouped = plot_df.groupby(group_cols, dropna=False)[metric].mean().reset_index(name="Mean")
    grouped = apply_smoothing(sorted_by_decade(grouped), "decade", "Mean", color, controls.smoothing)
    fig = px.line(grouped, x="decade", y="Mean", color=color, markers=True, title=title, labels={"Mean": label(metric)})
    if metric in {"plot_complexity", "violence_intensity", "emotional_intensity"}:
        fig.update_yaxes(range=[0, 5])
    return finish_fig(fig)


def rate_line(
    df: pd.DataFrame,
    feature: str,
    target: Any,
    title: str,
    controls: Controls,
    color: str | None = None,
    targets: set[Any] | None = None,
) -> go.Figure:
    cols = ["decade", feature] + ([color] if color else [])
    plot_df = valid_df(df, cols, controls).dropna(subset=["decade"])
    plot_df = filter_min_decade(plot_df, controls.min_films_per_decade)
    if targets is None:
        plot_df["_hit"] = plot_df[feature] == target
    else:
        plot_df["_hit"] = plot_df[feature].isin(targets)
    group_cols = ["decade"] + ([color] if color else [])
    grouped = plot_df.groupby(group_cols, dropna=False)["_hit"].mean().reset_index(name="Percent")
    grouped["Percent"] *= 100
    grouped = apply_smoothing(sorted_by_decade(grouped), "decade", "Percent", color, controls.smoothing)
    fig = px.line(grouped, x="decade", y="Percent", color=color, markers=True, title=title, labels={"Percent": "%"})
    return finish_fig(fig, percent_y=True)


def multi_rate_line(df: pd.DataFrame, feature: str, targets: list[Any], title: str, controls: Controls) -> go.Figure:
    parts = []
    for target in targets:
        plot_df = valid_df(df, ["decade", feature], controls).dropna(subset=["decade"])
        plot_df = filter_min_decade(plot_df, controls.min_films_per_decade)
        plot_df["_hit"] = plot_df[feature] == target
        tmp = plot_df.groupby("decade", dropna=False)["_hit"].mean().reset_index(name="Percent")
        tmp["Percent"] *= 100
        tmp["Metric"] = str(target)
        parts.append(tmp)
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    long = apply_smoothing(sorted_by_decade(long), "decade", "Percent", "Metric", controls.smoothing)
    fig = px.line(long, x="decade", y="Percent", color="Metric", markers=True, title=title)
    return finish_fig(fig, percent_y=True)


def boolean_rate_by_category(df: pd.DataFrame, category: str, feature: str, title: str, controls: Controls) -> go.Figure:
    plot_df = valid_df(df, [category, feature], controls).dropna(subset=[category])
    grouped = plot_df.groupby(category, dropna=False).agg(Percent=(feature, "mean"), Films=("title", "count")).reset_index()
    grouped = grouped[grouped["Films"] >= 1]
    if len(grouped) > controls.top_n:
        grouped = grouped.sort_values("Films", ascending=False).head(controls.top_n)
    grouped["Percent"] *= 100
    fig = px.bar(grouped.sort_values("Percent"), x="Percent", y=category, orientation="h", title=title, labels={"Percent": "%"})
    return finish_fig(fig, percent_y=False)


def boxplot(df: pd.DataFrame, x: str, y: str, title: str, controls: Controls, color: str | None = None) -> go.Figure:
    plot_df = valid_df(df, [x, y] + ([color] if color else []), controls).dropna(subset=[x, y])
    if x == "primary_genre":
        top = plot_df[x].value_counts().head(controls.top_n).index
        plot_df = plot_df[plot_df[x].isin(top)]
    fig = px.box(plot_df, x=x, y=y, color=color or x, points="outliers", title=title, labels={x: label(x), y: label(y)})
    fig.update_layout(showlegend=bool(color))
    if y in {"plot_complexity", "violence_intensity", "emotional_intensity"}:
        fig.update_yaxes(range=[0, 5])
    return finish_fig(fig, height=465)


def violin(df: pd.DataFrame, x: str, y: str, title: str, controls: Controls) -> go.Figure:
    plot_df = valid_df(df, [x, y], controls).dropna(subset=[x, y])
    if x == "primary_genre":
        top = plot_df[x].value_counts().head(controls.top_n).index
        plot_df = plot_df[plot_df[x].isin(top)]
    fig = px.violin(plot_df, x=x, y=y, color=x, box=True, points=False, title=title, labels={x: label(x), y: label(y)})
    fig.update_layout(showlegend=False)
    return finish_fig(fig, height=465)


def scatter(df: pd.DataFrame, x: str, y: str, title: str, controls: Controls, color: str = "primary_genre", size: str | None = None) -> go.Figure:
    cols = [x, y, color] + ([size] if size else [])
    plot_df = valid_df(df, cols, controls).dropna(subset=[x, y])
    fig = px.scatter(plot_df, x=x, y=y, color=color, size=size, hover_name="title", title=title, labels={x: label(x), y: label(y), color: label(color)})
    return finish_fig(fig, height=500)


def grouped_bar_rate(df: pd.DataFrame, x: str, feature: str, target: Any, title: str, controls: Controls) -> go.Figure:
    plot_df = valid_df(df, [x, feature], controls).dropna(subset=[x])
    plot_df["_hit"] = plot_df[feature] == target
    grouped = plot_df.groupby(x, dropna=False)["_hit"].mean().reset_index(name="Percent")
    grouped["Percent"] *= 100
    fig = px.bar(grouped, x=x, y="Percent", title=title, labels={x: label(x), "Percent": "%"})
    return finish_fig(fig, percent_y=True)


def crosstab_heatmap(df: pd.DataFrame, x: str, y: str, title: str, controls: Controls) -> go.Figure:
    return heatmap_percent(df, x, y, title, controls, top_n=controls.top_n)


def multi_metric_line(df: pd.DataFrame, metrics: list[str], title: str, controls: Controls, y_range: tuple[int, int] | None = None) -> go.Figure:
    available = [m for m in metrics if m in df]
    long = df.melt(id_vars=["decade"], value_vars=available, var_name="Feature", value_name="Value").dropna()
    long = filter_min_decade(long, controls.min_films_per_decade)
    grouped = long.groupby(["decade", "Feature"], dropna=False)["Value"].mean().reset_index()
    grouped["Feature"] = grouped["Feature"].map(label)
    grouped = apply_smoothing(sorted_by_decade(grouped), "decade", "Value", "Feature", controls.smoothing)
    fig = px.line(grouped, x="decade", y="Value", color="Feature", markers=True, title=title)
    if y_range:
        fig.update_yaxes(range=list(y_range))
    return finish_fig(fig)


def numeric_correlation_heatmap(df: pd.DataFrame, controls: Controls) -> go.Figure:
    metrics = [
        "word_count",
        "scene_count",
        "dialogue_density_pct",
        "plot_complexity",
        "violence_intensity",
        "emotional_intensity",
        "explicit_scene_count",
        "female_named_character_count",
        "male_named_character_count",
        "female_dialogue_share_pct",
        "male_dialogue_share_pct",
        "female_character_share",
        "dialogue_gender_gap",
    ]
    available = [
        col
        for col in metrics
        if col in df and pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
    ]
    plot_df = df[available].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if plot_df.shape[1] < 2:
        raise ValueError("Not enough numeric features available.")

    corr = plot_df.corr()
    labels = [label(col) for col in corr.columns]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="r"),
            text=np.round(corr.values, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} x %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title="Numeric Feature Correlations")
    fig.update_xaxes(tickangle=35)
    return finish_fig(fig, height=660)


def binary_feature_cooccurrence_heatmap(df: pd.DataFrame, controls: Controls) -> go.Figure:
    flags = {
        "Bechdel Test": df.get("bechdel_test"),
        "Female Protagonist": df.get("protagonist_gender").eq("female") if "protagonist_gender" in df else None,
        "LGBTQ+ Presence": df.get("lgbtq_presence"),
        "Racial Diversity": df.get("racial_diversity_presence"),
        "Economic Struggle": df.get("economic_struggle_presence"),
        "Drug Culture": df.get("drug_culture_presence"),
        "Sexual Content": df.get("sexual_content_presence"),
        "High Violence": df.get("violence_intensity").ge(4) if "violence_intensity" in df else None,
        "High Emotion": df.get("emotional_intensity").ge(4) if "emotional_intensity" in df else None,
        "Complex Plot": df.get("plot_complexity").ge(4) if "plot_complexity" in df else None,
        "Dark / Tense Tone": df.get("overall_tone").isin(["dark", "tense"]) if "overall_tone" in df else None,
        "Tragic / Ambiguous Ending": df.get("ending_tone").isin(["tragic", "ambiguous", "disturbing"]) if "ending_tone" in df else None,
        "Institutional Negativity": df.get("institutional_negativity"),
        "War Reference": df.get("war_reference").notna() & ~df.get("war_reference").astype(str).str.lower().isin(["none", "unknown", "nan"]) if "war_reference" in df else None,
        "Technology Theme": (
            df.get("ai_presence").fillna(False).astype(bool)
            | df.get("digital_revolution_presence").fillna(False).astype(bool)
            | df.get("human_vs_technology_conflict").fillna(False).astype(bool)
            if {"ai_presence", "digital_revolution_presence", "human_vs_technology_conflict"}.issubset(df.columns)
            else None
        ),
        "Historical / Political": (
            df.get("historical_event_salience").astype(str).str.lower().isin(["background", "central"])
            | ~df.get("political_climate_reference").astype(str).str.lower().isin(["none", "unknown", "nan"])
            if {"historical_event_salience", "political_climate_reference"}.issubset(df.columns)
            else None
        ),
    }
    matrix = pd.DataFrame({name: series for name, series in flags.items() if series is not None}, index=df.index)
    if matrix.shape[1] < 2:
        raise ValueError("Not enough binary features available.")
    matrix = matrix.fillna(False).astype(bool)

    cooccurrence = pd.DataFrame(index=matrix.columns, columns=matrix.columns, dtype=float)
    for row_feature in matrix.columns:
        for col_feature in matrix.columns:
            cooccurrence.loc[row_feature, col_feature] = (matrix[row_feature] & matrix[col_feature]).mean() * 100

    fig = go.Figure(
        data=go.Heatmap(
            z=cooccurrence.values,
            x=cooccurrence.columns,
            y=cooccurrence.index,
            zmin=0,
            zmax=max(1, float(np.nanmax(cooccurrence.values))),
            colorscale=["#f4eee5", "#0f766e"],
            colorbar=dict(title="% films"),
            text=np.round(cooccurrence.values, 1),
            texttemplate="%{text:.1f}%",
            hovertemplate="%{y} + %{x}<br>Co-occurrence: %{z:.1f}% of films<extra></extra>",
        )
    )
    fig.update_layout(title="Binary Feature Co-occurrence")
    fig.update_xaxes(tickangle=35)
    return finish_fig(fig, height=720)


def unknown_rate_chart(df: pd.DataFrame, controls: Controls, block: str | None = None) -> go.Figure:
    features = []
    blocks = [block] if block and block != "all" else list(FEATURE_BLOCKS)
    for section in blocks:
        for col in FEATURE_BLOCKS.get(section, []):
            if col in df:
                missing = df[col].isna() | df[col].astype(str).str.lower().isin(["unknown", "nan", ""])
                features.append({"Feature": label(col), "Unknown Rate": missing.mean() * 100, "Block": BLOCK_LABELS.get(section, section)})
    rates = pd.DataFrame(features).sort_values("Unknown Rate", ascending=True)
    fig = px.bar(rates, x="Unknown Rate", y="Feature", color="Block", orientation="h", title="Unknown / Null Rate by Feature")
    return finish_fig(fig, height=max(420, min(960, 28 * len(rates))))


def bechdel_by_decade(df: pd.DataFrame, controls: Controls) -> go.Figure:
    plot_df = valid_df(df, ["decade", "bechdel_test"], controls).dropna(subset=["decade"])
    plot_df = filter_min_decade(plot_df, controls.min_films_per_decade)
    grouped = plot_df.groupby("decade", dropna=False).agg(Rate=("bechdel_test", "mean"), Films=("title", "count")).reset_index()
    grouped["Percent"] = grouped["Rate"] * 100
    grouped["Decade"] = grouped["decade"].astype(str) + "<br>n=" + grouped["Films"].astype(str)
    grouped = sorted_by_decade(grouped)
    fig = px.line(grouped, x="Decade", y="Percent", markers=True, title="Bechdel Pass Rate by Decade", labels={"Percent": "%"})
    return finish_fig(fig, percent_y=True)


def filtered_table(df: pd.DataFrame, controls: Controls) -> None:
    cols = [
        "title",
        "release_year",
        "decade",
        "primary_genre",
        "protagonist_gender",
        "protagonist_race_coding",
        "bechdel_test",
        "female_dialogue_share_pct",
        "lgbtq_presence",
        "racial_diversity_presence",
        "violence_intensity",
        "overall_tone",
        "ending_tone",
    ]
    st.dataframe(df[[c for c in cols if c in df]], width="stretch", hide_index=True)


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    chart_title = ""
    if fig.layout and fig.layout.title and fig.layout.title.text:
        chart_title = re.sub(r"[^a-zA-Z0-9_]+", "_", str(fig.layout.title.text)).strip("_").lower()
    unique_key = key or f"chart_{next(CHART_COUNTER)}_{chart_title}"
    st.plotly_chart(fig, width="stretch", config=chart_config(), key=unique_key)


@dataclass
class ChartSpec:
    section: str
    title: str
    description: str
    builder: Callable[[pd.DataFrame, Controls], go.Figure]


def build_specs() -> list[ChartSpec]:
    S = ChartSpec
    return [
        S("Dataset and Temporal Coverage", "Films by Decade", "Bar chart of film counts by decade.", lambda d, c: bar_count(d, "decade", "Films by Decade", c)),
        S("Dataset and Temporal Coverage", "Films by Year", "Line chart of film counts by release year.", lambda d, c: line_count(d, "release_year", "Films by Year", c)),
        S("Dataset and Temporal Coverage", "Coverage by Decade and Genre", "Heatmap of decade by primary genre.", lambda d, c: heatmap_percent(d, "decade", "primary_genre", "Coverage by Decade and Genre", c, c.top_n)),
        S("Dataset and Temporal Coverage", "Unknown Rate by Decade", "Unknown percentage for a selected feature.", lambda d, c: selected_unknown_line(d, c)),
        S("Dataset and Temporal Coverage", "Extraction Quality", "Horizontal bars with unknown/null rates by feature.", lambda d, c: unknown_rate_chart(d, c)),
        S("Dataset and Temporal Coverage", "Films by Source", "Film counts by script source.", lambda d, c: bar_count(d, "script_source", "Films by Source", c)),
        S("Narrative", "Genre Evolution", "Stacked percentage of primary genres by decade.", lambda d, c: stacked_percent(d, "decade", "primary_genre", "Genre Evolution", c, c.top_n)),
        S("Narrative", "Dominant Genre by Decade", "Genre x decade heatmap.", lambda d, c: heatmap_percent(d, "decade", "primary_genre", "Dominant Genre by Decade", c, c.top_n)),
        S("Narrative", "Average Plot Complexity", "Mean plot complexity by decade.", lambda d, c: mean_line(d, "plot_complexity", "Average Plot Complexity", c)),
        S("Narrative", "Plot Complexity Distribution", "Boxplot of plot complexity by decade.", lambda d, c: boxplot(d, "decade", "plot_complexity", "Plot Complexity Distribution", c)),
        S("Narrative", "Plot Complexity by Genre", "Boxplot of plot complexity by genre.", lambda d, c: boxplot(d, "primary_genre", "plot_complexity", "Plot Complexity by Genre", c)),
        S("Narrative", "Narrative Structure by Decade", "Stacked structure percentages by decade.", lambda d, c: stacked_percent(d, "decade", "story_structure", "Narrative Structure by Decade", c)),
        S("Narrative", "Non-linear Narrative", "Percentage of story_structure = non-linear.", lambda d, c: rate_line(d, "story_structure", "non-linear", "Non-linear Narrative", c)),
        S("Narrative", "Opening Types", "Opening type heatmap by decade.", lambda d, c: heatmap_percent(d, "decade", "opening_type", "Opening Types", c)),
        S("Narrative", "Ending Types", "Stacked ending types by decade.", lambda d, c: stacked_percent(d, "decade", "ending_type", "Ending Types", c)),
        S("Narrative", "Ambiguous Endings", "Percentage of ending_type = ambiguity.", lambda d, c: rate_line(d, "ending_type", "ambiguity", "Ambiguous Endings", c)),
        S("Narrative", "Cliffhangers by Genre", "Percentage of cliffhanger endings by genre.", lambda d, c: grouped_bar_rate(d, "primary_genre", "ending_type", "cliffhanger", "Cliffhangers by Genre", c)),
        S("Narrative", "Protagonist Age", "Stacked protagonist age groups by decade.", lambda d, c: stacked_percent(d, "decade", "protagonist_age_group", "Protagonist Age", c)),
        S("Narrative", "Protagonist Gender", "Protagonist gender by decade.", lambda d, c: stacked_percent(d, "decade", "protagonist_gender", "Protagonist Gender", c)),
        S("Narrative", "Female Protagonists by Genre", "Heatmap of female protagonist share.", lambda d, c: female_protagonist_heatmap(d, c)),
        S("Narrative", "Protagonist Race Coding", "Stacked protagonist race coding by decade.", lambda d, c: stacked_percent(d, "decade", "protagonist_race_coding", "Protagonist Race Coding", c)),
        S("Narrative", "Protagonist Type", "Stacked protagonist type by decade.", lambda d, c: stacked_percent(d, "decade", "protagonist_type", "Protagonist Type", c)),
        S("Narrative", "Antihero by Decade", "Percentage of antiheroes by decade.", lambda d, c: rate_line(d, "protagonist_type", "antihero", "Antihero by Decade", c)),
        S("Narrative", "Character Arc Pattern", "Character arc heatmap by decade.", lambda d, c: heatmap_percent(d, "decade", "character_arc_pattern", "Character Arc Pattern", c)),
        S("Narrative", "Redemption and Corruption", "Line chart for redemption/corruption arcs.", lambda d, c: multi_rate_line(d, "character_arc_pattern", ["redemption", "corruption"], "Redemption and Corruption", c)),
        S("Narrative", "Protagonist Fate", "Stacked protagonist fate by decade.", lambda d, c: stacked_percent(d, "decade", "protagonist_fate", "Protagonist Fate", c)),
        S("Narrative", "Protagonist Deaths", "Percentage of protagonist_fate = dies.", lambda d, c: rate_line(d, "protagonist_fate", "dies", "Protagonist Deaths", c)),
        S("Narrative", "Lead Structure", "Stacked lead structure by decade.", lambda d, c: stacked_percent(d, "decade", "lead_structure", "Lead Structure", c)),
        S("Narrative", "Relationship Structure", "Relationship structure heatmap by decade.", lambda d, c: heatmap_percent(d, "decade", "relationship_structure", "Relationship Structure", c)),
        S("Technical Screenplay", "Average Script Length", "Mean word count by decade.", lambda d, c: mean_line(d, "word_count", "Average Script Length", c)),
        S("Technical Screenplay", "Script Length Distribution", "Word count distribution by decade.", lambda d, c: boxplot(d, "decade", "word_count", "Script Length Distribution", c)),
        S("Technical Screenplay", "Average Scenes by Decade", "Mean scene count by decade.", lambda d, c: mean_line(d, "scene_count", "Average Scenes by Decade", c)),
        S("Technical Screenplay", "Scenes vs Words", "Scatter of word count and scene count.", lambda d, c: scatter(d, "word_count", "scene_count", "Scenes vs Words", c, "primary_genre", "dialogue_density_pct")),
        S("Technical Screenplay", "Dialogue Density by Decade", "Mean dialogue density by decade.", lambda d, c: mean_line(d, "dialogue_density_pct", "Dialogue Density by Decade", c)),
        S("Technical Screenplay", "Dialogue Density by Genre", "Dialogue density boxplot by genre.", lambda d, c: boxplot(d, "primary_genre", "dialogue_density_pct", "Dialogue Density by Genre", c)),
        S("Technical Screenplay", "Most Dialogue-heavy Films", "Ranking of films by dialogue density.", lambda d, c: ranking_bar(d, "dialogue_density_pct", "Most Dialogue-heavy Films", c)),
        S("Technical Screenplay", "Interior vs Exterior", "Mean interior/exterior percentage by decade.", lambda d, c: multi_metric_line(d, ["int_pct", "ext_pct"], "Interior vs Exterior", c, (0, 100))),
        S("Technical Screenplay", "Interiority by Genre", "Interior percentage by genre.", lambda d, c: boxplot(d, "primary_genre", "int_pct", "Interiority by Genre", c)),
        S("Technical Screenplay", "Day vs Night", "Mean day/night percentage by decade.", lambda d, c: multi_metric_line(d, ["day_pct", "night_pct"], "Day vs Night", c, (0, 100))),
        S("Technical Screenplay", "Nocturnality by Genre", "Night percentage by genre.", lambda d, c: boxplot(d, "primary_genre", "night_pct", "Nocturnality by Genre", c)),
        S("Technical Screenplay", "Story Time Span", "Stacked story time span by decade.", lambda d, c: stacked_percent(d, "decade", "story_time_span", "Story Time Span", c)),
        S("Technical Screenplay", "Stories Across Years or Decades", "Percentage of years/decades spans.", lambda d, c: rate_line(d, "story_time_span", "years", "Stories Across Years or Decades", c, targets={"years", "decades"})),
        S("Technical Screenplay", "Geographic Setting", "Stacked geographic setting by decade.", lambda d, c: stacked_percent(d, "decade", "geographic_setting", "Geographic Setting", c)),
        S("Technical Screenplay", "Urban Cinema by Decade", "Percentage of urban settings.", lambda d, c: rate_line(d, "geographic_setting", "urban", "Urban Cinema by Decade", c)),
        S("Technical Screenplay", "Technical Non-linear Usage", "Stacked non-linear usage by decade.", lambda d, c: stacked_percent(d, "decade", "non_linear_usage", "Technical Non-linear Usage", c)),
        S("Gender Representation", "Named Female Characters", "Mean named female character count by decade.", lambda d, c: mean_line(d, "female_named_character_count", "Named Female Characters", c)),
        S("Gender Representation", "Named Male Characters", "Mean named male character count by decade.", lambda d, c: mean_line(d, "male_named_character_count", "Named Male Characters", c)),
        S("Gender Representation", "Female Character Share", "Mean female character share by decade.", lambda d, c: mean_line(d, "female_character_share", "Female Character Share", c)),
        S("Gender Representation", "Named Women vs Men", "Scatter of named women and men.", lambda d, c: scatter(d, "male_named_character_count", "female_named_character_count", "Named Women vs Men", c, "primary_genre")),
        S("Gender Representation", "Female Dialogue by Decade", "Mean female dialogue share by decade.", lambda d, c: mean_line(d, "female_dialogue_share_pct", "Female Dialogue by Decade", c)),
        S("Gender Representation", "Male Dialogue by Decade", "Mean male dialogue share by decade.", lambda d, c: mean_line(d, "male_dialogue_share_pct", "Male Dialogue by Decade", c)),
        S("Gender Representation", "Dialogue Gender Gap", "Female minus male dialogue share.", lambda d, c: dialogue_gap_line(d, c)),
        S("Gender Representation", "Female Dialogue Distribution", "Boxplot of female dialogue share by decade.", lambda d, c: boxplot(d, "decade", "female_dialogue_share_pct", "Female Dialogue Distribution", c)),
        S("Gender Representation", "Female Dialogue by Genre", "Boxplot by genre.", lambda d, c: boxplot(d, "primary_genre", "female_dialogue_share_pct", "Female Dialogue by Genre", c)),
        S("Gender Representation", "Gender Power Dynamics", "Stacked gender power dynamics by decade.", lambda d, c: stacked_percent(d, "decade", "gender_power_dynamics", "Gender Power Dynamics", c)),
        S("Gender Representation", "Male Dominated Cinema", "Percentage male_dominated.", lambda d, c: rate_line(d, "gender_power_dynamics", "male_dominated", "Male Dominated Cinema", c)),
        S("Gender Representation", "Balanced Cinema", "Percentage balanced.", lambda d, c: rate_line(d, "gender_power_dynamics", "balanced", "Balanced Cinema", c)),
        S("Gender Representation", "Bechdel by Decade", "Bechdel pass rate with n labels.", lambda d, c: bechdel_by_decade(d, c)),
        S("Gender Representation", "Bechdel by Genre", "Bechdel pass rate by genre.", lambda d, c: boolean_rate_by_category(d, "primary_genre", "bechdel_test", "Bechdel by Genre", c)),
        S("Gender Representation", "Bechdel vs Female Dialogue", "Female dialogue distribution by Bechdel result.", lambda d, c: boxplot(d, "bechdel_test", "female_dialogue_share_pct", "Bechdel vs Female Dialogue", c)),
        S("Gender Representation", "Bechdel vs Female Protagonist", "Bechdel rate by protagonist gender.", lambda d, c: boolean_rate_by_category(d, "protagonist_gender", "bechdel_test", "Bechdel vs Protagonist Gender", c)),
        S("Gender Representation", "Sexual Content by Decade", "Stacked sexual content by decade.", lambda d, c: stacked_percent(d, "decade", "sexual_content_presence", "Sexual Content by Decade", c)),
        S("Gender Representation", "Sexual Content by Genre", "Heatmap of sexual content by genre.", lambda d, c: heatmap_percent(d, "primary_genre", "sexual_content_presence", "Sexual Content by Genre", c)),
        S("LGBTQ, Race and Minorities", "LGBTQ+ Presence by Decade", "Percentage LGBTQ+ presence by decade.", lambda d, c: rate_line(d, "lgbtq_presence", True, "LGBTQ+ Presence by Decade", c)),
        S("LGBTQ, Race and Minorities", "LGBTQ+ Presence by Genre", "LGBTQ+ presence by genre.", lambda d, c: boolean_rate_by_category(d, "primary_genre", "lgbtq_presence", "LGBTQ+ Presence by Genre", c)),
        S("LGBTQ, Race and Minorities", "LGBTQ+ Character Importance", "Stacked LGBTQ+ importance by decade.", lambda d, c: stacked_percent(d, "decade", "lgbtq_character_importance", "LGBTQ+ Character Importance", c)),
        S("LGBTQ, Race and Minorities", "LGBTQ+ Main / Supporting", "Main or supporting LGBTQ+ characters by decade.", lambda d, c: rate_line(d, "lgbtq_character_importance", "main", "LGBTQ+ Main / Supporting", c, targets={"main", "supporting"})),
        S("LGBTQ, Race and Minorities", "Coming-out Narratives", "Coming-out narrative rate.", lambda d, c: rate_line(d, "coming_out_narrative", True, "Coming-out Narratives", c)),
        S("LGBTQ, Race and Minorities", "Same-sex Relationships", "Same-sex relationship rate.", lambda d, c: rate_line(d, "same_sex_relationships", True, "Same-sex Relationships", c)),
        S("LGBTQ, Race and Minorities", "LGBTQ+ vs Sexual Content", "Heatmap of LGBTQ+ presence and sexual content.", lambda d, c: crosstab_heatmap(d, "lgbtq_presence", "sexual_content_presence", "LGBTQ+ vs Sexual Content", c)),
        S("LGBTQ, Race and Minorities", "Explicit Racial Diversity", "Racial diversity presence by decade.", lambda d, c: rate_line(d, "racial_diversity_presence", True, "Explicit Racial Diversity", c)),
        S("LGBTQ, Race and Minorities", "Racial Diversity by Genre", "Racial diversity by genre.", lambda d, c: boolean_rate_by_category(d, "primary_genre", "racial_diversity_presence", "Racial Diversity by Genre", c)),
        S("LGBTQ, Race and Minorities", "Minority Salience", "Stacked minority salience by decade.", lambda d, c: stacked_percent(d, "decade", "minority_portrayal_salience", "Minority Salience", c)),
        S("LGBTQ, Race and Minorities", "Minority Tone", "Stacked minority tone by decade.", lambda d, c: stacked_percent(d, "decade", "minority_portrayal_tone", "Minority Tone", c)),
        S("LGBTQ, Race and Minorities", "Humanized vs Stereotyped", "Humanized/stereotyped minority tone.", lambda d, c: multi_rate_line(d, "minority_portrayal_tone", ["humanized", "stereotyped"], "Humanized vs Stereotyped", c)),
        S("LGBTQ, Race and Minorities", "Protagonist Race", "Stacked protagonist race coding.", lambda d, c: stacked_percent(d, "decade", "protagonist_race_coding", "Protagonist Race", c)),
        S("LGBTQ, Race and Minorities", "Explicit Non-white Protagonist", "Non-white coded protagonist rate.", lambda d, c: rate_line(d, "explicit_non_white_protagonist", True, "Explicit Non-white Protagonist", c)),
        S("LGBTQ, Race and Minorities", "Race vs Protagonist Gender", "Race and protagonist gender heatmap.", lambda d, c: crosstab_heatmap(d, "protagonist_gender", "protagonist_race_coding", "Race vs Protagonist Gender", c)),
        S("LGBTQ, Race and Minorities", "Race vs Bechdel", "Bechdel rate by protagonist race.", lambda d, c: boolean_rate_by_category(d, "protagonist_race_coding", "bechdel_test", "Race vs Bechdel", c)),
        S("LGBTQ, Race and Minorities", "Racial Diversity vs Law Enforcement", "Diversity and law portrayal heatmap.", lambda d, c: crosstab_heatmap(d, "racial_diversity_presence", "law_enforcement_portrayal", "Racial Diversity vs Law Enforcement", c)),
        S("Class, Family and Disability", "Class Representation", "Stacked class representation by decade.", lambda d, c: stacked_percent(d, "decade", "class_representation", "Class Representation", c)),
        S("Class, Family and Disability", "Working Class / Poverty", "Working class or poverty by decade.", lambda d, c: rate_line(d, "working_or_poverty", True, "Working Class / Poverty", c)),
        S("Class, Family and Disability", "Wealthy / Upper-middle", "Wealthy or upper-middle by decade.", lambda d, c: rate_line(d, "wealthy_or_upper_middle", True, "Wealthy / Upper-middle", c)),
        S("Class, Family and Disability", "Economic Struggle", "Economic struggle by decade.", lambda d, c: rate_line(d, "economic_struggle_presence", True, "Economic Struggle", c)),
        S("Class, Family and Disability", "Economic Struggle by Genre", "Economic struggle by genre.", lambda d, c: boolean_rate_by_category(d, "primary_genre", "economic_struggle_presence", "Economic Struggle by Genre", c)),
        S("Class, Family and Disability", "Wealth Disparity", "Stacked wealth disparity by decade.", lambda d, c: stacked_percent(d, "decade", "wealth_disparity", "Wealth Disparity", c)),
        S("Class, Family and Disability", "Central Wealth Disparity", "Central wealth disparity by decade.", lambda d, c: rate_line(d, "wealth_disparity", "central", "Central Wealth Disparity", c)),
        S("Class, Family and Disability", "Parental Figures", "Parental figure status by decade.", lambda d, c: stacked_percent(d, "decade", "parental_figure_status", "Parental Figures", c)),
        S("Class, Family and Disability", "Conflictual / Abusive Family", "Conflictual or abusive family rate.", lambda d, c: rate_line(d, "family_conflict_or_abuse", True, "Conflictual / Abusive Family", c)),
        S("Class, Family and Disability", "Family Dynamics", "Stacked family dynamics by decade.", lambda d, c: stacked_percent(d, "decade", "family_dynamics", "Family Dynamics", c)),
        S("Class, Family and Disability", "Divorce by Decade", "Divorce presence by decade.", lambda d, c: rate_line(d, "divorce_presence", True, "Divorce by Decade", c)),
        S("Class, Family and Disability", "Physical Disability", "Physical disability by decade.", lambda d, c: rate_line(d, "physical_disability_representation", True, "Physical Disability", c)),
        S("Class, Family and Disability", "Physical Disability by Genre", "Physical disability by genre.", lambda d, c: boolean_rate_by_category(d, "primary_genre", "physical_disability_representation", "Physical Disability by Genre", c)),
        S("Class, Family and Disability", "Family vs Ending Tone", "Family dynamics and ending tone.", lambda d, c: crosstab_heatmap(d, "family_dynamics", "ending_tone", "Family vs Ending Tone", c)),
        S("Class, Family and Disability", "Class vs Ending Tone", "Class representation and ending tone.", lambda d, c: crosstab_heatmap(d, "class_representation", "ending_tone", "Class vs Ending Tone", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Average Violence Intensity", "Mean violence intensity by decade.", lambda d, c: mean_line(d, "violence_intensity", "Average Violence Intensity", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Violence by Genre", "Violence boxplot by genre.", lambda d, c: boxplot(d, "primary_genre", "violence_intensity", "Violence by Genre", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Violence Distribution", "Stacked violence intensity by decade.", lambda d, c: stacked_percent(d, "decade", "violence_intensity", "Violence Distribution", c)),
        S("Violence, Morality, Language, Sex and Drugs", "High Violence", "Violence intensity >= 4.", lambda d, c: high_numeric_rate(d, "violence_intensity", 4, "High Violence", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Violence Type", "Stacked violence type by decade.", lambda d, c: stacked_percent(d, "decade", "violence_type", "Violence Type", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Psychological vs Physical Violence", "Psychological and physical violence lines.", lambda d, c: multi_rate_line(d, "violence_type", ["psychological", "physical"], "Psychological vs Physical Violence", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Violence Consequences", "Stacked violence consequences.", lambda d, c: stacked_percent(d, "decade", "violence_consequences", "Violence Consequences", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Rewarded Violence", "Rewarded violence by decade.", lambda d, c: rate_line(d, "violence_consequences", "rewarded", "Rewarded Violence", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Strong Language", "Strong language frequency.", lambda d, c: stacked_percent(d, "decade", "strong_language_frequency", "Strong Language", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Frequent Strong Language by Genre", "Frequent strong language by genre.", lambda d, c: grouped_bar_rate(d, "primary_genre", "strong_language_frequency", "frequent", "Frequent Strong Language by Genre", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Explicit Scenes", "Mean explicit scenes by decade.", lambda d, c: mean_line(d, "explicit_scene_count", "Explicit Scenes", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Moral Ambiguity", "Stacked moral ambiguity.", lambda d, c: stacked_percent(d, "decade", "moral_ambiguity", "Moral Ambiguity", c)),
        S("Violence, Morality, Language, Sex and Drugs", "High Moral Ambiguity", "High moral ambiguity rate.", lambda d, c: rate_line(d, "moral_ambiguity", "high", "High Moral Ambiguity", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Drug Culture", "Drug culture by decade.", lambda d, c: rate_line(d, "drug_culture_presence", True, "Drug Culture", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Drug Culture by Genre", "Drug culture by genre.", lambda d, c: boolean_rate_by_category(d, "primary_genre", "drug_culture_presence", "Drug Culture by Genre", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Drugs vs Violence", "Drug culture and violence heatmap.", lambda d, c: crosstab_heatmap(d, "drug_culture_presence", "violence_intensity", "Drugs vs Violence", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Moral Ambiguity vs Ending", "Moral ambiguity and ending tone.", lambda d, c: crosstab_heatmap(d, "moral_ambiguity", "ending_tone", "Moral Ambiguity vs Ending", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Violence vs Overall Tone", "Violence boxplot by tone.", lambda d, c: boxplot(d, "overall_tone", "violence_intensity", "Violence vs Overall Tone", c)),
        S("Violence, Morality, Language, Sex and Drugs", "Violence vs Bechdel", "Violence boxplot by Bechdel result.", lambda d, c: boxplot(d, "bechdel_test", "violence_intensity", "Violence vs Bechdel", c)),
        S("Institutions and Power", "Law Enforcement Portrayal", "Stacked law enforcement portrayal.", lambda d, c: stacked_percent(d, "decade", "law_enforcement_portrayal", "Law Enforcement Portrayal", c)),
        S("Institutions and Power", "Negative / Corrupt Police", "Negative or corrupt law portrayal.", lambda d, c: rate_line(d, "law_negative_or_corrupt", True, "Negative / Corrupt Police", c)),
        S("Institutions and Power", "Positive Police", "Positive law portrayal.", lambda d, c: rate_line(d, "law_enforcement_portrayal", "positive", "Positive Police", c)),
        S("Institutions and Power", "Government Representation", "Stacked government representation.", lambda d, c: stacked_percent(d, "decade", "government_representation", "Government Representation", c)),
        S("Institutions and Power", "Bad Government", "Oppressive or incompetent government.", lambda d, c: rate_line(d, "government_bad", True, "Bad Government", c)),
        S("Institutions and Power", "Trustworthy Government", "Trustworthy government.", lambda d, c: rate_line(d, "government_representation", "trustworthy", "Trustworthy Government", c)),
        S("Institutions and Power", "Corporate Power", "Stacked corporate power portrayal.", lambda d, c: stacked_percent(d, "decade", "corporate_power_portrayal", "Corporate Power", c)),
        S("Institutions and Power", "Negative Corporations", "Negative corporate power portrayal.", lambda d, c: rate_line(d, "corporate_power_portrayal", "negative", "Negative Corporations", c)),
        S("Institutions and Power", "Institutional Negativity", "Combined negative institution selector.", lambda d, c: multi_rate_line(d, "law_enforcement_portrayal", ["negative", "corrupt"], "Institutional Negativity", c)),
        S("Institutions and Power", "Institutions by Genre", "Genre by institution portrayal.", lambda d, c: crosstab_heatmap(d, "primary_genre", "law_enforcement_portrayal", "Institutions by Genre", c)),
        S("Institutions and Power", "Government vs Political Climate", "Government and political climate.", lambda d, c: crosstab_heatmap(d, "political_climate_reference", "government_representation", "Government vs Political Climate", c)),
        S("Institutions and Power", "Police vs Racial Diversity", "Negative/corrupt police by diversity.", lambda d, c: grouped_rate_by_bool(d, "racial_diversity_presence", "law_negative_or_corrupt", "Police vs Racial Diversity", c)),
        S("Technology, Science and Environment", "Environmental Concerns", "Stacked environmental concerns.", lambda d, c: stacked_percent(d, "decade", "environmental_concerns", "Environmental Concerns", c)),
        S("Technology, Science and Environment", "Central Environment", "Central environmental concern.", lambda d, c: rate_line(d, "environmental_concerns", "central", "Central Environment", c)),
        S("Technology, Science and Environment", "AI in Scripts", "AI presence by decade.", lambda d, c: rate_line(d, "ai_presence", True, "AI in Scripts", c)),
        S("Technology, Science and Environment", "AI by Genre", "AI presence by genre.", lambda d, c: boolean_rate_by_category(d, "primary_genre", "ai_presence", "AI by Genre", c)),
        S("Technology, Science and Environment", "Human vs Technology", "Human vs technology conflict.", lambda d, c: rate_line(d, "human_vs_technology_conflict", True, "Human vs Technology", c)),
        S("Technology, Science and Environment", "Digital Revolution", "Digital revolution presence.", lambda d, c: rate_line(d, "digital_revolution_presence", True, "Digital Revolution", c)),
        S("Technology, Science and Environment", "Scientific Progress Portrayal", "Stacked science portrayal.", lambda d, c: stacked_percent(d, "decade", "scientific_progress_portrayal", "Scientific Progress Portrayal", c)),
        S("Technology, Science and Environment", "Feared Science", "Feared scientific progress.", lambda d, c: rate_line(d, "scientific_progress_portrayal", "feared", "Feared Science", c)),
        S("Technology, Science and Environment", "AI vs Feared Science", "AI and science portrayal.", lambda d, c: crosstab_heatmap(d, "ai_presence", "scientific_progress_portrayal", "AI vs Feared Science", c)),
        S("Technology, Science and Environment", "AI vs Tech Conflict", "Tech conflict by AI presence.", lambda d, c: grouped_rate_by_bool(d, "ai_presence", "human_vs_technology_conflict", "AI vs Tech Conflict", c)),
        S("Technology, Science and Environment", "Environment by Genre", "Environmental concern by genre.", lambda d, c: heatmap_percent(d, "primary_genre", "environmental_concerns", "Environment by Genre", c)),
        S("Technology, Science and Environment", "Technology vs Ending Tone", "Technology conflict and ending tone.", lambda d, c: crosstab_heatmap(d, "human_vs_technology_conflict", "ending_tone", "Technology vs Ending Tone", c)),
        S("History, War and Politics", "War References", "Stacked war references.", lambda d, c: stacked_percent(d, "decade", "war_reference", "War References", c)),
        S("History, War and Politics", "WW2 / Vietnam / Cold War", "War reference lines.", lambda d, c: multi_rate_line(d, "war_reference", ["WW2", "Vietnam", "Cold War"], "WW2 / Vietnam / Cold War", c)),
        S("History, War and Politics", "War Portrayal", "Stacked war portrayal.", lambda d, c: stacked_percent(d, "decade", "war_portrayal", "War Portrayal", c)),
        S("History, War and Politics", "Anti-war Cinema", "Anti-war portrayal by decade.", lambda d, c: rate_line(d, "war_portrayal", "anti-war", "Anti-war Cinema", c)),
        S("History, War and Politics", "Historical Events", "Historical events heatmap.", lambda d, c: heatmap_percent(d, "decade", "historical_event_reference", "Historical Events", c, c.top_n)),
        S("History, War and Politics", "Historical Salience", "Stacked historical salience.", lambda d, c: stacked_percent(d, "decade", "historical_event_salience", "Historical Salience", c)),
        S("History, War and Politics", "Central Historical Events", "Central historical salience.", lambda d, c: rate_line(d, "historical_event_salience", "central", "Central Historical Events", c)),
        S("History, War and Politics", "Political Climate", "Stacked political climate.", lambda d, c: stacked_percent(d, "decade", "political_climate_reference", "Political Climate", c, c.top_n)),
        S("History, War and Politics", "Post-9/11 Terrorism", "Post-9/11 terrorism reference.", lambda d, c: rate_line(d, "political_climate_reference", "post-9/11 terrorism", "Post-9/11 Terrorism", c)),
        S("History, War and Politics", "Financial Crisis", "Financial crisis reference.", lambda d, c: rate_line(d, "political_climate_reference", "financial crisis", "Financial Crisis", c)),
        S("History, War and Politics", "Social Movements", "Social movement heatmap.", lambda d, c: heatmap_percent(d, "decade", "social_movement_focus", "Social Movements", c, c.top_n)),
        S("History, War and Politics", "Civil Rights / Women's Liberation / LGBTQ+ Rights", "Movement lines.", lambda d, c: multi_rate_line(d, "social_movement_focus", ["civil rights", "women's liberation", "LGBTQ+ rights"], "Civil Rights / Women's Liberation / LGBTQ+ Rights", c)),
        S("History, War and Politics", "Movements by Genre", "Movement focus by genre.", lambda d, c: heatmap_percent(d, "primary_genre", "social_movement_focus", "Movements by Genre", c, c.top_n)),
        S("History, War and Politics", "Politics vs Government", "Political climate and government.", lambda d, c: crosstab_heatmap(d, "political_climate_reference", "government_representation", "Politics vs Government", c)),
        S("History, War and Politics", "War vs Overall Tone", "War reference and tone.", lambda d, c: crosstab_heatmap(d, "war_reference", "overall_tone", "War vs Overall Tone", c)),
        S("History, War and Politics", "War vs Violence", "Violence by war reference.", lambda d, c: boxplot(d, "war_reference", "violence_intensity", "War vs Violence", c)),
        S("Tone, Emotion and Themes", "Overall Tone by Decade", "Stacked overall tone.", lambda d, c: stacked_percent(d, "decade", "overall_tone", "Overall Tone by Decade", c, c.top_n)),
        S("Tone, Emotion and Themes", "Dark Tone by Decade", "Dark tone rate.", lambda d, c: rate_line(d, "overall_tone", "dark", "Dark Tone by Decade", c)),
        S("Tone, Emotion and Themes", "Satirical / Ironic Tone", "Satirical and ironic tone.", lambda d, c: multi_rate_line(d, "overall_tone", ["satirical", "ironic"], "Satirical / Ironic Tone", c)),
        S("Tone, Emotion and Themes", "Average Emotional Intensity", "Mean emotional intensity.", lambda d, c: mean_line(d, "emotional_intensity", "Average Emotional Intensity", c)),
        S("Tone, Emotion and Themes", "Emotional Intensity by Genre", "Emotional intensity boxplot.", lambda d, c: boxplot(d, "primary_genre", "emotional_intensity", "Emotional Intensity by Genre", c)),
        S("Tone, Emotion and Themes", "Tonal Shifts", "Stacked tonal shift presence.", lambda d, c: stacked_percent(d, "decade", "tonal_shift_presence", "Tonal Shifts", c)),
        S("Tone, Emotion and Themes", "Major Tonal Shifts", "Major tonal shift rate.", lambda d, c: rate_line(d, "tonal_shift_presence", "major", "Major Tonal Shifts", c)),
        S("Tone, Emotion and Themes", "Humor Type", "Stacked humor type.", lambda d, c: stacked_percent(d, "decade", "humor_type", "Humor Type", c, c.top_n)),
        S("Tone, Emotion and Themes", "Dark Humor", "Dark humor rate.", lambda d, c: rate_line(d, "humor_type", "dark", "Dark Humor", c)),
        S("Tone, Emotion and Themes", "Ending Tone", "Stacked ending tone.", lambda d, c: stacked_percent(d, "decade", "ending_tone", "Ending Tone", c)),
        S("Tone, Emotion and Themes", "Happy Endings", "Happy ending rate.", lambda d, c: rate_line(d, "ending_tone", "happy", "Happy Endings", c)),
        S("Tone, Emotion and Themes", "Tragic Endings", "Tragic ending rate.", lambda d, c: rate_line(d, "ending_tone", "tragic", "Tragic Endings", c)),
        S("Tone, Emotion and Themes", "Ambiguous / Disturbing Endings", "Ambiguous and disturbing endings.", lambda d, c: multi_rate_line(d, "ending_tone", ["ambiguous", "disturbing"], "Ambiguous / Disturbing Endings", c)),
        S("Tone, Emotion and Themes", "Primary Universal Theme", "Theme heatmap by decade.", lambda d, c: heatmap_percent(d, "decade", "primary_universal_theme", "Primary Universal Theme", c, c.top_n)),
        S("Tone, Emotion and Themes", "Theme by Genre", "Theme by genre heatmap.", lambda d, c: heatmap_percent(d, "primary_genre", "primary_universal_theme", "Theme by Genre", c, c.top_n)),
        S("Tone, Emotion and Themes", "Identity Theme by Decade", "Identity theme rate.", lambda d, c: rate_line(d, "identity_theme", True, "Identity Theme by Decade", c)),
        S("Tone, Emotion and Themes", "Power Theme by Decade", "Power theme rate.", lambda d, c: rate_line(d, "power_theme", True, "Power Theme by Decade", c)),
        S("Tone, Emotion and Themes", "Freedom vs Control Theme", "Freedom vs control theme rate.", lambda d, c: rate_line(d, "freedom_control_theme", True, "Freedom vs Control Theme", c)),
        S("Tone, Emotion and Themes", "Theme vs Ending Tone", "Theme and ending tone.", lambda d, c: crosstab_heatmap(d, "primary_universal_theme", "ending_tone", "Theme vs Ending Tone", c)),
        S("Tone, Emotion and Themes", "Theme vs Protagonist", "Theme and protagonist gender.", lambda d, c: crosstab_heatmap(d, "primary_universal_theme", "protagonist_gender", "Theme vs Protagonist", c)),
        S("Cross-analysis", "Numeric Feature Correlations", "Correlation heatmap across numeric screenplay and representation metrics.", lambda d, c: numeric_correlation_heatmap(d, c)),
        S("Cross-analysis", "Binary Feature Co-occurrence", "Pairwise co-occurrence heatmap for the main binary and derived flags.", lambda d, c: binary_feature_cooccurrence_heatmap(d, c)),
        S("Cross-analysis", "Has Female Representation Increased?", "Female dialogue by decade and genre.", lambda d, c: mean_line(d, "female_dialogue_share_pct", "Has Female Representation Increased?", c, "primary_genre")),
        S("Cross-analysis", "Does Bechdel Improve Over Time?", "Bechdel over time by genre.", lambda d, c: rate_line(d, "bechdel_test", True, "Does Bechdel Improve Over Time?", c, "primary_genre")),
        S("Cross-analysis", "Female Protagonists and Ending Tone", "Protagonist gender and ending tone.", lambda d, c: crosstab_heatmap(d, "protagonist_gender", "ending_tone", "Female Protagonists and Ending Tone", c)),
        S("Cross-analysis", "Is Cinema Getting Darker?", "Dark tone by decade and genre.", lambda d, c: rate_line(d, "overall_tone", "dark", "Is Cinema Getting Darker?", c, "primary_genre")),
        S("Cross-analysis", "Is Moral Ambiguity Rising?", "High moral ambiguity by genre.", lambda d, c: rate_line(d, "moral_ambiguity", "high", "Is Moral Ambiguity Rising?", c, "primary_genre")),
        S("Cross-analysis", "Does the Antihero Grow?", "Antihero by decade and genre.", lambda d, c: rate_line(d, "protagonist_type", "antihero", "Does the Antihero Grow?", c, "primary_genre")),
        S("Cross-analysis", "Does Violence Increase?", "Violence intensity by genre.", lambda d, c: mean_line(d, "violence_intensity", "Does Violence Increase?", c, "primary_genre")),
        S("Cross-analysis", "Is Violence Punished Less?", "Minimal or rewarded violence by genre.", lambda d, c: rate_line(d, "violence_minimal_or_rewarded", True, "Is Violence Punished Less?", c, "primary_genre")),
        S("Cross-analysis", "Does Urban Cinema Dominate More?", "Urban setting by genre.", lambda d, c: rate_line(d, "geographic_setting", "urban", "Does Urban Cinema Dominate More?", c, "primary_genre")),
        S("Cross-analysis", "Technology Since the 90s / 2000s", "AI, digital and tech conflict over time.", lambda d, c: multi_metric_line(d, ["ai_presence", "digital_revolution_presence", "human_vs_technology_conflict"], "Technology Since the 90s / 2000s", c, (0, 1))),
        S("Cross-analysis", "Is AI Associated with Fear?", "AI and feared science.", lambda d, c: grouped_bar_rate(d, "ai_presence", "scientific_progress_portrayal", "feared", "Is AI Associated with Fear?", c)),
        S("Cross-analysis", "Are AI Films More Dystopian?", "AI and ending tone.", lambda d, c: crosstab_heatmap(d, "ai_presence", "ending_tone", "Are AI Films More Dystopian?", c)),
        S("Cross-analysis", "Racial Diversity and Minority Salience", "Medium/high minority salience by diversity.", lambda d, c: grouped_rate_by_bool(d, "racial_diversity_presence", "minority_salience_medium_high", "Racial Diversity and Minority Salience", c)),
        S("Cross-analysis", "Police Portrayal and Racial Diversity", "Negative/corrupt police by diversity.", lambda d, c: grouped_rate_by_bool(d, "racial_diversity_presence", "law_negative_or_corrupt", "Police Portrayal and Racial Diversity", c)),
        S("Cross-analysis", "Post-9/11 Climate and Tone", "Political climate and dark/tense tone.", lambda d, c: grouped_rate_by_bool(d, "political_climate_reference", "dark_or_tense_tone", "Post-9/11 Climate and Tone", c)),
        S("Cross-analysis", "Economic Struggle and Ending Tone", "Economic struggle and ending tone.", lambda d, c: crosstab_heatmap(d, "economic_struggle_presence", "ending_tone", "Economic Struggle and Ending Tone", c)),
        S("Cross-analysis", "Class and Overall Tone", "Class and tone heatmap.", lambda d, c: crosstab_heatmap(d, "class_representation", "overall_tone", "Class and Overall Tone", c)),
        S("Cross-analysis", "Family Conflict and Emotional Intensity", "Emotional intensity by family dynamics.", lambda d, c: boxplot(d, "family_dynamics", "emotional_intensity", "Family Conflict and Emotional Intensity", c)),
        S("Cross-analysis", "Drugs and Violence", "Violence by drug culture.", lambda d, c: boxplot(d, "drug_culture_presence", "violence_intensity", "Drugs and Violence", c)),
        S("Cross-analysis", "Strong Language Over Time", "Frequent strong language by genre.", lambda d, c: rate_line(d, "strong_language_frequency", "frequent", "Strong Language Over Time", c, "primary_genre")),
        S("Cross-analysis", "Female Dialogue and Bechdel", "Female dialogue by Bechdel result.", lambda d, c: boxplot(d, "bechdel_test", "female_dialogue_share_pct", "Female Dialogue and Bechdel", c, "primary_genre")),
        S("Cross-analysis", "Female Protagonists by Genre Over Time", "Female protagonist heatmap.", lambda d, c: female_protagonist_heatmap(d, c)),
        S("Cross-analysis", "Ambiguous Endings and Complexity", "Plot complexity by ending type.", lambda d, c: boxplot(d, "ending_type", "plot_complexity", "Ambiguous Endings and Complexity", c, "primary_genre")),
        S("Cross-analysis", "Are Non-linear Stories More Complex?", "Plot complexity by non-linear usage.", lambda d, c: boxplot(d, "non_linear_usage", "plot_complexity", "Are Non-linear Stories More Complex?", c, "primary_genre")),
        S("Cross-analysis", "Scenes and Complexity", "Scene count and plot complexity.", lambda d, c: scatter(d, "scene_count", "plot_complexity", "Scenes and Complexity", c)),
        S("Cross-analysis", "Dialogue and Violence", "Dialogue density and violence.", lambda d, c: scatter(d, "dialogue_density_pct", "violence_intensity", "Dialogue and Violence", c)),
        S("Cross-analysis", "Historical Cinema and Emotion", "Emotion by historical salience.", lambda d, c: boxplot(d, "historical_event_salience", "emotional_intensity", "Historical Cinema and Emotion", c, "primary_genre")),
        S("Cross-analysis", "Are War Films More Anti-war Over Time?", "Anti-war portrayal by war reference.", lambda d, c: rate_line(d, "war_portrayal", "anti-war", "Are War Films More Anti-war Over Time?", c, "war_reference")),
        S("Cross-analysis", "Identity Theme, LGBTQ+ and Racial Diversity", "Identity theme, LGBTQ+ and racial diversity.", lambda d, c: identity_representation_line(d, c)),
    ]


def selected_unknown_line(df: pd.DataFrame, controls: Controls) -> go.Figure:
    candidates = [c for c in df.columns if not c.startswith("evidence__") and c not in {"source_file"}]
    feature = st.selectbox("Feature for unknown rate", sorted(candidates, key=label), format_func=label)
    plot_df = df.dropna(subset=["decade"]).copy()
    plot_df["_unknown"] = plot_df[feature].isna() | plot_df[feature].astype(str).str.lower().isin(["unknown", "nan", ""])
    grouped = plot_df.groupby("decade")["_unknown"].mean().reset_index(name="Percent")
    grouped["Percent"] *= 100
    grouped = sorted_by_decade(grouped)
    fig = px.line(grouped, x="decade", y="Percent", markers=True, title=f"Unknown Rate: {label(feature)}")
    return finish_fig(fig, percent_y=True)


def female_protagonist_heatmap(df: pd.DataFrame, controls: Controls) -> go.Figure:
    plot_df = valid_df(df, ["decade", "primary_genre", "protagonist_gender"], controls)
    plot_df = plot_df[plot_df["primary_genre"].isin(plot_df["primary_genre"].value_counts().head(controls.top_n).index)]
    plot_df["_female"] = plot_df["protagonist_gender"].eq("female")
    grouped = plot_df.groupby(["primary_genre", "decade"], dropna=False)["_female"].mean().reset_index(name="Percent")
    grouped["Percent"] *= 100
    fig = px.density_heatmap(grouped, x="primary_genre", y="decade", z="Percent", histfunc="sum", title="Female Protagonists by Genre", color_continuous_scale=["#f4eee5", "#0f766e"])
    fig.update_coloraxes(colorbar_ticksuffix="%")
    return finish_fig(fig, height=520)


def ranking_bar(df: pd.DataFrame, metric: str, title: str, controls: Controls) -> go.Figure:
    plot_df = df.dropna(subset=[metric]).nlargest(controls.top_n, metric).sort_values(metric)
    fig = px.bar(plot_df, x=metric, y="title", orientation="h", color="primary_genre", title=title, labels={metric: label(metric), "title": "Film"})
    return finish_fig(fig, height=max(420, controls.top_n * 28))


def dialogue_gap_line(df: pd.DataFrame, controls: Controls) -> go.Figure:
    fig = mean_line(df, "dialogue_gender_gap", "Dialogue Gender Gap", controls)
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
    return fig


def high_numeric_rate(df: pd.DataFrame, metric: str, threshold: float, title: str, controls: Controls) -> go.Figure:
    plot_df = df.dropna(subset=["decade", metric]).copy()
    plot_df["_hit"] = plot_df[metric] >= threshold
    grouped = plot_df.groupby("decade")["_hit"].mean().reset_index(name="Percent")
    grouped["Percent"] *= 100
    grouped = sorted_by_decade(grouped)
    fig = px.line(grouped, x="decade", y="Percent", markers=True, title=title)
    return finish_fig(fig, percent_y=True)


def grouped_rate_by_bool(df: pd.DataFrame, group: str, flag: str, title: str, controls: Controls) -> go.Figure:
    plot_df = valid_df(df, [group, flag], controls).dropna(subset=[group])
    grouped = plot_df.groupby(group, dropna=False)[flag].mean().reset_index(name="Percent")
    grouped["Percent"] *= 100
    fig = px.bar(grouped, x=group, y="Percent", title=title, labels={group: label(group), "Percent": "%"})
    return finish_fig(fig, percent_y=True)


def identity_representation_line(df: pd.DataFrame, controls: Controls) -> go.Figure:
    plot_df = df.dropna(subset=["decade"]).copy()
    metrics = {
        "Identity theme": "identity_theme",
        "LGBTQ+ presence": "lgbtq_presence",
        "Racial diversity": "racial_diversity_presence",
    }
    parts = []
    for name, col in metrics.items():
        if col in plot_df:
            tmp = plot_df.groupby("decade")[col].mean().reset_index(name="Percent")
            tmp["Percent"] *= 100
            tmp["Metric"] = name
            parts.append(tmp)
    long = pd.concat(parts, ignore_index=True)
    long = sorted_by_decade(long)
    fig = px.line(long, x="decade", y="Percent", color="Metric", markers=True, title="Identity Theme, LGBTQ+ and Racial Diversity")
    return finish_fig(fig, percent_y=True)


def render_chart_grid(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec], columns: int = 2) -> None:
    for i in range(0, len(specs), columns):
        cols = st.columns(columns)
        for col, spec in zip(cols, specs[i : i + columns]):
            with col:
                try:
                    show_chart(spec.builder(df, controls))
                except Exception as exc:
                    st.warning(f"{spec.title} could not be rendered: {exc}")


def select_specs(specs: list[ChartSpec], titles: list[str]) -> list[ChartSpec]:
    by_title = {spec.title: spec for spec in specs}
    return [by_title[title] for title in titles if title in by_title]


def kpi_row(df: pd.DataFrame) -> None:
    c = st.columns(5)
    c[0].metric("Films analyzed", f"{len(df):,}")
    if "release_year" in df and df["release_year"].notna().any():
        c[1].metric("Year range", f"{int(df['release_year'].min())}-{int(df['release_year'].max())}")
        c[2].metric("Decades covered", f"{df['decade'].nunique():,}")
        c[3].metric("Known year", f"{df['release_year'].notna().mean() * 100:.1f}%")
    else:
        c[1].metric("Year range", "n/a")
        c[2].metric("Decades covered", "n/a")
        c[3].metric("Known year", "n/a")
    c[4].metric("Bechdel true", f"{df['bechdel_test'].mean() * 100:.1f}%" if "bechdel_test" in df else "n/a")
    c = st.columns(4)
    c[0].metric("LGBTQ+ presence", f"{df['lgbtq_presence'].mean() * 100:.1f}%" if "lgbtq_presence" in df else "n/a")
    c[1].metric("Racial diversity", f"{df['racial_diversity_presence'].mean() * 100:.1f}%" if "racial_diversity_presence" in df else "n/a")
    c[2].metric("Avg. violence", f"{df['violence_intensity'].mean():.2f}" if "violence_intensity" in df else "n/a")
    c[3].metric("Avg. female dialogue", f"{df['female_dialogue_share_pct'].mean():.1f}%" if "female_dialogue_share_pct" in df else "n/a")


def overview_dashboard(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec]) -> None:
    st.markdown('<div class="kicker">Dashboard 1 - General Overview</div>', unsafe_allow_html=True)
    kpi_row(df)
    wanted = [
        "Numeric Feature Correlations",
        "Binary Feature Co-occurrence",
        "Films by Decade",
        "Genre Evolution",
        "Coverage by Decade and Genre",
        "Extraction Quality",
    ]
    render_chart_grid(df, controls, select_specs(specs, wanted))
    st.markdown("#### Filtered Films")
    filtered_table(df, controls)


def social_dashboard(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec]) -> None:
    st.markdown('<div class="kicker">Dashboard 2 - Social Evolution</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="note">These features are conservative: a false or unknown value often means no explicit evidence was detected in the script, not necessarily real-world absence.</p>',
        unsafe_allow_html=True,
    )
    wanted = [
        "Bechdel by Decade",
        "Female Dialogue by Decade",
        "Protagonist Gender",
        "LGBTQ+ Presence by Decade",
        "Explicit Racial Diversity",
        "Economic Struggle",
        "Physical Disability",
        "Social Movements",
        "Minority Tone",
        "Identity Theme by Decade",
    ]
    render_chart_grid(df, controls, select_specs(specs, wanted))


def narrative_dashboard(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec]) -> None:
    st.markdown('<div class="kicker">Dashboard 3 - Narrative Evolution</div>', unsafe_allow_html=True)
    wanted = [
        "Genre Evolution",
        "Average Plot Complexity",
        "Non-linear Narrative",
        "Opening Types",
        "Ending Types",
        "Ending Tone",
        "Protagonist Type",
        "Antihero by Decade",
        "Character Arc Pattern",
        "Relationship Structure",
    ]
    render_chart_grid(df, controls, select_specs(specs, wanted))


def violence_dashboard(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec]) -> None:
    st.markdown('<div class="kicker">Dashboard 4 - Violence, Morality and Tone</div>', unsafe_allow_html=True)
    wanted = [
        "Average Violence Intensity",
        "Violence Type",
        "Violence Consequences",
        "High Moral Ambiguity",
        "Strong Language",
        "Drug Culture",
        "Dark Tone by Decade",
        "Ambiguous / Disturbing Endings",
        "Moral Ambiguity vs Ending",
        "Drugs vs Violence",
    ]
    render_chart_grid(df, controls, select_specs(specs, wanted))


def institutions_dashboard(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec]) -> None:
    st.markdown('<div class="kicker">Dashboard 5 - Institutions, Politics and History</div>', unsafe_allow_html=True)
    wanted = [
        "Law Enforcement Portrayal",
        "Government Representation",
        "Negative Corporations",
        "War References",
        "War Portrayal",
        "Historical Events",
        "Historical Salience",
        "Political Climate",
        "Social Movements",
        "Government vs Political Climate",
    ]
    render_chart_grid(df, controls, select_specs(specs, wanted))


def technology_dashboard(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec]) -> None:
    st.markdown('<div class="kicker">Dashboard 6 - Technology, Science and the Future</div>', unsafe_allow_html=True)
    wanted = [
        "AI in Scripts",
        "Digital Revolution",
        "Human vs Technology",
        "Scientific Progress Portrayal",
        "Environmental Concerns",
        "AI vs Feared Science",
        "Technology vs Ending Tone",
        "Central Environment",
        "Is Cinema Getting Darker?",
        "Are AI Films More Dystopian?",
    ]
    render_chart_grid(df, controls, select_specs(specs, wanted))


def cross_analysis_dashboard(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec]) -> None:
    st.markdown('<div class="kicker">Dashboard 7 - Cross-analysis</div>', unsafe_allow_html=True)
    wanted = [
        "Numeric Feature Correlations",
        "Binary Feature Co-occurrence",
        "Has Female Representation Increased?",
        "Does Bechdel Improve Over Time?",
        "Is Cinema Getting Darker?",
        "Does Violence Increase?",
        "Economic Struggle and Ending Tone",
        "Female Dialogue and Bechdel",
        "Ambiguous Endings and Complexity",
        "Scenes and Complexity",
        "Dialogue and Violence",
        "Identity Theme, LGBTQ+ and Racial Diversity",
    ]
    render_chart_grid(df, controls, select_specs(specs, wanted))


def explorer_dashboard(df: pd.DataFrame, controls: Controls, specs: list[ChartSpec]) -> None:
    st.markdown('<div class="kicker">Chart Explorer</div>', unsafe_allow_html=True)
    sections = ["All"] + sorted({s.section for s in specs})
    section = st.selectbox("Chart section", sections)
    visible = specs if section == "All" else [s for s in specs if s.section == section]
    search = st.text_input("Search charts", placeholder="Bechdel, violence, AI, protagonist...")
    if search:
        visible = [s for s in visible if search.lower() in s.title.lower() or search.lower() in s.description.lower()]

    titles = [s.title for s in visible]
    selected = st.selectbox("Recommended charts", titles)
    spec = next(s for s in visible if s.title == selected)
    st.caption(spec.description)
    try:
        show_chart(spec.builder(df, controls))
    except Exception as exc:
        st.error(f"This chart could not be rendered: {exc}")

    with st.expander("Custom chart"):
        custom_chart(df, controls)

    if controls.show_evidence:
        evidence_views(df)


def custom_chart(df: pd.DataFrame, controls: Controls) -> None:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "source_file"]
    categorical = [c for c in df.columns if c not in numeric and not c.startswith("evidence__") and df[c].nunique(dropna=True) < 300]
    c1, c2, c3, c4 = st.columns(4)
    chart_type = c1.selectbox("Chart type", ["Scatter", "Bar", "Line", "Box", "Violin", "Heatmap"])
    x = c2.selectbox("X axis", sorted(categorical + numeric, key=label), format_func=label)
    y_options = ["Count"] + sorted(numeric, key=label)
    y = c3.selectbox("Y axis", y_options, format_func=lambda v: v if v == "Count" else label(v))
    color = c4.selectbox("Color", ["None"] + sorted(categorical, key=label), format_func=lambda v: v if v == "None" else label(v))
    color_col = None if color == "None" else color
    try:
        if chart_type == "Scatter" and y != "Count":
            fig = scatter(df, x, y, "Custom Scatter", controls, color_col or "primary_genre")
        elif chart_type == "Line" and y != "Count":
            fig = mean_line(df, y, "Custom Line", controls, color_col)
        elif chart_type == "Box" and y != "Count":
            fig = boxplot(df, x, y, "Custom Boxplot", controls, color_col)
        elif chart_type == "Violin" and y != "Count":
            fig = violin(df, x, y, "Custom Violin", controls)
        elif chart_type == "Heatmap":
            yy = c3.selectbox("Heatmap Y axis", sorted(categorical, key=label), format_func=label, key="custom_heatmap_y")
            fig = heatmap_percent(df, x, yy, "Custom Heatmap", controls, controls.top_n)
        else:
            fig = bar_count(df, x, "Custom Bar Chart", controls, color_col)
        show_chart(fig)
    except Exception as exc:
        st.warning(f"Custom chart could not be rendered: {exc}")


def evidence_column_for(feature: str) -> str | None:
    candidates = [f"evidence__{feature}", f"evidence__{feature}_evidence", f"evidence__{feature}_payload"]
    return next((c for c in candidates if c in st.session_state.get("_columns", [])), None)


def evidence_views(df: pd.DataFrame) -> None:
    st.markdown("#### Evidence Payload Views")
    st.session_state["_columns"] = list(df.columns)
    view = st.selectbox(
        "Evidence table",
        [
            "Bechdel evidence",
            "LGBTQ+ examples",
            "Historical events",
            "Social movements",
            "Minorities detected",
            "Protagonist race evidence",
            "Low implicit confidence",
            "Film search",
        ],
    )
    if view == "Film search":
        cols = [c for c in df.columns if not c.startswith("evidence__")]
        search = st.text_input("Search films in table", key="evidence_search")
        out = df[df["title"].astype(str).str.contains(search, case=False, na=False)] if search else df
        st.dataframe(out[[c for c in ["title", "release_year", "decade", "primary_genre"] + cols if c in out]].head(300), width="stretch", hide_index=True)
        return
    feature_map = {
        "Bechdel evidence": "bechdel_test",
        "LGBTQ+ examples": "lgbtq_presence",
        "Historical events": "historical_event_reference",
        "Social movements": "social_movement_focus",
        "Minorities detected": "racial_diversity_presence",
        "Protagonist race evidence": "protagonist_race_coding",
    }
    if view == "Low implicit confidence":
        unknown_cols = [c for c in FEATURE_LABELS if c in df]
        selected = st.selectbox("Feature", sorted(unknown_cols, key=label), format_func=label)
        out = df[df[selected].isna() | df[selected].astype(str).str.lower().isin(["unknown", "nan", ""])]
        st.dataframe(out[["title", "release_year", "decade", "primary_genre", selected]].head(300), width="stretch", hide_index=True)
        return
    feature = feature_map[view]
    evidence_col = evidence_column_for(feature)
    cols = ["title", "release_year", "decade", "primary_genre", feature]
    if evidence_col:
        cols.append(evidence_col)
    st.dataframe(df[[c for c in cols if c in df]].head(300), width="stretch", hide_index=True)


def format_value(value: Any) -> str:
    if is_missing_value(value):
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def film_view(df: pd.DataFrame) -> None:
    st.markdown('<div class="kicker">Film Detail</div>', unsafe_allow_html=True)
    if df.empty:
        st.warning("No films match the current filters.")
        return
    selected = st.selectbox("Select a film", df["title"].astype(str).tolist())
    row = df[df["title"].astype(str) == selected].iloc[0]
    st.markdown(
        f"""
        <div class="panel">
            <h2 style="margin:0 0 6px;">{format_value(row.get("title"))}</h2>
            <div class="pill-row">
                <span class="pill">{format_value(row.get("release_year"))}</span>
                <span class="pill">{format_value(row.get("decade"))}</span>
                <span class="pill">{format_value(row.get("primary_genre"))}</span>
                <span class="pill">Bechdel: {format_value(row.get("bechdel_test"))}</span>
                <span class="pill">{format_value(row.get("overall_tone"))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c = st.columns(4)
    c[0].metric("Words", format_value(row.get("word_count")))
    c[1].metric("Scenes", format_value(row.get("scene_count")))
    c[2].metric("Female dialogue", f"{row.get('female_dialogue_share_pct'):.1f}%" if pd.notna(row.get("female_dialogue_share_pct")) else "n/a")
    c[3].metric("Violence", format_value(row.get("violence_intensity")))
    for section in BLOCKS[1:]:
        cols = [col for col in row.index if col.startswith(f"{section}__")]
        if not cols:
            continue
        st.markdown(f"#### {BLOCK_LABELS.get(section, section)}")
        cards = ['<div class="feature-grid">']
        for col in cols:
            key = col.split("__", 1)[1]
            cards.append(f'<div class="feature-card"><div class="label">{label(key)}</div><div class="value">{format_value(row[col])}</div></div>')
        cards.append("</div>")
        st.markdown("".join(cards), unsafe_allow_html=True)


def data_view(df: pd.DataFrame) -> None:
    st.markdown('<div class="kicker">Data</div>', unsafe_allow_html=True)
    block = st.selectbox("Feature block", ["All"] + [BLOCK_LABELS[b] for b in BLOCKS])
    if block == "All":
        cols = [c for c in df.columns if not c.startswith("evidence__")]
    else:
        section = next(k for k, v in BLOCK_LABELS.items() if v == block)
        base = ["title", "release_year", "decade", "primary_genre"]
        cols = base + [c for c in FEATURE_BLOCKS.get(section, []) if c in df]
    st.dataframe(df[[c for c in cols if c in df]], width="stretch", hide_index=True)
    st.download_button("Download filtered CSV", df.to_csv(index=False).encode("utf-8"), "filtered_film_analysis.csv", "text/csv")


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="hero">
            <h1>Film Narrative Analysis</h1>
            <p>
            A clean analytical workspace for exploring the final JSON feature set:
            dataset coverage, narrative structure, screenplay technique, representation,
            identity, class, violence, institutions, technology, history, tone and evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not DATA_DIR.exists():
        st.error(f"Results folder not found: {DATA_DIR}")
        st.stop()
    with st.spinner("Loading JSON files and preparing derived features..."):
        df = load_dataset(DATA_DIR, data_signature(DATA_DIR))
    if df.empty:
        st.error("No valid JSON files could be loaded.")
        st.stop()

    filtered, controls = apply_global_controls(df)
    specs = build_specs()

    tabs = st.tabs(
        [
            "Overview",
            "Social Evolution",
            "Narrative",
            "Violence & Tone",
            "Institutions & History",
            "Technology",
            "Cross-analysis",
            "Explorer",
            "Films",
            "Data",
        ]
    )
    with tabs[0]:
        overview_dashboard(filtered, controls, specs)
    with tabs[1]:
        social_dashboard(filtered, controls, specs)
    with tabs[2]:
        narrative_dashboard(filtered, controls, specs)
    with tabs[3]:
        violence_dashboard(filtered, controls, specs)
    with tabs[4]:
        institutions_dashboard(filtered, controls, specs)
    with tabs[5]:
        technology_dashboard(filtered, controls, specs)
    with tabs[6]:
        cross_analysis_dashboard(filtered, controls, specs)
    with tabs[7]:
        explorer_dashboard(filtered, controls, specs)
    with tabs[8]:
        film_view(filtered)
    with tabs[9]:
        data_view(filtered)


if __name__ == "__main__":
    main()
