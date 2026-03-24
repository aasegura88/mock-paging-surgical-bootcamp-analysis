#!/usr/bin/env python3
"""
Reproducible analysis pipeline for the mock paging manuscript.

This script:
1. Loads the de-identified survey CSV.
2. Cleans and validates the dataset.
3. Verifies composite score sums against item-level responses.
4. Builds Table 1, Table 2, Supplementary Table S1.
5. Builds Figure 1 and Supplementary Figure S1.
6. Writes a manuscript-ready results paragraph and a console summary.

Recommended repository layout
----------------------------
repo/
├── run_analysis.py
├── requirements.txt
├── README.md
├── CITATION.cff
├── data/
│   ├── README.md
│   └── raw/
│       └── Mock_Page_data_cleaned.csv
└── results/
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ID_COL = "student_id"
TIME_COL = "survey_pre_0_post_0"
CLASS_COL = "class_year"
PAGES_COL = "1_number_of_pages_answered"
CONF_COL = "2_composite_confidence"
ANX_COL = "3_composite_anxiety"
COMM_COL = "4_composite_communication"

TIME_MAP = {0: "pre", 1: "post"}
DOMAIN_ORDER = ["Confidence", "Anxiety", "Communication"]

CONFIDENCE_ITEMS = [
    ("2a_infected_wound", "Infected wound"),
    ("2b_nausea_and_vomiting", "Nausea and vomiting"),
    ("2c_acidotic_in_icu", "Acidotic patient in ICU"),
    ("2d_febrile_and_tachycardic", "Febrile and tachycardic"),
    ("2e_chest_pain", "Chest pain"),
    ("2f_shortness_of_breath", "Shortness of breath"),
    ("2g_hypertension", "Hypertension"),
    ("2h_pain_after_surgery", "Pain after surgery"),
    ("2i_cannot_sleep", "Cannot sleep"),
    ("2j_hypoxic", "Hypoxic"),
    ("2k_altered_mental_state", "Altered mental status"),
]

ANXIETY_ITEMS = [
    ("3a_infected_wound", "Infected wound"),
    ("3b_nausea_and_vomiting", "Nausea and vomiting"),
    ("3c_acidotic_in_icu", "Acidotic patient in ICU"),
    ("3d_febrile_and_tachycardic", "Febrile and tachycardic"),
    ("3e_chest_pain", "Chest pain"),
    ("3f_shortness_of_breath", "Shortness of breath"),
    ("3g_hypertension", "Hypertension"),
    ("3h_pain_after_surgery", "Pain after surgery"),
    ("3i_cannot_sleep", "Cannot sleep"),
    ("3j_hypoxic", "Hypoxic"),
    ("3k_altered_mental_state", "Altered mental status"),
]

COMMUNICATION_ITEMS = [
    ("4a_instructive", "Instructiveness"),
    ("4b_precise", "Precision"),
    ("4c_directing", "Directiveness"),
    ("4d_assertive", "Assertiveness"),
    ("4e_solicits_information", "Solicits information"),
    ("4f_engaged", "Engagement"),
    ("4g_structured", "Structure"),
    ("4h_cooperative", "Cooperation"),
]

DOMAIN_ITEMS = {
    "Confidence": {
        "composite_col": CONF_COL,
        "scale_range": "1-10",
        "items": CONFIDENCE_ITEMS,
    },
    "Anxiety": {
        "composite_col": ANX_COL,
        "scale_range": "1-10",
        "items": ANXIETY_ITEMS,
    },
    "Communication": {
        "composite_col": COMM_COL,
        "scale_range": "1-7",
        "items": COMMUNICATION_ITEMS,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full mock paging manuscript analysis."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to the input CSV. If omitted, common repository locations are checked.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for generated tables and figures. Defaults to ./results next to the input file.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster DPI for PNG and TIFF exports.",
    )
    return parser.parse_args()


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_mean_sd(values: Iterable[float]) -> str:
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        return "NA"
    if len(s) == 1:
        return f"{s.iloc[0]:.1f} (NA)"
    return f"{s.mean():.1f} ({s.std(ddof=1):.1f})"


def format_ci(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return "NA"
    return f"{low:.1f} to {high:.1f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display_df = df.fillna("").astype(str)

    widths = []
    for col in display_df.columns:
        max_width = max(len(col), display_df[col].map(len).max())
        widths.append(max_width)

    def make_row(values: list[str]) -> str:
        padded = [f" {str(v).ljust(widths[i])} " for i, v in enumerate(values)]
        return "|" + "|".join(padded) + "|"

    header = make_row(display_df.columns.tolist())
    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join(make_row(row) for row in display_df.values.tolist())
    return "\n".join([header, separator, body])


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)

    ranked = p.dropna().sort_values()
    n = len(ranked)
    if n == 0:
        return out

    adjusted = pd.Series(index=ranked.index, dtype=float)
    previous = 1.0
    for i in range(n, 0, -1):
        idx = ranked.index[i - 1]
        raw = ranked.iloc[i - 1]
        current = min(previous, raw * n / i)
        adjusted.loc[idx] = current
        previous = current

    out.loc[adjusted.index] = adjusted
    return out


def resolve_input_path(user_path: str) -> Path:
    candidates: list[Path] = []

    if user_path:
        candidates.append(Path(user_path).expanduser())

    candidates.extend(
        [
            Path.cwd() / "data" / "raw" / "Mock_Page_data_cleaned.csv",
            Path.cwd() / "Mock_Page_data_cleaned.csv",
            Path("/mnt/data/Mock_Page_data_cleaned.csv"),
            Path("/home/asegura/Mock_Page_data_cleaned.csv"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find input CSV. Checked:\n{searched}")


def resolve_output_dir(user_dir: str, input_path: Path) -> Path:
    if user_dir:
        outdir = Path(user_dir).expanduser()
    else:
        outdir = input_path.parent.parent / "results" if input_path.parent.name == "raw" else input_path.parent / "results"

    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_unique_student_time_records(df: pd.DataFrame) -> None:
    dup = (
        df.groupby([ID_COL, TIME_COL], dropna=False)
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    if not dup.empty:
        raise ValueError(
            "Found duplicate records for at least one student within the same timepoint. "
            "Resolve duplicates before running the analysis.\n\n"
            f"{dup.to_string(index=False)}"
        )


def load_data(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df.dropna(how="all").copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    required = [
        ID_COL,
        TIME_COL,
        CLASS_COL,
        PAGES_COL,
        CONF_COL,
        ANX_COL,
        COMM_COL,
    ]
    for domain_info in DOMAIN_ITEMS.values():
        required.extend(col for col, _ in domain_info["items"])

    validate_required_columns(df, sorted(set(required)))

    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype("Int64")
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").astype("Int64")
    df[CLASS_COL] = pd.to_numeric(df[CLASS_COL], errors="coerce").astype("Int64")

    for score_col in [CONF_COL, ANX_COL, COMM_COL]:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    for domain_info in DOMAIN_ITEMS.values():
        for col, _ in domain_info["items"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if PAGES_COL in df.columns:
        df[PAGES_COL] = df[PAGES_COL].astype("object").where(df[PAGES_COL].notna(), np.nan)
        page_fix = {
            "1-3 pages ": "1-3 pages",
            "4-10 pages ": "4-10 pages",
            "more than 10 ": "more than 10",
            "never ": "never",
        }
        df[PAGES_COL] = df[PAGES_COL].replace(page_fix)
        df[PAGES_COL] = df[PAGES_COL].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

    df["timepoint"] = df[TIME_COL].map(TIME_MAP)

    invalid_timepoints = df["timepoint"].isna()
    if invalid_timepoints.any():
        bad_rows = df.loc[invalid_timepoints, [ID_COL, TIME_COL]]
        raise ValueError(
            "Found invalid survey_pre_0_post_0 values. Expected only 0 or 1.\n"
            f"{bad_rows.to_string(index=False)}"
        )

    validate_unique_student_time_records(df)
    return df


def split_pre_post(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = df.loc[df["timepoint"] == "pre"].copy()
    post = df.loc[df["timepoint"] == "post"].copy()
    return pre, post


def build_wide_dataset(df: pd.DataFrame) -> pd.DataFrame:
    pre, post = split_pre_post(df)

    keep_cols = [ID_COL, CLASS_COL, CONF_COL, ANX_COL, COMM_COL]
    for domain_info in DOMAIN_ITEMS.values():
        keep_cols.extend(col for col, _ in domain_info["items"])

    wide = pre[keep_cols].merge(
        post[keep_cols],
        on=ID_COL,
        how="inner",
        suffixes=("_pre", "_post"),
    )

    if wide.empty:
        raise ValueError("No matched pre/post pairs were found after merging by student_id.")

    return wide


def validate_composites(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    pre, post = split_pre_post(df)

    validation_rows: list[dict] = []
    mismatch_tables: dict[str, pd.DataFrame] = {}

    for frame_name, frame in [("pre", pre), ("post", post)]:
        for domain_name, domain_info in DOMAIN_ITEMS.items():
            item_cols = [col for col, _ in domain_info["items"]]
            composite_col = domain_info["composite_col"]

            calculated = frame[item_cols].sum(axis=1, min_count=len(item_cols))
            recorded = frame[composite_col]
            mismatch = ~(calculated.eq(recorded) | (calculated.isna() & recorded.isna()))

            validation_rows.append(
                {
                    "timepoint": frame_name,
                    "domain": domain_name,
                    "composite_column": composite_col,
                    "n_rows": int(len(frame)),
                    "n_complete_rows": int(calculated.notna().sum()),
                    "n_mismatched_rows": int(mismatch.sum()),
                }
            )

            if mismatch.any():
                mismatch_key = f"{frame_name}_{domain_name.lower()}_mismatches"
                mismatch_tables[mismatch_key] = frame.loc[
                    mismatch,
                    [ID_COL, composite_col] + item_cols,
                ].copy()
                mismatch_tables[mismatch_key]["calculated_composite"] = calculated.loc[mismatch].to_numpy()

    return pd.DataFrame(validation_rows), mismatch_tables


def build_sample_flow(df: pd.DataFrame, wide_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[int]]:
    pre, post = split_pre_post(df)

    matched_ids = set(wide_df[ID_COL].dropna().astype(int).tolist())
    pre_ids = set(pre[ID_COL].dropna().astype(int).tolist())
    post_ids = set(post[ID_COL].dropna().astype(int).tolist())

    pre_only_ids = sorted(pre_ids - matched_ids)
    post_only_ids = sorted(post_ids - matched_ids)

    year_consistency = wide_df[[ID_COL, f"{CLASS_COL}_pre", f"{CLASS_COL}_post"]].copy()
    year_consistency["class_year_matches"] = year_consistency[f"{CLASS_COL}_pre"] == year_consistency[f"{CLASS_COL}_post"]

    flow = pd.DataFrame(
        [
            {"metric": "Unique students in file", "value": int(df[ID_COL].nunique())},
            {"metric": "Pre survey rows", "value": int(len(pre))},
            {"metric": "Post survey rows", "value": int(len(post))},
            {"metric": "Matched students by student_id", "value": int(len(matched_ids))},
            {"metric": "Pre-only student_ids", "value": ", ".join(map(str, pre_only_ids)) if pre_only_ids else ""},
            {"metric": "Post-only student_ids", "value": ", ".join(map(str, post_only_ids)) if post_only_ids else ""},
            {"metric": "Class year mismatches among matched pairs", "value": int((~year_consistency["class_year_matches"]).sum())},
        ]
    )

    return flow, pre_only_ids, post_only_ids


def fmt_n_pct(n: int, denom: int) -> str:
    if denom == 0:
        return "NA"
    return f"{int(n)} ({100.0 * float(n) / float(denom):.1f})"


def build_table_1(df: pd.DataFrame, wide_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pre, post = split_pre_post(df)
    matched_post_ids = set(post[ID_COL].dropna().astype(int).tolist())

    pre = pre.copy()
    pre["paired"] = pre[ID_COL].astype("Int64").isin(matched_post_ids)

    paired = pre.loc[pre["paired"]].copy()
    pre_only = pre.loc[~pre["paired"]].copy()

    n_pre = len(pre)
    n_post = len(post)
    n_paired = wide_df[ID_COL].nunique()
    n_pre_only = len(pre_only)
    n_any = df[ID_COL].nunique()

    col_overall = f"Overall pre cohort (n={n_pre})"
    col_paired = f"Paired analytic cohort (n={n_paired})"

    rows: list[dict[str, str]] = []

    def add_section(label: str) -> None:
        rows.append({"Characteristic": label, col_overall: "", col_paired: ""})

    def add_row(label: str, overall_value: str, paired_value: str) -> None:
        rows.append({"Characteristic": label, col_overall: overall_value, col_paired: paired_value})

    add_row("Completed paired post survey, n (%)", fmt_n_pct(n_paired, n_pre), fmt_n_pct(n_paired, n_paired))
    add_row("Pre-only respondent, n (%)", fmt_n_pct(n_pre_only, n_pre), fmt_n_pct(0, n_paired))

    add_section("Class year, n (%)")
    year_values = sorted([int(x) for x in pre[CLASS_COL].dropna().unique()])
    for year in year_values:
        add_row(
            str(year),
            fmt_n_pct(int((pre[CLASS_COL] == year).sum()), n_pre),
            fmt_n_pct(int((paired[CLASS_COL] == year).sum()), n_paired),
        )

    add_section("Prior paging exposure at baseline, n (%)")
    page_order = ["never", "1-3 pages", "4-10 pages", "more than 10"]
    page_labels = {
        "never": "Never",
        "1-3 pages": "1-3 pages",
        "4-10 pages": "4-10 pages",
        "more than 10": ">10 pages",
    }
    for category in page_order:
        add_row(
            page_labels[category],
            fmt_n_pct(int((pre[PAGES_COL] == category).sum()), n_pre),
            fmt_n_pct(int((paired[PAGES_COL] == category).sum()), n_paired),
        )

    add_row(
        "Missing",
        fmt_n_pct(int(pre[PAGES_COL].isna().sum()), n_pre),
        fmt_n_pct(int(paired[PAGES_COL].isna().sum()), n_paired),
    )

    add_section("Baseline composite score, mean (SD)")
    add_row("Confidence (11-item sum; range 11-110)", format_mean_sd(pre[CONF_COL]), format_mean_sd(paired[CONF_COL]))
    add_row("Anxiety (11-item sum; range 11-110)", format_mean_sd(pre[ANX_COL]), format_mean_sd(paired[ANX_COL]))
    add_row("Communication (8-item sum; range 8-56)", format_mean_sd(pre[COMM_COL]), format_mean_sd(paired[COMM_COL]))

    notes = [
        "Values are n (%) unless otherwise indicated.",
        "Percentages use the column denominator.",
        "Class year is shown from the preintervention record only and should be treated as descriptive.",
        "Confidence and anxiety are 11-item summed scales; communication is an 8-item summed scale.",
        (
            f"The dataset contains {n_any} unique students, {n_pre} preintervention rows, {n_post} "
            f"postintervention rows, {n_paired} matched pre/post pairs, and {n_pre_only} pre-only records."
        ),
    ]

    return pd.DataFrame(rows), notes


def paired_summary(pre_series: pd.Series, post_series: pd.Series) -> dict[str, float]:
    mask = pre_series.notna() & post_series.notna()
    x = pre_series.loc[mask].astype(float).to_numpy()
    y = post_series.loc[mask].astype(float).to_numpy()

    n = len(x)
    if n < 2:
        raise ValueError("Each outcome requires at least 2 complete paired observations.")

    diff = y - x
    pre_mean = float(np.mean(x))
    pre_sd = float(np.std(x, ddof=1))
    post_mean = float(np.mean(y))
    post_sd = float(np.std(y, ddof=1))
    diff_mean = float(np.mean(diff))
    diff_sd = float(np.std(diff, ddof=1))

    se_diff = diff_sd / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = diff_mean - t_crit * se_diff
    ci_high = diff_mean + t_crit * se_diff

    t_result = stats.ttest_rel(y, x, nan_policy="omit")

    try:
        w_result = stats.wilcoxon(
            y,
            x,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            mode="auto",
        )
        wilcoxon_stat = float(w_result.statistic)
        wilcoxon_p = float(w_result.pvalue)
    except ValueError:
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan

    cohens_dz = diff_mean / diff_sd if diff_sd != 0 else np.nan

    return {
        "n": n,
        "pre_mean": pre_mean,
        "pre_sd": pre_sd,
        "post_mean": post_mean,
        "post_sd": post_sd,
        "diff_mean": diff_mean,
        "diff_sd": diff_sd,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_stat": float(t_result.statistic),
        "df": int(n - 1),
        "p_t": float(t_result.pvalue),
        "wilcoxon_stat": wilcoxon_stat,
        "p_w": wilcoxon_p,
        "cohens_dz": float(cohens_dz) if not pd.isna(cohens_dz) else np.nan,
    }


def build_table_2(wide_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome_specs = [
        ("Confidence", f"{CONF_COL}_pre", f"{CONF_COL}_post"),
        ("Anxiety", f"{ANX_COL}_pre", f"{ANX_COL}_post"),
        ("Communication", f"{COMM_COL}_pre", f"{COMM_COL}_post"),
    ]

    raw_rows: list[dict[str, object]] = []
    pretty_rows: list[dict[str, object]] = []

    for label, pre_col, post_col in outcome_specs:
        stats_dict = paired_summary(wide_df[pre_col], wide_df[post_col])
        raw_rows.append(
            {
                "outcome": label,
                "n_pairs": stats_dict["n"],
                "pre_mean": stats_dict["pre_mean"],
                "pre_sd": stats_dict["pre_sd"],
                "post_mean": stats_dict["post_mean"],
                "post_sd": stats_dict["post_sd"],
                "mean_change": stats_dict["diff_mean"],
                "change_sd": stats_dict["diff_sd"],
                "ci_low": stats_dict["ci_low"],
                "ci_high": stats_dict["ci_high"],
                "t_statistic": stats_dict["t_stat"],
                "df": stats_dict["df"],
                "p_value": stats_dict["p_t"],
                "wilcoxon_statistic": stats_dict["wilcoxon_stat"],
                "wilcoxon_p_value": stats_dict["p_w"],
                "cohens_dz": stats_dict["cohens_dz"],
            }
        )
        pretty_rows.append(
            {
                "Outcome": label,
                "Paired n": stats_dict["n"],
                "Preintervention mean (SD)": f"{stats_dict['pre_mean']:.1f} ({stats_dict['pre_sd']:.1f})",
                "Postintervention mean (SD)": f"{stats_dict['post_mean']:.1f} ({stats_dict['post_sd']:.1f})",
                "Mean paired change": f"{stats_dict['diff_mean']:.1f}",
                "95% CI": format_ci(stats_dict["ci_low"], stats_dict["ci_high"]),
                "Paired t-test p value": format_p_value(stats_dict["p_t"]),
                "Wilcoxon p value": format_p_value(stats_dict["p_w"]),
                "Cohen dz": f"{stats_dict['cohens_dz']:.2f}" if not pd.isna(stats_dict["cohens_dz"]) else "NA",
            }
        )

    raw_df = pd.DataFrame(raw_rows)
    raw_df["p_value_bh"] = benjamini_hochberg(raw_df["p_value"])
    raw_df["wilcoxon_p_value_bh"] = benjamini_hochberg(raw_df["wilcoxon_p_value"])

    return pd.DataFrame(pretty_rows), raw_df


def paired_item_summary(pre_series: pd.Series, post_series: pd.Series) -> dict[str, float]:
    mask = pre_series.notna() & post_series.notna()
    x = pre_series.loc[mask].astype(float).to_numpy()
    y = post_series.loc[mask].astype(float).to_numpy()

    n = len(x)
    if n < 2:
        raise ValueError("Each item requires at least 2 complete paired observations.")

    diff = y - x
    pre_mean = float(np.mean(x))
    pre_sd = float(np.std(x, ddof=1))
    post_mean = float(np.mean(y))
    post_sd = float(np.std(y, ddof=1))
    diff_mean = float(np.mean(diff))
    diff_sd = float(np.std(diff, ddof=1))

    se = diff_sd / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = diff_mean - t_crit * se
    ci_high = diff_mean + t_crit * se

    t_res = stats.ttest_rel(y, x, nan_policy="omit")

    return {
        "n": n,
        "pre_mean": pre_mean,
        "pre_sd": pre_sd,
        "post_mean": post_mean,
        "post_sd": post_sd,
        "diff_mean": diff_mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_t": float(t_res.pvalue),
    }


def build_supplementary_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for domain_name in DOMAIN_ORDER:
        domain_info = DOMAIN_ITEMS[domain_name]
        for item_order, (col, label) in enumerate(domain_info["items"], start=1):
            stats_dict = paired_item_summary(wide_df[f"{col}_pre"], wide_df[f"{col}_post"])
            rows.append(
                {
                    "Domain": domain_name,
                    "Item order": item_order,
                    "Item": label,
                    "Scale range": domain_info["scale_range"],
                    "Paired n": stats_dict["n"],
                    "Preintervention mean (SD)": f"{stats_dict['pre_mean']:.1f} ({stats_dict['pre_sd']:.1f})",
                    "Postintervention mean (SD)": f"{stats_dict['post_mean']:.1f} ({stats_dict['post_sd']:.1f})",
                    "Mean change": round(stats_dict["diff_mean"], 1),
                    "95% CI": format_ci(stats_dict["ci_low"], stats_dict["ci_high"]),
                    "Paired t-test p value": stats_dict["p_t"],
                    "_change_num": stats_dict["diff_mean"],
                    "_ci_low_num": stats_dict["ci_low"],
                    "_ci_high_num": stats_dict["ci_high"],
                }
            )

    out = pd.DataFrame(rows)
    out["BH-adjusted q value"] = benjamini_hochberg(out["Paired t-test p value"])
    out["Paired t-test p value"] = out["Paired t-test p value"].apply(format_p_value)
    out["BH-adjusted q value"] = out["BH-adjusted q value"].apply(format_p_value)
    return out


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 observations to compute a confidence interval.")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - tcrit * se, mean + tcrit * se


def apply_global_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_header(ax: plt.Axes, panel_label: str, panel_title: str) -> None:
    ax.text(
        0.00,
        1.03,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.08,
        1.03,
        panel_title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
    )


def format_axes(ax: plt.Axes, y_label: str, y_max: float, y_ticks: list[float]) -> None:
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, y_max)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_yticks(y_ticks)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#e7ebf0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", width=1.0, length=5)


def plot_main_panel(
    ax: plt.Axes,
    wide_df: pd.DataFrame,
    pre_col: str,
    post_col: str,
    panel_label: str,
    panel_title: str,
    y_label: str,
    y_max: float,
    y_ticks: list[float],
    seed: int,
    p_label: str,
) -> dict[str, object]:
    subset = wide_df[[ID_COL, pre_col, post_col]].dropna().copy()
    pre = subset[pre_col].to_numpy(dtype=float)
    post = subset[post_col].to_numpy(dtype=float)
    n = len(subset)

    rng = np.random.default_rng(seed)
    x_pre = np.zeros(n) + rng.normal(0, 0.012, size=n)
    x_post = np.ones(n) + rng.normal(0, 0.012, size=n)

    raw_line_color = "#d5dbe2"
    raw_point_color = "#98a2ad"
    summary_color = "#1f4e79"
    label_edge_color = "#d9dee5"

    for i in range(n):
        ax.plot(
            [x_pre[i], x_post[i]],
            [pre[i], post[i]],
            color=raw_line_color,
            linewidth=0.7,
            alpha=0.24,
            zorder=1,
        )

    ax.scatter(x_pre, pre, s=25, color=raw_point_color, alpha=0.9, linewidths=0, zorder=2)
    ax.scatter(x_post, post, s=25, color=raw_point_color, alpha=0.9, linewidths=0, zorder=2)

    pre_mean, pre_low, pre_high = mean_ci(pre)
    post_mean, post_low, post_high = mean_ci(post)

    ax.errorbar(
        [0, 1],
        [pre_mean, post_mean],
        yerr=[
            [pre_mean - pre_low, post_mean - post_low],
            [pre_high - pre_mean, post_high - post_mean],
        ],
        fmt="o-",
        color=summary_color,
        ecolor=summary_color,
        linewidth=2.6,
        markersize=8,
        markerfacecolor=summary_color,
        markeredgecolor=summary_color,
        capsize=4,
        zorder=4,
    )

    label_box = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": label_edge_color,
        "linewidth": 0.8,
        "alpha": 1.0,
    }

    ax.text(
        -0.22,
        pre_mean,
        f"{pre_mean:.1f}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        bbox=label_box,
        zorder=5,
        clip_on=True,
    )
    ax.text(
        1.22,
        post_mean,
        f"{post_mean:.1f}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        bbox=label_box,
        zorder=5,
        clip_on=True,
    )
    ax.text(
        0.24,
        0.98,
        f"n={n}, {p_label}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )

    add_panel_header(ax, panel_label, panel_title)
    format_axes(ax, y_label, y_max, y_ticks)

    return {
        "panel": panel_label.strip(),
        "outcome": panel_title.strip(),
        "n_pairs": n,
        "pre_mean": pre_mean,
        "pre_ci_low": pre_low,
        "pre_ci_high": pre_high,
        "post_mean": post_mean,
        "post_ci_low": post_low,
        "post_ci_high": post_high,
    }


def save_figure(fig: plt.Figure, outdir: Path, stem: str, dpi: int) -> list[Path]:
    output_paths = [
        outdir / f"{stem}.png",
        outdir / f"{stem}.pdf",
        outdir / f"{stem}.svg",
        outdir / f"{stem}.tiff",
    ]
    for path in output_paths:
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            edgecolor="none",
        )
    return output_paths


def build_main_figure(wide_df: pd.DataFrame, output_dir: Path, dpi: int) -> tuple[list[Path], pd.DataFrame]:
    apply_global_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2))
    plt.subplots_adjust(left=0.05, right=0.995, bottom=0.15, top=0.88, wspace=0.30)

    summary_rows = []

    summary_rows.append(
        plot_main_panel(
            ax=axes[0],
            wide_df=wide_df,
            pre_col=f"{CONF_COL}_pre",
            post_col=f"{CONF_COL}_post",
            panel_label="A",
            panel_title="Confidence increased",
            y_label="Confidence score (0-110)",
            y_max=110,
            y_ticks=[0, 20, 40, 60, 80, 100],
            seed=101,
            p_label="p<0.001",
        )
    )

    summary_rows.append(
        plot_main_panel(
            ax=axes[1],
            wide_df=wide_df,
            pre_col=f"{ANX_COL}_pre",
            post_col=f"{ANX_COL}_post",
            panel_label="B",
            panel_title="Anxiety decreased",
            y_label="Anxiety score (0-110)",
            y_max=110,
            y_ticks=[0, 20, 40, 60, 80, 100],
            seed=202,
            p_label="p<0.001",
        )
    )

    summary_rows.append(
        plot_main_panel(
            ax=axes[2],
            wide_df=wide_df,
            pre_col=f"{COMM_COL}_pre",
            post_col=f"{COMM_COL}_post",
            panel_label="C",
            panel_title="Communication improved",
            y_label="Communication score (0-56)",
            y_max=56,
            y_ticks=[0, 10, 20, 30, 40, 50],
            seed=303,
            p_label="p<0.001",
        )
    )

    paths = save_figure(fig, output_dir, "figure_1_composite_pre_post", dpi)
    plt.close(fig)
    return paths, pd.DataFrame(summary_rows)


def build_supplementary_figure(table_df: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    plot_df = table_df.copy()
    plot_df["Domain"] = pd.Categorical(plot_df["Domain"], categories=DOMAIN_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["Domain", "Item order"]).reset_index(drop=True)

    parts = []
    domain_meta: dict[str, dict[str, float]] = {}
    y_cursor = 0.0

    for domain in DOMAIN_ORDER:
        sub = plot_df.loc[plot_df["Domain"] == domain].copy().reset_index(drop=True)
        if sub.empty:
            continue
        n_rows = len(sub)
        sub["y"] = np.arange(y_cursor, y_cursor + n_rows, 1.0)
        domain_meta[domain] = {
            "start": float(sub["y"].min()),
            "end": float(sub["y"].max()),
            "mid": float((sub["y"].min() + sub["y"].max()) / 2.0),
        }
        parts.append(sub)
        y_cursor = float(sub["y"].max()) + 2.0

    plot_df = pd.concat(parts, ignore_index=True)

    xmin = float(plot_df["_ci_low_num"].min())
    xmax = float(plot_df["_ci_high_num"].max())
    xpad = 0.8
    lower_err = plot_df["_change_num"] - plot_df["_ci_low_num"]
    upper_err = plot_df["_ci_high_num"] - plot_df["_change_num"]

    fig_height = max(9.5, 0.38 * len(plot_df) + 2.3)
    fig, ax = plt.subplots(figsize=(10.8, fig_height))

    ax.errorbar(
        x=plot_df["_change_num"],
        y=plot_df["y"],
        xerr=[lower_err, upper_err],
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        markersize=5.5,
        markerfacecolor="black",
        markeredgecolor="black",
    )
    ax.axvline(0, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(plot_df["Item"], fontsize=11)

    domain_label_offsets = {"Confidence": 6.0, "Anxiety": 6.0, "Communication": 4.5}
    for domain in DOMAIN_ORDER:
        if domain not in domain_meta:
            continue
        meta = domain_meta[domain]
        ax.text(
            -0.29,
            meta["mid"] - domain_label_offsets.get(domain, 1.2),
            domain,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )
        ax.hlines(
            y=meta["end"] + 0.8,
            xmin=xmin - xpad,
            xmax=xmax + xpad,
            color="0.85",
            linewidth=0.9,
        )

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_xlabel("Mean paired change (postintervention minus preintervention)", fontsize=13)
    ax.set_title("Supplementary Figure S1. Item-level paired changes after the mock paging exercise", fontsize=15, pad=10)
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.subplots_adjust(left=0.42, right=0.98, top=0.94, bottom=0.07)

    paths = save_figure(fig, output_dir, "supplementary_figure_s1_item_level_mean_changes", dpi)
    plt.close(fig)
    return paths


def build_results_paragraph(table_2_raw: pd.DataFrame) -> str:
    conf = table_2_raw.loc[table_2_raw["outcome"] == "Confidence"].iloc[0]
    anx = table_2_raw.loc[table_2_raw["outcome"] == "Anxiety"].iloc[0]
    comm = table_2_raw.loc[table_2_raw["outcome"] == "Communication"].iloc[0]

    return (
        f"In paired analyses, confidence increased from {conf['pre_mean']:.1f} ({conf['pre_sd']:.1f}) "
        f"to {conf['post_mean']:.1f} ({conf['post_sd']:.1f}) "
        f"(mean change {conf['mean_change']:.1f}, 95% CI {conf['ci_low']:.1f} to {conf['ci_high']:.1f}; "
        f"p={format_p_value(conf['p_value'])}). "
        f"Anxiety decreased from {anx['pre_mean']:.1f} ({anx['pre_sd']:.1f}) to {anx['post_mean']:.1f} "
        f"({anx['post_sd']:.1f}) "
        f"(mean change {anx['mean_change']:.1f}, 95% CI {anx['ci_low']:.1f} to {anx['ci_high']:.1f}; "
        f"p={format_p_value(anx['p_value'])}). "
        f"Communication increased from {comm['pre_mean']:.1f} ({comm['pre_sd']:.1f}) to {comm['post_mean']:.1f} "
        f"({comm['post_sd']:.1f}) "
        f"(mean change {comm['mean_change']:.1f}, 95% CI {comm['ci_low']:.1f} to {comm['ci_high']:.1f}; "
        f"p={format_p_value(comm['p_value'])})."
    )


def write_text_file(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_csv = resolve_input_path(args.input)
    output_dir = resolve_output_dir(args.output_dir, input_csv)

    df = load_data(input_csv)
    wide_df = build_wide_dataset(df)

    validation_df, mismatch_tables = validate_composites(df)
    sample_flow_df, pre_only_ids, post_only_ids = build_sample_flow(df, wide_df)
    year_consistency_df = wide_df[[ID_COL, f"{CLASS_COL}_pre", f"{CLASS_COL}_post"]].copy()
    year_consistency_df["class_year_matches"] = year_consistency_df[f"{CLASS_COL}_pre"] == year_consistency_df[f"{CLASS_COL}_post"]

    table_1_df, table_1_notes = build_table_1(df, wide_df)
    table_2_pretty_df, table_2_raw_df = build_table_2(wide_df)
    supp_table_df = build_supplementary_table(wide_df)
    figure_1_paths, figure_1_source_df = build_main_figure(wide_df, output_dir, args.dpi)
    supp_figure_paths = build_supplementary_figure(supp_table_df, output_dir, args.dpi)

    table_1_df.to_csv(output_dir / "table_1_baseline_characteristics.csv", index=False)
    write_text_file(
        output_dir / "table_1_baseline_characteristics.md",
        "# Table 1. Baseline characteristics of the mock paging cohort\n\n"
        + dataframe_to_markdown(table_1_df)
        + "\n\nNotes:\n"
        + "\n".join(f"{i}. {note}" for i, note in enumerate(table_1_notes, start=1)),
    )

    table_2_pretty_df.to_csv(output_dir / "table_2_paired_outcomes.csv", index=False)
    write_text_file(
        output_dir / "table_2_paired_outcomes.md",
        "# Table 2. Mock paging increased confidence and communication and reduced anxiety in paired analyses\n\n"
        + dataframe_to_markdown(table_2_pretty_df)
        + "\n\nNote. Values are mean (SD) unless otherwise indicated. Mean paired change was calculated as "
        + "postintervention minus preintervention. P values are from 2-sided paired t tests. "
        + "Wilcoxon signed-rank tests were performed as sensitivity analyses.",
    )

    table_2_raw_df.to_csv(output_dir / "table_2_paired_outcomes_raw_stats.csv", index=False)

    supp_export = supp_table_df.drop(columns=["_change_num", "_ci_low_num", "_ci_high_num"]).copy()
    supp_export.to_csv(output_dir / "supplementary_table_s1_item_level_paired_changes.csv", index=False)

    validation_df.to_csv(output_dir / "composite_validation_summary.csv", index=False)
    sample_flow_df.to_csv(output_dir / "sample_flow.csv", index=False)
    year_consistency_df.to_csv(output_dir / "class_year_consistency_check.csv", index=False)
    figure_1_source_df.to_csv(output_dir / "figure_1_source_data.csv", index=False)

    for name, mismatch_df in mismatch_tables.items():
        mismatch_df.to_csv(output_dir / f"{name}.csv", index=False)

    results_paragraph = build_results_paragraph(table_2_raw_df)
    write_text_file(output_dir / "draft_results_paragraph.txt", results_paragraph)

    summary_lines = [
        "Mock paging manuscript analysis completed.",
        f"Input file: {input_csv}",
        f"Output directory: {output_dir}",
        "",
        f"Unique students: {df[ID_COL].nunique()}",
        f"Pre rows: {(df['timepoint'] == 'pre').sum()}",
        f"Post rows: {(df['timepoint'] == 'post').sum()}",
        f"Matched student pairs: {wide_df[ID_COL].nunique()}",
        f"Pre-only IDs: {pre_only_ids if pre_only_ids else 'None'}",
        f"Post-only IDs: {post_only_ids if post_only_ids else 'None'}",
        "",
        "Primary paired composite results:",
        table_2_raw_df[
            ["outcome", "n_pairs", "pre_mean", "post_mean", "mean_change", "ci_low", "ci_high", "p_value", "cohens_dz"]
        ].round(4).to_string(index=False),
        "",
        "Generated files:",
        "  - table_1_baseline_characteristics.csv",
        "  - table_1_baseline_characteristics.md",
        "  - table_2_paired_outcomes.csv",
        "  - table_2_paired_outcomes.md",
        "  - table_2_paired_outcomes_raw_stats.csv",
        "  - supplementary_table_s1_item_level_paired_changes.csv",
        "  - figure_1_composite_pre_post.[png|pdf|svg|tiff]",
        "  - supplementary_figure_s1_item_level_mean_changes.[png|pdf|svg|tiff]",
        "  - composite_validation_summary.csv",
        "  - class_year_consistency_check.csv",
        "  - sample_flow.csv",
        "  - figure_1_source_data.csv",
        "  - draft_results_paragraph.txt",
    ]
    summary_text = "\n".join(summary_lines)
    write_text_file(output_dir / "analysis_console_summary.txt", summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()