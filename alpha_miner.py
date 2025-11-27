#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import time

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

import csv


def detect_separator(path, default=","):
    try:
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(2048)
            # If file is very small, read full
            if not sample:
                return default
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
    except Exception:
        # Fallback: if semicolons in header, prefer ';'
        try:
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline()
                if ";" in header and header.count(";") > header.count(","):
                    return ";"
        except Exception:
            pass
        return default


from typing import Optional


def load_event_log(
    csv_path,
    case_id="STUDIENR",
    activity="KURSTXT",
    timestamp="SEMESTER_START",
    sep=None,
    sample_cases: Optional[int] = None,
    top_activities: Optional[int] = None,
):
    # Auto-detect separator if not provided
    if sep is None:
        sep = detect_separator(csv_path)

    df = pd.read_csv(csv_path, sep=sep)

    # Convert timestamp column if present
    if timestamp in df.columns:
        df[timestamp] = pd.to_datetime(df[timestamp], errors="coerce")

    # Ensure pm4py timestamp columns are recognized and convert
    df = dataframe_utils.convert_timestamp_columns_in_df(df)

    # Rename to standard pm4py column names
    rename_map = {}
    if case_id in df.columns:
        rename_map[case_id] = "case:concept:name"
    if activity in df.columns:
        rename_map[activity] = "concept:name"
    if timestamp in df.columns:
        rename_map[timestamp] = "time:timestamp"

    df = df.rename(columns=rename_map)

    # Keep only relevant columns (pm4py can work without timestamp but prefer to include if present)
    keep_cols = [c for c in ["case:concept:name", "concept:name", "time:timestamp"] if c in df.columns]
    df = df[keep_cols]

    # Optionally sample cases to reduce runtime
    if "case:concept:name" in df.columns and sample_cases is not None:
        unique_cases = df["case:concept:name"].unique()
        if len(unique_cases) > sample_cases:
            sampled = pd.Series(unique_cases).sample(n=sample_cases, random_state=1).tolist()
            df = df[df["case:concept:name"].isin(sampled)]
            print(f"Sampled {sample_cases} cases (reduced from {len(unique_cases)}).")

    # Optionally keep only top-N frequent activities
    if "concept:name" in df.columns and top_activities is not None:
        top = df["concept:name"].value_counts().nlargest(top_activities).index.tolist()
        before = df["concept:name"].nunique()
        df = df[df["concept:name"].isin(top)]
        after = df["concept:name"].nunique()
        print(f"Filtered activities: kept top {after} of {before} activities by frequency.")

    # Print basic stats to give user feedback and a heuristic about runtime
    if "case:concept:name" in df.columns and "concept:name" in df.columns:
        n_events = len(df)
        n_cases = df["case:concept:name"].nunique()
        n_activities = df["concept:name"].nunique()
        print(f"Events: {n_events}, Cases: {n_cases}, Unique activities: {n_activities}, sep='{sep}'")
        if n_activities > 40:
            print("Warning: Alpha miner runtime grows fast with the number of unique activities (>40).")
            print("Consider reducing activity set, sampling cases, or using Heuristic/Inductive miner for large logs.")

    # Convert pandas DataFrame to pm4py Event Log
    log = log_converter.apply(df)
    return log


def mine_alpha(
    csv_path,
    out_dir="outputs",
    case_id="STUDIENR",
    activity="KURSTXT",
    timestamp="SEMESTER_START",
    sep=None,
    sample_cases: Optional[int] = None,
    top_activities: Optional[int] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading CSV from: {csv_path}")
    log = load_event_log(
        csv_path, case_id, activity, timestamp, sep=sep, sample_cases=sample_cases, top_activities=top_activities
    )

    print("Running Alpha Miner...")
    t0 = time.time()
    net, initial_marking, final_marking = alpha_miner.apply(log)
    t1 = time.time()
    print(f"Alpha miner finished in {t1 - t0:.1f} seconds.")

    # Visualize and save PNG
    print("Visualizing Petri net and saving outputs...")
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    png_path = os.path.join(out_dir, "alpha_petrinet.png")
    pn_visualizer.save(gviz, png_path)

    # Export PNML (if exporter is available)
    pnml_path = os.path.join(out_dir, "alpha_petrinet.pnml")
    if pnml_exporter is not None:
        try:
            pnml_exporter.apply(net, initial_marking, pnml_path)
            print(f"Saved Petri net PNML to: {pnml_path}")
        except Exception as e:
            print(f"Could not export PNML: {e}")
    else:
        print("PNML exporter not available in your pm4py installation. Skipping PNML export.")

    print(f"Saved Petri net PNG to: {png_path}")
    return net, initial_marking, final_marking


def build_argparser():
    p = argparse.ArgumentParser(description="Simple Alpha Miner wrapper using pm4py")
    p.add_argument("--input", "-i", default="DTU_Curricula_Data_Filtered.csv", help="Path to filtered CSV file")
    p.add_argument("--out", "-o", default="outputs", help="Output directory for PNML/PNG")
    p.add_argument("--case", default="STUDIENR", help="Case id column name")
    p.add_argument("--activity", default="KURSTXT", help="Activity column name")
    p.add_argument("--timestamp", default="SEMESTER_START", help="Timestamp column name (optional)")
    p.add_argument("--sep", default=None, help="CSV separator (auto-detected if not set)")
    p.add_argument("--sample", type=int, default=None, help="Sample this many cases (to reduce runtime)")
    p.add_argument("--top-activities", type=int, default=None, help="Keep only top-N frequent activities")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        raise SystemExit(1)

    mine_alpha(
        args.input,
        args.out,
        args.case,
        args.activity,
        args.timestamp,
        sep=args.sep,
        sample_cases=args.sample,
        top_activities=args.top_activities,
    )
