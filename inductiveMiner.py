#!/usr/bin/env python3
"""
Inductive Miner for DTU Curricula Process Mining

This script applies the Inductive Miner algorithm to discover a process model
from DTU student course completion data.
"""

import argparse
import os

import pandas as pd
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.log.util import dataframe_utils

from utils import export_petri_net, visualize_model

# Constants
INPUT_PATH = "DTU_Curricula_Data_Filtered.csv"
OUTPUT_PETRI_NET_PATH = "outputs/inductive_miner_petri_net.pnml"
OUTPUT_VISUALIZATION_PATH = "outputs/inductive_miner_model.png"


def load_event_log(file_path: str) -> pd.DataFrame:
    """Load and prepare the event log from CSV file."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total events: {len(df)}")
    print(f"Number of students (cases): {df['STUDIENR'].nunique()}")
    print(f"Number of unique courses: {df['KURSKODE'].nunique()}")

    return df


def prepare_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for PM4Py by renaming columns to standard names.
    PM4Py expects: case:concept:name, concept:name, time:timestamp
    """
    print("\nPreparing event log...")

    # Create a copy to avoid modifying the original
    log_df = df.copy()

    # Rename columns to PM4Py standard format
    log_df = log_df.rename(
        columns={
            "STUDIENR": "case:concept:name",  # Case ID (student)
            "KURSTXT": "concept:name",  # Activity name (course)
            "SEMESTER_END": "time:timestamp",  # Timestamp
        }
    )

    # Convert timestamp to datetime
    log_df["time:timestamp"] = pd.to_datetime(log_df["time:timestamp"])

    # Add additional attributes that might be useful for filtering/analysis
    log_df["ECTS"] = df["ECTS"]
    log_df["BEDOMMELSE"] = df["BEDOMMELSE"]
    log_df["ATTEMPT"] = df["ATTEMPT"]

    # Sort by case and timestamp
    log_df = log_df.sort_values(["case:concept:name", "time:timestamp"])

    print(f"Event log prepared with {len(log_df)} events")

    return log_df


def discover_model_inductive(log_df: pd.DataFrame, noise_threshold: float = 0.0):
    """
    Apply Inductive Miner algorithm to discover a process model.

    Parameters:
    -----------
    log_df : pd.DataFrame
        Event log in PM4Py format
    noise_threshold : float
        Noise threshold for Inductive Miner (0.0 = no noise filtering,
        higher values = more noise filtering, typically 0.0 - 0.5)

    Returns:
    --------
    tuple : (net, initial_marking, final_marking)
        Petri net model and markings
    """
    print(f"\nApplying Inductive Miner (noise_threshold={noise_threshold})...")

    # Convert dataframe to event log
    log_df = dataframe_utils.convert_timestamp_columns_in_df(log_df)
    event_log = log_converter.apply(log_df)

    # Apply Inductive Miner - returns a process tree
    process_tree = inductive_miner.apply(
        event_log,
        variant=inductive_miner.Variants.IMf,  # IMf is more flexible with noise
        parameters={"noise_threshold": noise_threshold},
    )

    print("Process tree discovered successfully!")

    # Convert process tree to Petri net
    net, initial_marking, final_marking = pt_converter.apply(process_tree)

    print("Converted to Petri net:")
    print(f"Number of places: {len(net.places)}")
    print(f"Number of transitions: {len(net.transitions)}")

    return net, initial_marking, final_marking


def build_argparser():
    p = argparse.ArgumentParser(description="Inductive Miner wrapper using pm4py")
    p.add_argument("--input", "-i", default=INPUT_PATH, help="Path to filtered CSV file")
    p.add_argument("--out", "-o", default="outputs", help="Output directory for PNML/PNG")
    p.add_argument("--noise", type=float, default=0.0, help="Noise threshold for Inductive Miner")
    p.add_argument("--sample", type=int, default=None, help="Sample this many cases (to reduce runtime)")
    p.add_argument("--top-activities", type=int, default=None, help="Keep only top-N frequent activities")
    return p


def main():
    """Main execution function."""
    print("=" * 60)
    print("DTU Curricula - Inductive Miner Process Discovery")
    print("=" * 60)
    parser = build_argparser()
    args = parser.parse_args()

    # Ensure outputs directory exists
    os.makedirs(args.out, exist_ok=True)

    # Load data
    df = load_event_log(args.input)

    # Optionally sample and filter activities before preparing
    if args.sample is not None:
        unique_cases = df["STUDIENR"].unique()
        if len(unique_cases) > args.sample:
            sampled = pd.Series(unique_cases).sample(n=args.sample, random_state=1).tolist()
            df = df[df["STUDIENR"].isin(sampled)]
            print(f"Sampled {args.sample} cases (reduced from {len(unique_cases)}).")

    if args.top_activities is not None:
        before = df["KURSTXT"].nunique()
        top = df["KURSTXT"].value_counts().nlargest(args.top_activities).index.tolist()
        df = df[df["KURSTXT"].isin(top)]
        after = df["KURSTXT"].nunique()
        print(f"Filtered activities: kept top {after} of {before} activities by frequency.")

    # Prepare event log
    log_df = prepare_event_log(df)

    # Discover model using Inductive Miner
    # Adjust noise_threshold as needed (0.0 = strict, 0.2 = moderate filtering)
    net, initial_marking, final_marking = discover_model_inductive(log_df, noise_threshold=args.noise)

    # Warn about large models
    if len(net.transitions) > 200:
        print(f"\nWarning: The model is very large ({len(net.transitions)} transitions).")
        print("Visualization may take several minutes. Please be patient...")

    # Visualize and save the model
    vis_path = os.path.join(args.out, os.path.basename(OUTPUT_VISUALIZATION_PATH))
    pnml_path = os.path.join(args.out, os.path.basename(OUTPUT_PETRI_NET_PATH))

    visualize_model(net, initial_marking, final_marking, vis_path)

    # Export Petri net (this is fast even for large models)
    export_petri_net(net, initial_marking, final_marking, pnml_path)

    print("\n" + "=" * 60)
    print("Process discovery completed successfully!")
    print(f"Visualization: {vis_path}")
    print(f"Petri Net PNML: {pnml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
