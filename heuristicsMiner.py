#!/usr/bin/env python3
"""
Heuristics Miner for DTU Curricula Process Mining

This script applies the Heuristics Miner algorithm to discover a process model
from DTU student course completion data. Heuristics Miner is particularly good
at handling noise and finding the main process flow.
"""

import pandas as pd
import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

# Constants
INPUT_PATH = "DTU_Curricula_Data_Filtered.csv"
OUTPUT_HEURISTICS_NET_PATH = "heuristics_miner_model.png"
OUTPUT_PETRI_NET_PATH = "heuristics_miner_petri_net.png"

def load_event_log(file_path: str) -> pd.DataFrame:
    """Load and prepare the event log from CSV file."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Display basic statistics
    print(f"\nDataset Statistics:")
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
    log_df = log_df.rename(columns={
        'STUDIENR': 'case:concept:name',      # Case ID (student)
        'KURSTXT': 'concept:name',             # Activity name (course)
        'SEMESTER_END': 'time:timestamp'       # Timestamp
    })
    
    # Convert timestamp to datetime
    log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'])
    
    # Add additional attributes that might be useful for filtering/analysis
    log_df['ECTS'] = df['ECTS']
    log_df['BEDOMMELSE'] = df['BEDOMMELSE']
    log_df['ATTEMPT'] = df['ATTEMPT']
    
    # Sort by case and timestamp
    log_df = log_df.sort_values(['case:concept:name', 'time:timestamp'])
    
    print(f"Event log prepared with {len(log_df)} events")
    
    return log_df

def discover_model_heuristics(log_df: pd.DataFrame, 
                               dependency_threshold: float = 0.5,
                               and_threshold: float = 0.65,
                               loop_two_threshold: float = 0.5):
    """
    Apply Heuristics Miner algorithm to discover a process model.
    
    Parameters:
    -----------
    log_df : pd.DataFrame
        Event log in PM4Py format
    dependency_threshold : float
        Threshold for dependency between activities (0.0-1.0)
        Higher = stricter, shows only strong dependencies
    and_threshold : float
        Threshold for detecting AND splits/joins (0.0-1.0)
    loop_two_threshold : float
        Threshold for detecting length-two loops (0.0-1.0)
    
    Returns:
    --------
    heuristics_net : HeuristicsNet
        Discovered heuristics net model
    """
    print(f"\nApplying Heuristics Miner...")
    print(f"  Dependency threshold: {dependency_threshold}")
    print(f"  AND threshold: {and_threshold}")
    print(f"  Loop-two threshold: {loop_two_threshold}")
    
    # Convert dataframe to event log
    log_df = dataframe_utils.convert_timestamp_columns_in_df(log_df)
    event_log = log_converter.apply(log_df)
    
    # Apply Heuristics Miner
    result = heuristics_miner.apply(
        event_log,
        parameters={
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: dependency_threshold,
            heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: and_threshold,
            heuristics_miner.Variants.CLASSIC.value.Parameters.LOOP_LENGTH_TWO_THRESH: loop_two_threshold
        }
    )
    
    # The result might be a tuple (heuristics_net, dependency_matrix, dfg)
    # or just the heuristics_net depending on the PM4Py version
    if isinstance(result, tuple):
        heuristics_net = result[0]
    else:
        heuristics_net = result
    
    print(f"Heuristics net discovered successfully!")
    
    return heuristics_net

def visualize_heuristics_net(heuristics_net, output_path: str):
    """Visualize the heuristics net and save to file."""
    print(f"\nVisualizing heuristics net and saving to {output_path}...")
    
    try:
        gviz = hn_visualizer.apply(
            heuristics_net,
            parameters={hn_visualizer.Variants.PYDOTPLUS.value.Parameters.FORMAT: "png"}
        )
        
        hn_visualizer.save(gviz, output_path)
        print(f"Heuristics net visualization saved successfully!")
    except Exception as e:
        print(f"\nWarning: Could not create heuristics net visualization.")
        print(f"Error: {e}")
        print(f"\nTo fix: Install Graphviz from https://graphviz.org/download/")
        print(f"Or use: choco install graphviz (if you have Chocolatey)")

def convert_and_visualize_petri_net(heuristics_net, output_path: str):
    """Convert heuristics net to Petri net and visualize."""
    print(f"\nConverting to Petri net and saving to {output_path}...")
    
    # Convert heuristics net to Petri net
    net, initial_marking, final_marking = hn_converter.apply(heuristics_net)
    
    print(f"Petri net conversion completed:")
    print(f"  Places: {len(net.places)}")
    print(f"  Transitions: {len(net.transitions)}")
    
    # Visualize Petri net
    try:
        gviz = pn_visualizer.apply(
            net,
            initial_marking,
            final_marking,
            parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"}
        )
        
        pn_visualizer.save(gviz, output_path)
        print(f"Petri net visualization saved successfully!")
    except Exception as e:
        print(f"\nWarning: Could not create Petri net visualization.")
        print(f"Error: {e}")
        print(f"\nTo fix: Install Graphviz from https://graphviz.org/download/")
        print(f"Or use: choco install graphviz (if you have Chocolatey)")
    
    return net, initial_marking, final_marking

def visualize_model(net, output_path: str):
    """Visualize a Petri net that was returned directly."""
    print(f"\nVisualizing Petri net and saving to {output_path}...")
    
    try:
        # Create simple visualization without markings since we don't have them
        gviz = pn_visualizer.apply(
            net,
            parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"}
        )
        
        pn_visualizer.save(gviz, output_path)
        print(f"Petri net visualization saved successfully!")
    except Exception as e:
        print(f"\nWarning: Could not create Petri net visualization.")
        print(f"Error: {e}")
        print(f"\nTo fix: Install Graphviz from https://graphviz.org/download/")
        print(f"Or use: choco install graphviz (if you have Chocolatey)")

def print_statistics(model):
    """Print statistics about the discovered model."""
    print("\n" + "=" * 60)
    print("Model Statistics:")
    print("=" * 60)
    
    # Check if it's a HeuristicsNet or PetriNet
    if hasattr(model, 'nodes'):
        # It's a HeuristicsNet
        activities = len(model.nodes)
        print(f"Number of activities: {activities}")
        
        # Count dependencies
        total_deps = 0
        for node in model.nodes:
            total_deps += len(node.output_connections)
        print(f"Number of dependencies: {total_deps}")
        
        # Display top activities by frequency
        print("\nTop 10 most frequent activities:")
        activity_freq = sorted(
            [(node.node_name, node.node_occ) for node in model.nodes],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (activity, freq) in enumerate(activity_freq, 1):
            print(f"  {i}. {activity}: {freq} occurrences")
    else:
        # It's a PetriNet (returned directly by newer PM4Py versions)
        print(f"Number of places: {len(model.places)}")
        print(f"Number of transitions: {len(model.transitions)}")
        print("\nNote: Newer PM4Py versions return a Petri net directly from Heuristics Miner.")

def main():
    """Main execution function."""
    print("=" * 60)
    print("DTU Curricula - Heuristics Miner Process Discovery")
    print("=" * 60)
    
    # Load data
    df = load_event_log(INPUT_PATH)
    
    # Prepare event log
    log_df = prepare_event_log(df)
    
    # Discover model using Heuristics Miner
    # Adjust thresholds as needed:
    # - dependency_threshold: 0.95 is default (higher = fewer edges, stricter)
    # - and_threshold: 0.65 is default (affects parallel behavior detection)
    # - loop_two_threshold: 0.5 is default (affects loop detection)
    model = discover_model_heuristics(
        log_df,
        dependency_threshold=0.8,
        and_threshold=0.65,
        loop_two_threshold=0.5
    )
    
    # Print statistics
    print_statistics(model)
    
    # Check if we got a HeuristicsNet or PetriNet
    if hasattr(model, 'nodes'):
        # It's a HeuristicsNet - visualize and convert
        visualize_heuristics_net(model, OUTPUT_HEURISTICS_NET_PATH)
        net, initial_marking, final_marking = convert_and_visualize_petri_net(
            model, 
            OUTPUT_PETRI_NET_PATH
        )
    else:
        # It's already a PetriNet - just visualize it
        print("\nNote: PM4Py returned a Petri net directly from Heuristics Miner.")
        
        # Warn about large models
        if len(model.transitions) > 200:
            print(f"\nWarning: The model is very large ({len(model.transitions)} transitions).")
            print("Visualization may take several minutes. Please be patient...")
        
        visualize_model(model, OUTPUT_PETRI_NET_PATH)
    
    print("\n" + "=" * 60)
    print("Process discovery completed successfully!")
    print(f"Model visualization: {OUTPUT_PETRI_NET_PATH}")
    print("=" * 60)
    
    print("\nTip: Adjust the threshold parameters in the script to:")
    print("  - Increase dependency_threshold to simplify the model")
    print("  - Decrease dependency_threshold to show more relationships")
    print("  - Modify and_threshold to better detect parallel activities")

if __name__ == "__main__":
    main()
