from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def export_petri_net(net, initial_marking, final_marking, output_path: str):
    """Export the Petri net to PNML format."""
    print(f"\nExporting Petri net to {output_path}...")

    pnml_exporter.apply(net, initial_marking, output_path, final_marking=final_marking)

    print(f"Petri net exported successfully!")


def visualize_model(net, initial_marking, final_marking, output_path: str):
    """Visualize the discovered Petri net and save to file."""
    print(f"\nVisualizing model and saving to {output_path}...")

    try:
        gviz = pn_visualizer.apply(
            net,
            initial_marking,
            final_marking,
            parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"},
        )

        pn_visualizer.save(gviz, output_path)
        print(f"Visualization saved successfully!")
    except Exception as e:
        print(f"\nWarning: Could not create visualization.")
        print(f"Error: {e}")
        print(f"\nTo fix: Install Graphviz from https://graphviz.org/download/")
        print(f"Or use: choco install graphviz (if you have Chocolatey)")
