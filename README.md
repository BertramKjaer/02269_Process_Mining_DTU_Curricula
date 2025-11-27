# DTU Curricula Process Mining Project

This repository contains a reproducible workflow for process mining on DTU curricula data using Python and pm4py. It includes scripts for data cleaning, process discovery (Alpha, Heuristics, Inductive Miners), model evaluation, and output management.

## Project Structure

```
├── alpha.py                # Alpha Miner script
├── heuristicsMiner.py      # Heuristics Miner script
├── inductiveMiner.py       # Inductive Miner script
├── prepare-data.py         # Data cleaning and filtering
├── evaluate_models.py      # Model evaluation and metrics
├── DTU_Curricula_Data.csv  # Raw event log (input)
├── requirements.txt        # Python dependencies
├── outputs/                # All generated outputs (PNML, visualizations, metrics)
└── __pycache__/            # Python cache files
```

## Setup Instructions

1. **Install Python 3.8+** (recommended: use a virtual environment)
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Prepare the data:**
   ```powershell
   python prepare-data.py
   ```
   This creates `DTU_Curricula_Data_Filtered.csv` for mining.

## Running the Miners

- **Alpha Miner:**
  ```powershell
  python alpha.py
  ```
- **Heuristics Miner:**
  ```powershell
  python heuristicsMiner.py
  ```
- **Inductive Miner:**
  ```powershell
  python inductiveMiner.py
  ```

All outputs (PNML files, visualizations) are saved in the `outputs/` folder.

## Model Evaluation

To compute metrics (replay fitness, precision, complexity) for all discovered models:
```powershell
python evaluate_models.py --sample 10 --top-activities 10 --noise 0.4 --heur-dependency 0.95
```
- Results are saved in `outputs/model_metrics.csv` and `outputs/model_summary.md`.
- You can adjust parameters for sampling, activity filtering, and miner thresholds.

## Output Management

- All generated files are stored in `outputs/`.
- The `.gitignore` excludes outputs and filtered data to keep the repository clean.

## Troubleshooting

- If metrics are missing, ensure initial/final markings are set (handled automatically in scripts).
- For pm4py errors, check your installed version matches `requirements.txt`.
- For large logs, use sampling and activity filtering to reduce runtime.

## License

This project is for educational use at DTU. See individual scripts for author credits.
