# comparative_painting

## Overview

This project runs comparative painting experiments involving various models and raters. It controls experiments and plotting via a unified script and allows flexible configurations for different analyses.

## Running Experiments and Generating Plots

All experiments and plotting are controlled from a single script: `run_all_experiments.py`.

**Important:** Always run the script from inside the `Code` directory.


## Configuration

To control the experiments, edit the **CONFIGURATION** section at the top of the `run_all_experiments.py` script:

- **EXPERIMENT_MODE:** Choose which analysis to run.
  - `'AVERAGE'`: Runs the experiment on the average of all raters.
  - `'WITHIN_RATER'`: Runs a separate experiment for each rater specified in `RATERS_TO_RUN`.
  - `'CROSS_RATER'`: Runs the leave-one-out cross-rater analysis for each rater in `RATERS_TO_RUN`.

- **MODELS_TO_RUN:** Choose which model(s) to execute.
  - `'regression'`: Runs only the baseline regression model.
  - `'pairwise'`: Runs only the pairwise comparison model.
  - `'both'`: Runs both models for a direct comparison.

- **RATERS_TO_RUN:** Specify which raters to process for the `WITHIN_RATER` and `CROSS_RATER` modes.
  - `range(1, 6)`: Processes raters 1 through 5.
  - `[1, 5, 10]`: Processes only raters 1, 5, and 10.

## Results

After the experiments for a given mode/rater are complete, the script will automatically aggregate the results and generate the corresponding plots.

---

*This README provides instructions to run all experiments from one place with flexibility on models and raters, suitable for replicable and controlled analysis.*
