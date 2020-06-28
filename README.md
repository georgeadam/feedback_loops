# ML Feedback Loops
This repository contains the necessary code to run experiments investigating feedback loops that may occur when models are given the ability to influence future data via their predictions.

## Structure

    .
    ├── configs                 # Configuration files written in YAML that specify various experiments parameters such as: models, data, update types, regularization, etc.
    ├── slurm                   # Bash scripts for running experiments on a slurm cluster. These can be ignored if running locally on one's machine
    ├── src                     # Python source code
    │   ├── models              # Functions for building sklearn and PyTorch models
    │   ├── scripts             # Scripts that actually run experiments, and corresponding helper functions
    │   └── utils               # Functions for computing metrics, reweighting samples, and performing updates
    └── ...
    
## Running Experiments

To compare the effect of different update types on MIMIC-IV temporally sorted data using an XGBoost model, and initial
desired FPR of 0.2, and dynamic resetting of threshold each epoch use:

`python scripts/compare_update_types.py data=temporal data.type=mimic_iv_demographic model=xgboost rates.idr=fpr rates.idv=0.2 rates.ddr=fpr`

## Extending Current Framework

### Learning to Defer

This requires modifying `replace_labels()` in `src/utils/update.py` by having clinician trust be dependent on the 
learning to defer framework, rather using blind trust, constant trust, or conditional trust, all of which are currently
implemented.

An easy way to implement this would be to modify the model to output a multifactor vector (i.e. currently the model 
outputs 0 for negative class, 1 for positive class, we would need it to output 2 for **pass**)
then we can simply check for the **pass** prediction and defer to clinician accordingly.
