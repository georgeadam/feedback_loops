# ML Feedback Loops
This repository contains the necessary code to run experiments investigating feedback loops that may occur when models are given the ability to influence future data via their predictions.

## Structure

    .
    ├── configs                 # Configuration files written in YAML that specify various experiments parameters such as: models, data, update types, regularization, etc.
    ├── slurm                   # Bash scripts for running experiments on a slurm cluster. These can be ignored if running locally on one's machine
    ├── src                     # Python source code
    │   ├──                     # 
    │   ├──                     # 
    │   └──                     # 
    └── ...

- `configs` contains configuration files written in YAML that specify various experiments parameters such as: models, data, update types, regularization, etc.
- `slurm` contains bash scripts for running experiments on a slurm cluster. These can be ignored if running locally on one's machine
- `src` contains all python source code 
