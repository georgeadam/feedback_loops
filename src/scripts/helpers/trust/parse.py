from src.utils.parse import percentage, str2bool
from argparse import ArgumentParser

parent_parser = ArgumentParser()
parent_parser.add_argument("--data-type", default="mimic_iv", choices=["sklearn", "mimic_iii", "mimic_iv", "support2", "gaussian",
                                                                "mimic_iv_12h", "mimic_iv_24h"], type=str)
parent_parser.add_argument("--seeds", default=1, type=int)
parent_parser.add_argument("--model", default="lr", type=str)
parent_parser.add_argument("--warm-start", default=False, type=str2bool)
parent_parser.add_argument("--class-weight", default=None, type=str)
parent_parser.add_argument("--balanced", default=False, type=str2bool)
parent_parser.add_argument("--temporal", default=True, type=str2bool)
parent_parser.add_argument("--normalization", default="yearly", choices=["all", "yearly", "none"])

# Only applies to non-temporal datasets
parent_parser.add_argument("--n-train", default=0.1, type=percentage)
parent_parser.add_argument("--n-update", default=0.7, type=percentage)
parent_parser.add_argument("--n-test", default=0.2, type=percentage)
parent_parser.add_argument("--num-updates", default=50, type=int)

# Only applies to temporal datasets
parent_parser.add_argument("--train-year-limit", default=1997, type=int)
parent_parser.add_argument("--update-year-limit", default=2019, type=int)
parent_parser.add_argument("--next-year", default=True, type=str2bool)
parent_parser.add_argument("--sorted", default=False, type=str2bool)

# Only for synthetic datasets
parent_parser.add_argument("--num-features", default=2, type=int)

parent_parser.add_argument("--initial-desired-rate", default="fpr", type=str)
parent_parser.add_argument("--initial-desired-value", default=0.2, type=float)
parent_parser.add_argument("--threshold-validation-percentage", default=0.2, type=float)
parent_parser.add_argument("--dynamic-desired-rate", default="fpr", type=str)
parent_parser.add_argument("--dynamic-desired-partition", default="all", type=str, choices=["train", "update_current",
                                                                                     "update_cumulative", "all"])
parent_parser.add_argument("--clinician-fprs", default=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2], type=float)
parent_parser.add_argument("--model-fprs", default=[0.2], type=float)

parent_parser.add_argument("--rate-types", default=["auc"], type=str)

parent_parser.add_argument("--lr", default=0.01, type=float)
parent_parser.add_argument("--online-lr", default=0.01, type=float)
parent_parser.add_argument("--optimizer", default="adam", type=str)
parent_parser.add_argument("--reset-optim", default=False, type=str2bool)
parent_parser.add_argument("--iterations", default=3000, type=int)
parent_parser.add_argument("--importance", default=100000.0, type=float)
parent_parser.add_argument("--tol", default=0.0001, type=float)
parent_parser.add_argument("--hidden-layers", default=0, type=int)
parent_parser.add_argument("--activation", default="Tanh", type=str)

parent_parser.add_argument("--bad-model", default=False, type=str2bool)
parent_parser.add_argument("--worst-case", default=False, type=str2bool)
parent_parser.add_argument("--update-type", default="feedback_full_fit_conditional_trust", type=str)

parent_parser.add_argument("--save-dir", default="figures/temp/trust_experiments", type=str)
parent_parser.add_argument("--file-name", default="timestamp", type=str, choices=["timestamp", "intuitive"])