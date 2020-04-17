import numpy as np
import os
import seaborn as sns

sns.set_style("white")

from src.scripts.helpers.generic.stats import summarize_stats

from src.scripts.helpers.generic.loops import get_update_loop
from src.scripts.helpers.updates.plotting import get_plot_fn
from src.scripts.helpers.updates.result_formatting import get_result_formatting_fn
from src.utils.data import get_data_fn
from src.utils.misc import create_config_file_name, create_plot_file_name, create_csv_file_name
from src.utils.model import get_model_fn
from src.utils.parse import percentage, str2bool, str2none
from src.utils.update import get_update_fn
from src.utils.save import create_file_path, save_json
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="mimic_iv_demographic", choices=["sklearn", "mimic_iii", "mimic_iv", "support2", "gaussian",
                                                                "mimic_iv_12h", "mimic_iv_24h", "mimic_iv_demographic",
                                                                            "mimic_iv_12h_demographic"], type=str)
parser.add_argument("--seeds", default=1, type=int)
parser.add_argument("--model", default="xgboost", type=str)
parser.add_argument("--warm-start", default=False, type=str2bool)
parser.add_argument("--class-weight", default=None, type=str)
parser.add_argument("--balanced", default=False, type=str2bool)
parser.add_argument("--temporal", default=True, type=str2bool)
parser.add_argument("--normalization", default=True, type=str2bool)

# Only applies to non-temporal datasets
parser.add_argument("--n-train", default=0.1, type=percentage)
parser.add_argument("--n-update", default=0.7, type=percentage)
parser.add_argument("--n-test", default=0.2, type=percentage)
parser.add_argument("--num-updates", default=50, type=int)

# Only applies to temporal datasets
parser.add_argument("--train-year-limit", default=1997, type=int)
parser.add_argument("--update-year-limit", default=2019, type=int)
parser.add_argument("--next-year", default=True, type=str2bool)
parser.add_argument("--sorted", default=False, type=str2bool)

# Only for synthetic datasets
parser.add_argument("--num-features", default=2, type=int)

parser.add_argument("--initial-desired-rate", default="fpr", type=str)
parser.add_argument("--initial-desired-value", default=0.05, type=float)
parser.add_argument("--threshold-validation-percentage", default=0.2, type=float)
parser.add_argument("--dynamic-desired-rate", default="fpr", type=str2none)
parser.add_argument("--dynamic-desired-partition", default="all", type=str, choices=["train", "update_current",
                                                                                     "update_cumulative", "all"])
parser.add_argument("--clinician-fpr", default=0.0, type=float)

parser.add_argument("--rate-types", default=["auc"], nargs="+")

# Only for pytorch models trained with GD
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--online-lr", default=0.0001, type=float)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--reset-optim", default=False, type=str2bool)
parser.add_argument("--iterations", default=3000, type=int)
parser.add_argument("--importance", default=100000.0, type=float)
parser.add_argument("--tol", default=0.0001, type=float)
parser.add_argument("--hidden-layers", default=0, type=int)
parser.add_argument("--activation", default="Tanh", type=str)
parser.add_argument("--soft", default=False, type=str2bool)

parser.add_argument("--bad-model", default=False, type=str2bool)
parser.add_argument("--worst-case", default=False, type=str2bool)
parser.add_argument("--update-types", default=["feedback_full_fit_drop_random", "feedback_full_fit_drop_everything", "feedback_full_fit_oracle"], nargs="+")
parser.add_argument("--limit-plot-range", default=False, type=str2bool)

parser.add_argument("--save-dir", default="figures/temp/temp", type=str)
parser.add_argument("--file-name", default="timestamp", type=str, choices=["timestamp", "intuitive"])


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    if args.save_dir is not None:
        results_dir = args.save_dir
    else:
        results_dir = os.environ.get("DESIRED_FPR_RESULTS_DIR")

    results_dir = os.path.join(ROOT_DIR, results_dir)

    data_fn = get_data_fn(args)
    model_fn = get_model_fn(args)
    train_update_loop = get_update_loop(args.temporal)
    result_formatting_fn = get_result_formatting_fn(args.temporal)
    plot_fn = get_plot_fn(args.temporal)
    temporal = args.temporal

    rates = {}
    stats = {}
    if type(args.update_types) is str:
        update_types = ["feedback_full_fit", "no_feedback_full_fit",
                        "feedback_full_fit_{}".format(args.update_types), "no_feedback_full_fit_{}".format(args.update_types),
                        "evaluate"]
    else:
        update_types = args.update_types

    for update_type in update_types:
        update_fn = get_update_fn(update_type, temporal=temporal)
        temp_rates, temp_stats = train_update_loop(model_fn=model_fn, n_train=args.n_train, n_update=args.n_update,
                                                   n_test=args.n_test, num_updates=args.num_updates,
                                                   num_features=args.num_features,
                                                   train_year_limit=args.train_year_limit,
                                                   update_year_limit=args.update_year_limit,
                                                   initial_desired_rate=args.initial_desired_rate,
                                                   initial_desired_value=args.initial_desired_value,
                                                   threshold_validation_percentage=args.threshold_validation_percentage,
                                                   dynamic_desired_rate=args.dynamic_desired_rate,
                                                   dynamic_desired_partition=args.dynamic_desired_partition,
                                                   data_fn=data_fn, update_fn=update_fn, bad_model=args.bad_model,
                                                   next_year=args.next_year, seeds=args.seeds, clinician_fpr=args.clinician_fpr)
        rates[update_type] = temp_rates
        stats[update_type] = temp_stats

    data = result_formatting_fn(rates, args.train_year_limit, args.update_year_limit)
    # for update_type in update_types:
    #     stats[update_type] = summarize_stats(stats[update_type])

    plot_name = "{}_{}_{}_{}_{}_{}".format(args.data_type, args.model, args.train_year_limit, args.update_year_limit, args.next_year, args.rate_types)
    plot_file_name = create_plot_file_name(args.file_name, plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)
    plot_title = ""
    create_file_path(plot_path)
    plot_fn(data, args.rate_types, update_types, args.limit_plot_range, plot_title, plot_path)

    config_file_name = create_config_file_name(args.file_name, plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)

    csv_file_name = create_csv_file_name(args.file_name, plot_name, timestamp)
    csv_path = os.path.join(results_dir, csv_file_name)
    data.to_csv(csv_path, index=False, header=True)


if __name__ == "__main__":
    main(parser.parse_args())