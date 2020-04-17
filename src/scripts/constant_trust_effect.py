import os
import seaborn as sns

from src.scripts.helpers.generic.stats import summarize_stats

sns.set_style("white")

from src.scripts.helpers.generic.loops import get_update_loop
from src.scripts.helpers.trust.parse import parent_parser
from src.scripts.helpers.trust.plotting import get_plot_fn
from src.scripts.helpers.trust.result_formatting import get_result_formatting_fn
from src.utils.data import get_data_fn
from src.utils.misc import create_config_file_name, create_plot_file_name, create_csv_file_name
from src.utils.model import get_model_fn
from src.utils.parse import percentage, str2bool
from src.utils.update import get_update_fn
from src.utils.save import create_file_path, save_json
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser(parents=[parent_parser], conflict_handler="resolve")
parser.add_argument("--data-type", default="mimic_iv", choices=["sklearn", "mimic_iii", "mimic_iv", "support2", "gaussian",
                                                                "mimic_iv_12h", "mimic_iv_24h", "mimic_iv_demographic",
                                                                            "mimic_iv_12h_demographic"], type=str)
parser.add_argument("--seeds", default=1, type=int)
parser.add_argument("--model", default="lr", type=str)
parser.add_argument("--temporal", default=True, type=str2bool)

parser.add_argument("--clinician-fpr", default=0.2, type=float)
parser.add_argument("--model-fpr", default=0.2, type=float)
parser.add_argument("--clinician-trusts", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], nargs="+")

parser.add_argument("--update-type", default="feedback_full_fit_constant_trust", type=str)
parser.add_argument("--rate-types", default=["auc", "fpr"], nargs="+")

parser.add_argument("--save-dir", default="figures/temp/trust_experiments/constant", type=str)
parser.add_argument("--file-name", default="timestamp", type=str, choices=["timestamp", "intuitive"])


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()
    results_dir = args.save_dir
    results_dir = os.path.join(ROOT_DIR, results_dir)

    data_fn = get_data_fn(args)
    model_fn = get_model_fn(args)
    train_update_loop = get_update_loop(args.temporal)
    result_formatting_fn = get_result_formatting_fn(args.temporal, "constant")
    plot_fn = get_plot_fn(args.temporal, "constant")
    temporal = args.temporal

    rates = {}
    stats = {}

    update_type = args.update_type

    for trust in args.clinician_trusts:
        update_fn = get_update_fn(update_type, temporal=temporal)
        temp_rates, temp_stats = train_update_loop(model_fn=model_fn, n_train=args.n_train, n_update=args.n_update,
                                                   n_test=args.n_test, num_updates=args.num_updates,
                                                   num_features=args.num_features,
                                                   train_year_limit=args.train_year_limit,
                                                   update_year_limit=args.update_year_limit,
                                                   initial_desired_rate=args.initial_desired_rate,
                                                   initial_desired_value=args.model_fpr,
                                                   threshold_validation_percentage=args.threshold_validation_percentage,
                                                   dynamic_desired_rate=args.dynamic_desired_rate,
                                                   dynamic_desired_partition=args.dynamic_desired_partition,
                                                   data_fn=data_fn, update_fn=update_fn, bad_model=args.bad_model,
                                                   next_year=args.next_year, seeds=args.seeds, clinician_fpr=args.clinician_fpr,
                                                   clinician_trust=trust)
        rates[trust] = temp_rates
        stats[trust] = temp_stats

    data = result_formatting_fn(rates, args.train_year_limit, args.update_year_limit)

    for trust in args.clinician_trusts:
        stats[trust] = summarize_stats(stats[trust])

    for rate_type in args.rate_types:
        plot_name = "{}_{}_{}_{}_{}".format(args.data_type, args.model, rate_type, args.clinician_fpr, args.model_fpr)
        plot_file_name = create_plot_file_name(args.file_name, plot_name, timestamp)
        plot_path = os.path.join(results_dir, plot_file_name)
        plot_title = ""
        create_file_path(plot_path)
        plot_fn(data, rate_type, plot_title, plot_path)

    config_file_name = create_config_file_name(args.file_name, plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)

    csv_file_name = create_csv_file_name(args.file_name, plot_name, timestamp)
    csv_path = os.path.join(results_dir, csv_file_name)
    data.to_csv(csv_path, index=False, header=True)


if __name__ == "__main__":
    main(parser.parse_args())