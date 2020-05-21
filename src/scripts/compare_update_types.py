import hydra
import os
import seaborn as sns

sns.set_style("white")

from src.scripts.helpers.generic.loops import call_update_loop
from src.scripts.helpers.updates.result_formatting import get_result_formatting_fn
from src.utils.data import get_data_fn
from src.utils.model import get_model_fn
from src.utils.update import get_update_fn
from src.utils.save import CSV_FILE

from omegaconf import DictConfig
from settings import ROOT_DIR

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs/update_types.yaml")
@hydra.main(config_path=config_path)
def main(args: DictConfig):
    print(args.pretty())
    print("Saving to: {}".format(os.getcwd()))

    data_fn = get_data_fn(args.data, args.model)
    model_fn = get_model_fn(args.model)
    result_formatting_fn = get_result_formatting_fn(args.data.temporal)
    temporal = args.data.temporal

    rates = {}
    stats = {}
    if type(args.misc.update_types) is str:
        update_types = ["feedback_full_fit", "no_feedback_full_fit",
                        "feedback_full_fit_{}".format(args.misc.update_types), "no_feedback_full_fit_{}".format(args.misc.update_types),
                        "evaluate"]
    else:
        update_types = args.misc.update_types

    for update_type in update_types:
        update_fn = get_update_fn(update_type, temporal=temporal)
        temp_rates, temp_stats = call_update_loop(args, data_fn, model_fn, update_fn)
        rates[update_type] = temp_rates
        stats[update_type] = temp_stats

    data = result_formatting_fn(rates, args.data.tyl, args.data.uyl)

    csv_file_name = CSV_FILE
    data.to_csv(csv_file_name, index=False, header=True)


if __name__ == "__main__":
    main()