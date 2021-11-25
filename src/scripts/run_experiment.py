import hydra
import os

from src.scripts.experiment.loop import call_experiment_loop
from src.scripts.experiment.formatting import get_result_formatting_fn
from src.data import get_data_fn, get_data_wrapper_fn
from src.models import get_model_fn
from src.scripts.experiment.update import get_update_fn
from src.utils.save import CSV_FILE
from src.trainers import get_trainer

from omegaconf import DictConfig
from settings import ROOT_DIR

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
@hydra.main(config_path=config_path, config_name="update_types")
def main(args: DictConfig):
    print(args.pretty())
    print("Saving to: {}".format(os.getcwd()))

    inner_data_fn = get_data_fn(args)
    data_fn = lambda: inner_data_fn(args.data.n_train, args.data.n_val, args.data.n_update, args.data.n_test, args.data.num_features)
    data_wrapper_fn = get_data_wrapper_fn(args)

    model_fn = get_model_fn(args)
    trainer = get_trainer(args)
    update_fn = get_update_fn(args)

    result_formatting_fn = get_result_formatting_fn(args.data.temporal)
    rates = {}

    temp_rates, _ = call_experiment_loop(args, data_fn, data_wrapper_fn, model_fn, trainer, update_fn)
    rates[args.update_params.type] = temp_rates
    data = result_formatting_fn(rates, args.data.tyl, args.data.uyl)

    csv_file_name = CSV_FILE
    data.to_csv(csv_file_name, index=False, header=True)


if __name__ == "__main__":
    main()