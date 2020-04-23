import hydra
import os
import seaborn as sns

sns.set_style("white")

from src.scripts.helpers.generic.loops import call_update_loop
from src.scripts.helpers.trust.result_formatting import get_result_formatting_fn
from src.utils.data import get_data_fn
from src.utils.model import get_model_fn
from src.utils.update import get_update_fn
from src.utils.save import CSV_FILE
from settings import ROOT_DIR

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs/conditional_trust.yaml")
@hydra.main(config_path=config_path)
def main(args):
    print(args.pretty())

    data_fn = get_data_fn(args.data, args.model)
    model_fn = get_model_fn(args.model)
    result_formatting_fn = get_result_formatting_fn(args.data.temporal, "conditional")

    rates = {update_type: {model_fpr: {} for model_fpr in args.rates.model_fprs} for update_type in args.misc.update_types}

    for update_type in args.misc.update_types:
        for model_fpr in args.rates.model_fprs:
            if update_type == "feedback_full_fit":
                clinician_fprs = [0.0]
            else:
                clinician_fprs = args.rates.clinician_fprs

            args.rates.idv = model_fpr
            for clinician_fpr in clinician_fprs:
                args.rates.clinician_fpr = clinician_fpr
                update_fn = get_update_fn(update_type, temporal=args.data.temporal)
                temp_rates, temp_stats = call_update_loop(args, data_fn, model_fn, update_fn)
                rates[update_type][model_fpr][clinician_fpr] = temp_rates

    data = result_formatting_fn(rates, args.data.tyl, args.data.uyl)

    csv_file_name = CSV_FILE
    data.to_csv(csv_file_name, index=False, header=True)



if __name__ == "__main__":
    main()