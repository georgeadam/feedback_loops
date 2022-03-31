from .blobs import generate_blobs_dataset
from .circles import generate_circles_dataset
from .gaussian import generate_gaussian_dataset
from .mimic_iii import load_mimic_iii_data
from .mimic_iv import load_mimic_iv_data, mimic_iv_paths
from .moons import generate_moons_dataset
from .s import generate_s_shape_dataset
from .sklearn import generate_sklearn_make_classification_dataset
from .static import generate_real_dataset_static
from .support import load_support2cls_data
from .temporal import generate_real_dataset_temporal
from .adult import generate_adult_dataset
from .credit_g import generate_credit_g_dataset
from .compas import generate_compas_dataset

STATIC_DATA_TYPES = ["gaussian", "sklearn", "moons", "mimic_iii", "support2", "sklearn_noisy_update", "moons_noisy_update", "compas", "adult", "credit"]
TEMPORAL_DATA_TYPES = ["mimic_iv", "mimic_iv_12h", "mimic_iv_24h"]