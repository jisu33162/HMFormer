"""Internal fallback defaults. Set experiment parameters in the root parameter.txt."""

SEEDS = [1, 2, 3, 4, 5]
WINDOWS = [90, 120]
NOISE_RATES = [0.0, 0.2, 0.4]
BASELINE_ALPHA = 0.5
ADDITIVE_BETA = 0.5
HMM_STATES = 6
HMM_FIT_SAMPLES = 10000
CPU_THREADS = 2

TRAINING = {
    "width": 128,
    "heads": 8,
    "layers": 3,
    "dropout": 0.1,
    "lr": 0.0005,
    "weight_decay": 0.01,
    "batch_size": 1024,
    "max_epochs": 12,
    "patience": 4,
    "earliest_stop_epoch": 6,
    "gradient_clip": 1.0,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "feedforward_dim": None,
}

HMM_OPTIONS = {
    "covariance_floor": 0.001,
    "covariance_type": "full",
    "initialization": "temporal_gmm_soft_full",
    "max_iterations": 5000,
    "tolerance": 0.001,
    "batch_size": 256,
    "cm_max_iterations": 10000,
    "cm_tolerance": 1e-8,
    "gmm_max_iterations": 5000,
    "gmm_tolerance": 1e-8,
    "gmm_n_init": 1,
    "gmm_init_params": "random_from_data",
    "gmm_reg_covar": 0.001,
    "accept_gmm_max_iterations": True,
    "restarts": 1,
}

BASELINE_METHODS = [
    "Transformer", "CNN-TID-adapt", "Co-teaching",
    "JoCoR", "DivideMix", "CPNN-adapt",
]
BASELINE_EPOCHS = 100
BASELINE_BATCH_SIZE = 256
CPNN_PASSES = 10
CPNN_CAP = 4096
BASELINE_FINAL_LR = 5e-6
BASELINE_WARMUP_EPOCHS = 10
SWEEPS = {
    "K": [2, 4, 6, 8, 10],
    "alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
    "N_fit": [2000, 5000, 10000, 20000, 40000],
}
