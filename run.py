"""Run comparison, sensitivity, or ablation experiments on CA-TID-90/120."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import shlex
from statistics import mean, stdev
import sys

from experiments import config

ROOT = Path(__file__).resolve().parent


def cases_for(args):
    unique = {}
    methods = args.methods or (config.BASELINE_METHODS + ["HMFormer"]
                               if args.experiment == "comparison" else ["HMFormer"])
    if args.experiment == "ablation":
        methods = args.methods or [f"V{index}" for index in range(1, 8)]
    settings = [("", "", args.hmm_states, args.baseline_alpha, args.fit_samples)]
    if args.experiment == "sensitivity":
        settings = [("K", k, k, args.baseline_alpha, args.fit_samples) for k in args.k_values]
        settings += [("alpha", a, args.hmm_states, a, args.fit_samples) for a in args.alpha_values]
        settings += [("N_fit", n, args.hmm_states, args.baseline_alpha, n) for n in args.fit_sample_values]
    for window in args.windows:
        for seed in args.seeds:
            for rate in args.noise_rates:
                for method in methods:
                    for family, value, states, alpha, samples in settings:
                        key = (window, seed, rate, method, states, alpha, samples)
                        if key not in unique:
                            unique[key] = dict(window=window, seed=seed, noise_rate=rate,
                                               method=method, K=states, alpha=alpha,
                                               N_fit=samples, beta=args.additive_beta, sweeps=[])
                        if family:
                            unique[key]["sweeps"].append(dict(family=family, value=value))
    return list(unique.values())


def summarize(results, output):
    from experiments.train import METRICS, write_csv

    rows = []
    groups = defaultdict(list)
    for result in results:
        case = result["case"]
        row = {key: value for key, value in case.items() if key != "sweeps"}
        row.update(best_epoch=result["best_epoch"], elapsed_seconds=result["elapsed_seconds"])
        row.update({metric: result["metrics"][metric] for metric in METRICS})
        rows.append(row)
        for sweep in case["sweeps"] or [dict(family="", value="")]:
            key = (case["window"], case["method"], case["noise_rate"], case["K"],
                   case["alpha"], case["N_fit"], sweep["family"], sweep["value"])
            groups[key].append(result)
    write_csv(output / "runs.csv", rows)
    for view in ("metrics", "calibrated_metrics"):
        summaries = []
        for key, group in groups.items():
            valid = [result[view] for result in group if view in result]
            if not valid:
                continue
            row = dict(zip(("window", "method", "noise_rate", "K", "alpha", "N_fit", "family", "value"), key))
            row["n"] = len(valid)
            for metric in METRICS:
                values = [item[metric] for item in valid]
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_std"] = stdev(values) if len(values) > 1 else ""
            summaries.append(row)
        name = "summary.csv" if view == "metrics" else "summary_calibrated.csv"
        write_csv(output / name, summaries)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("Use a finite number.")
    return number


def positive_float(value):
    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Use a positive number.")
    return number


def parameters_from_text(path, parser):
    if not path.is_file():
        parser.error(f"Parameter file does not exist: {path}")
    options = []
    experiment = []
    seen = set()
    lists = {"seeds", "windows", "noise_rates", "methods",
             "k_values", "alpha_values", "fit_sample_values"}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key:
            parser.error(f"{path}, line {number}: use parameter = value.")
        if key in seen:
            parser.error(f"{path}, line {number}: duplicate parameter {key}.")
        seen.add(key)
        if key == "parameter_file":
            parser.error("Set --parameter-file when launching the program.")
        if not value:
            continue
        if key in {"data_root", "output"}:
            if value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            directory = Path(value).expanduser()
            if not directory.is_absolute():
                directory = path.parent / directory
            tokens = [str(directory.resolve())]
        else:
            lexer = shlex.shlex(value.replace(",", " ") if key in lists else value, posix=True)
            lexer.whitespace_split = True
            lexer.escape = ""
            try:
                tokens = list(lexer)
            except ValueError as error:
                parser.error(f"{path}, line {number}: {error}")
        if not tokens:
            continue
        if key == "experiment":
            experiment = tokens
        elif key == "accept_gmm_max_iterations":
            boolean = tokens[0].lower()
            if len(tokens) != 1 or boolean not in {"true", "false"}:
                parser.error(f"{path}, line {number}: use true or false.")
            option = "--accept-gmm-max-iterations" if boolean == "true" else "--no-accept-gmm-max-iterations"
            options.append(option)
        else:
            options.extend(["--" + key.replace("_", "-"), *tokens])
    return experiment + options


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiment", nargs="?", choices=("comparison", "sensitivity", "ablation"),
                        help="Override the experiment selected in parameter.txt.")
    parser.add_argument("--parameter-file", type=Path, default=ROOT / "parameter.txt",
                        help="Text file containing experiment parameters.")

    execution = parser.add_argument_group("Data and execution")
    execution.add_argument("--data-root", type=Path, default=ROOT / "Data", help="Dataset parent directory.")
    execution.add_argument("--output", type=Path, help="Result directory.")
    execution.add_argument("--seeds", type=positive_int, nargs="+", default=config.SEEDS, help="Repeat seeds.")
    execution.add_argument("--windows", type=int, nargs="+", choices=(90, 120), default=config.WINDOWS,
                           help="Window lengths in minutes.")
    execution.add_argument("--noise-rates", type=finite_float, nargs="+", default=config.NOISE_RATES,
                           help="Training label noise rates.")
    execution.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Classification device.")
    execution.add_argument("--cpu-threads", type=positive_int, default=config.CPU_THREADS,
                           help="CPU threads used by PyTorch.")
    execution.add_argument("--methods", nargs="+", help="Selected comparison methods or V1-V7 ablation variants.")

    model = parser.add_argument_group("Model")
    model.add_argument("--width", type=positive_int, default=config.TRAINING["width"], help="Transformer width.")
    model.add_argument("--heads", type=positive_int, default=config.TRAINING["heads"], help="Attention heads.")
    model.add_argument("--layers", type=positive_int, default=config.TRAINING["layers"], help="Encoder layers.")
    model.add_argument("--ffn-dim", type=positive_int, default=config.TRAINING["feedforward_dim"],
                       help="Feed-forward dimension. Uses twice the width when omitted.")
    model.add_argument("--dropout", type=finite_float, default=config.TRAINING["dropout"], help="Dropout rate.")
    model.add_argument("--hmm-states", "--k", type=positive_int, default=config.HMM_STATES,
                       help="HMM states. Fixed outside the K sweep.")
    model.add_argument("--baseline-alpha", "--alpha", type=finite_float, default=config.BASELINE_ALPHA,
                       help="Multiplicative gate strength. Fixed in K and N_fit sweeps.")
    model.add_argument("--additive-beta", type=finite_float, default=config.ADDITIVE_BETA,
                       help="Additive fusion strength for ablation V4.")
    model.add_argument("--fit-samples", type=positive_int, default=config.HMM_FIT_SAMPLES,
                       help="HMM fitting sequences. Fixed outside the N_fit sweep.")

    training = parser.add_argument_group("Training")
    training.add_argument("--epochs", type=positive_int, help="Override both classifier epoch limits.")
    training.add_argument("--hmformer-epochs", type=positive_int, default=config.TRAINING["max_epochs"],
                          help="HMFormer and ablation epoch limit.")
    training.add_argument("--baseline-epochs", type=positive_int, default=config.BASELINE_EPOCHS,
                          help="Neural comparison baseline epoch limit.")
    training.add_argument("--batch-size", type=positive_int, help="Override both classification batch sizes.")
    training.add_argument("--hmformer-batch-size", type=positive_int, default=config.TRAINING["batch_size"],
                          help="HMFormer and ablation batch size.")
    training.add_argument("--baseline-batch-size", type=positive_int, default=config.BASELINE_BATCH_SIZE,
                          help="Comparison baseline batch size.")
    training.add_argument("--lr", type=positive_float, default=config.TRAINING["lr"], help="AdamW learning rate.")
    training.add_argument("--weight-decay", type=finite_float, default=config.TRAINING["weight_decay"],
                          help="AdamW weight decay.")
    training.add_argument("--beta1", type=finite_float, default=config.TRAINING["betas"][0], help="AdamW beta1.")
    training.add_argument("--beta2", type=finite_float, default=config.TRAINING["betas"][1], help="AdamW beta2.")
    training.add_argument("--eps", type=positive_float, default=config.TRAINING["eps"], help="AdamW epsilon.")
    training.add_argument("--patience", type=positive_int, default=config.TRAINING["patience"],
                          help="HMFormer and ablation early-stopping patience.")
    training.add_argument("--earliest-stop-epoch", type=positive_int, default=config.TRAINING["earliest_stop_epoch"],
                          help="First epoch eligible for early stopping.")
    training.add_argument("--gradient-clip", type=positive_float, default=config.TRAINING["gradient_clip"],
                          help="Maximum gradient norm.")
    training.add_argument("--baseline-final-lr", type=finite_float, default=config.BASELINE_FINAL_LR,
                          help="Final cosine learning rate for neural comparison baselines.")
    training.add_argument("--baseline-warmup-epochs", type=int, default=config.BASELINE_WARMUP_EPOCHS,
                          help="Constant-LR epochs and DivideMix supervised warmup epochs.")
    training.add_argument("--cpnn-passes", type=positive_int, default=config.CPNN_PASSES, help="CPNN pass limit.")
    training.add_argument("--cpnn-cap", type=positive_int, default=config.CPNN_CAP, help="CPNN prototype limit.")

    hmm = parser.add_argument_group("HMM and temporal GMM fitting")
    hmm.add_argument("--hmm-initialization", choices=("temporal_gmm_soft_full", "temporal_gmm_hard_full",
                      "temporal_gmm", "gmm", "kmeans", "random"), default=config.HMM_OPTIONS["initialization"],
                     help="HMM initialization method.")
    hmm.add_argument("--hmm-covariance-type", choices=("full", "diag"),
                     default=config.HMM_OPTIONS["covariance_type"], help="Emission covariance type.")
    hmm.add_argument("--hmm-covariance-floor", type=positive_float, default=config.HMM_OPTIONS["covariance_floor"],
                     help="Minimum emission covariance eigenvalue.")
    hmm.add_argument("--hmm-max-iterations", type=positive_int, default=config.HMM_OPTIONS["max_iterations"],
                     help="Outer EM iteration limit.")
    hmm.add_argument("--hmm-tolerance", type=positive_float, default=config.HMM_OPTIONS["tolerance"],
                     help="EM tolerance at 10000 sequences, scaled by actual sample count.")
    hmm.add_argument("--hmm-batch-size", type=positive_int, default=config.HMM_OPTIONS["batch_size"],
                     help="HMM fitting batch size.")
    hmm.add_argument("--cm-max-iterations", type=positive_int, default=config.HMM_OPTIONS["cm_max_iterations"],
                     help="Conditional maximization iteration limit.")
    hmm.add_argument("--cm-tolerance", type=positive_float, default=config.HMM_OPTIONS["cm_tolerance"],
                     help="Conditional maximization tolerance.")
    hmm.add_argument("--gmm-max-iterations", type=positive_int, default=config.HMM_OPTIONS["gmm_max_iterations"],
                     help="Temporal GMM iteration limit.")
    hmm.add_argument("--gmm-tolerance", type=positive_float, default=config.HMM_OPTIONS["gmm_tolerance"],
                     help="Temporal GMM tolerance.")
    hmm.add_argument("--gmm-n-init", type=positive_int, default=config.HMM_OPTIONS["gmm_n_init"],
                     help="Temporal GMM initialization count.")
    hmm.add_argument("--gmm-init-params", choices=("random_from_data", "kmeans", "k-means++", "random"),
                     default=config.HMM_OPTIONS["gmm_init_params"], help="Temporal GMM initialization rule.")
    hmm.add_argument("--gmm-reg-covar", type=finite_float, default=config.HMM_OPTIONS["gmm_reg_covar"],
                     help="Temporal GMM covariance regularization.")
    hmm.add_argument("--accept-gmm-max-iterations", action=argparse.BooleanOptionalAction,
                     default=config.HMM_OPTIONS["accept_gmm_max_iterations"],
                     help="Allow a valid GMM initialization at the iteration limit.")

    sweeps = parser.add_argument_group("Sensitivity sweep values")
    sweeps.add_argument("--k-values", type=positive_int, nargs="+", default=config.SWEEPS["K"], help="K sweep.")
    sweeps.add_argument("--alpha-values", type=finite_float, nargs="+", default=config.SWEEPS["alpha"],
                        help="Gate strength sweep.")
    sweeps.add_argument("--fit-sample-values", type=positive_int, nargs="+", default=config.SWEEPS["N_fit"],
                        help="N_fit sweep.")
    if "-h" in argv or "--help" in argv:
        parser.parse_args(argv)
    location_parser = argparse.ArgumentParser(add_help=False)
    location_parser.add_argument("--parameter-file", type=Path, default=ROOT / "parameter.txt")
    location, _ = location_parser.parse_known_args(argv)
    parameter_file = location.parameter_file.expanduser().resolve()
    defaults = parser.parse_args(parameters_from_text(parameter_file, parser))
    selected_experiment = defaults.experiment or "comparison"
    args = parser.parse_args(argv, namespace=defaults)
    args.experiment = args.experiment or selected_experiment
    args.parameter_file = parameter_file
    for name in ("seeds", "windows", "noise_rates", "k_values", "alpha_values", "fit_sample_values"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"Use distinct values for --{name.replace('_', '-')}.")
    if any(rate < 0 or rate > 1 for rate in args.noise_rates):
        parser.error("Noise rates must lie in [0, 1].")
    if args.width % args.heads:
        parser.error("--width must be divisible by --heads.")
    if args.hmm_states < 2 or min(args.k_values) < 2:
        parser.error("HMM state counts must be at least 2.")
    if args.cpnn_cap < 3:
        parser.error("--cpnn-cap must be at least 3 for three-class classification.")
    if not 0 < args.baseline_alpha < 1 or any(not 0 < value < 1 for value in args.alpha_values):
        parser.error("Gate strengths must lie strictly between 0 and 1.")
    if not 0 <= args.dropout < 1 or not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        parser.error("Dropout and optimizer betas must lie in [0, 1).")
    if min(args.additive_beta, args.weight_decay, args.baseline_final_lr,
           args.baseline_warmup_epochs, args.gmm_reg_covar) < 0:
        parser.error("Fusion, regularization, final learning rate, and warmup values must be nonnegative.")
    allowed = set(config.BASELINE_METHODS + ["HMFormer"])
    if args.experiment == "ablation":
        allowed = {f"V{index}" for index in range(1, 8)}
    elif args.experiment == "sensitivity":
        allowed = {"HMFormer"}
    if args.methods and any(method not in allowed for method in args.methods):
        parser.error(f"Allowed methods for {args.experiment}: {', '.join(sorted(allowed))}")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from experiments.train import load_data, run_case, write_csv

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch or pass --device cpu.")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    suffix = "" if args.experiment == "ablation" else f"_alpha{args.baseline_alpha:g}"
    output = (args.output or ROOT / "outputs" / f"{args.experiment}{suffix}").resolve()
    if output.is_relative_to(args.data_root.resolve()):
        raise ValueError("Choose an output directory outside Data.")
    output.mkdir(parents=True, exist_ok=True)
    training = dict(width=args.width, heads=args.heads, layers=args.layers,
                    feedforward_dim=args.ffn_dim, dropout=args.dropout,
                    lr=args.lr, weight_decay=args.weight_decay, betas=[args.beta1, args.beta2], eps=args.eps,
                    max_epochs=args.epochs or args.hmformer_epochs,
                    batch_size=args.batch_size or args.hmformer_batch_size,
                    patience=args.patience, earliest_stop_epoch=args.earliest_stop_epoch,
                    gradient_clip=args.gradient_clip)
    fitting = dict(initialization=args.hmm_initialization, covariance_floor=args.hmm_covariance_floor,
                   covariance_type=args.hmm_covariance_type, max_iterations=args.hmm_max_iterations,
                   tolerance=args.hmm_tolerance, batch_size=args.hmm_batch_size,
                   cm_max_iterations=args.cm_max_iterations, cm_tolerance=args.cm_tolerance,
                   gmm_max_iterations=args.gmm_max_iterations, gmm_tolerance=args.gmm_tolerance,
                   gmm_n_init=args.gmm_n_init, gmm_init_params=args.gmm_init_params,
                   gmm_reg_covar=args.gmm_reg_covar,
                   accept_gmm_max_iterations=args.accept_gmm_max_iterations, restarts=1)
    baseline = dict(cpnn_passes=args.cpnn_passes, cpnn_cap=args.cpnn_cap,
                    final_lr=args.baseline_final_lr, warmup_epochs=args.baseline_warmup_epochs)
    baseline_epochs = args.epochs or args.baseline_epochs
    baseline_batch = args.batch_size or args.baseline_batch_size
    results = []
    failures = []
    current_window = None
    data = None
    cases = cases_for(args)
    for index, case in enumerate(cases, 1):
        if current_window != case["window"]:
            current_window = case["window"]
            data = load_data(args.data_root, current_window)
        print(f"[{index}/{len(cases)}] {case['method']}, window={case['window']}, "
              f"seed={case['seed']}, noise={case['noise_rate']:g}", flush=True)
        try:
            result = run_case(case, data, output, training, baseline_epochs, baseline_batch, device,
                              calibrated=args.experiment == "sensitivity",
                              hmm_options=fitting, baseline_options=baseline)
            results.append(result)
        except (RuntimeError, ValueError) as error:
            failures.append({key: value for key, value in case.items() if key != "sweeps"} | {"error": str(error)})
            print(f"FAILED: {error}", flush=True)
        summarize(results, output)
        write_csv(output / "failures.csv", failures)
    if not failures and (output / "failures.csv").exists():
        (output / "failures.csv").unlink()
    print(f"Finished {len(results)}/{len(cases)} runs. Results: {output}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
