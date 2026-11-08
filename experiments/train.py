"""Training and evaluation for HMFormer, ablations, and comparison methods."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .ablation import AblationModel
from .baseline_training import Trainer
from .common import seed_all
from .cpnn import CPNN
from .hmm import HMMConvergenceError, SharedARHMM, fit_shared_hmm
from .hmformer import TinyTransformerHMMGate

METRICS = ("accuracy", "macro_f1", "DR", "FAR")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_data(root, window):
    dataset = {}
    for split in ("Train", "Validation", "Test"):
        directory = Path(root) / f"CA-TID-{window}" / split
        features = np.load(directory / "zscore_speed_occ_flow.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(directory / "labels.npy", allow_pickle=False)
        if features.shape != (len(labels), window // 5, 3):
            raise ValueError(f"Invalid feature shape in {directory}: {features.shape}")
        if labels.ndim != 1 or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError(f"Invalid labels in {directory}")
        dataset[split] = (features, labels.astype(np.int64, copy=False))
    return dataset


def noisy_labels(clean, rate, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(len(clean))
    offsets = rng.integers(1, 3, size=len(clean))
    selected = order[:round(len(clean) * rate)]
    labels = clean.copy()
    labels[selected] = (clean[selected] + offsets[selected]) % 3
    return labels


def scores(labels, predictions):
    matrix = np.bincount(3 * labels + predictions, minlength=9).reshape(3, 3)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * matrix.diagonal(), denominator,
                   out=np.zeros(3, dtype=np.float64), where=denominator > 0)
    events = matrix[:2].sum()
    normal = matrix[2].sum()
    return {
        "accuracy": 100 * float(matrix.trace() / matrix.sum()),
        "macro_f1": 100 * float(f1.mean()),
        "DR": 100 * float(matrix[:2, :2].sum() / events) if events else 0.0,
        "FAR": 100 * float(matrix[2, :2].sum() / normal) if normal else 0.0,
        "confusion_matrix": matrix.tolist(),
    }


def calibration_bias(labels, logits):
    best = None
    for tick in range(-50, 51):
        bias = tick / 20
        adjusted = logits.copy()
        adjusted[:, 2] += np.float32(bias)
        metrics = scores(labels, adjusted.argmax(axis=1))
        rank = (metrics["macro_f1"], metrics["accuracy"], -abs(bias), -bias)
        if best is None or rank > best[0]:
            best = (rank, bias)
    return best[1]


def fit_hmm(features, window, states, samples, seed, ar_order, output, hmm_options=None):
    count = min(samples, len(features))
    options = dict(config.HMM_OPTIONS if hmm_options is None else hmm_options)
    covariance_floor = options.pop("covariance_floor", 0.001)
    covariance_type = options.pop("covariance_type", "full")
    options["tolerance"] *= count / 10000
    settings = dict(window=window, states=states, samples=count, seed=seed,
                    ar_order=ar_order, covariance_floor=covariance_floor, covariance_type=covariance_type,
                    fitting=options)
    path = Path(output) / "hmms" / f"w{window}_s{seed}_k{states}_n{count}_ar{ar_order}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["settings"] != settings:
            raise ValueError(f"HMM settings changed. Choose a new output directory: {path}")
        return saved["state"]
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(features))[:count]
    sample = np.ascontiguousarray(features[order], dtype=np.float64)
    for fitting_seed in (seed, seed + 100):
        seed_all(fitting_seed)
        hmm = SharedARHMM(states=states, ar_order=ar_order, covariance_floor=covariance_floor,
                         covariance_type=covariance_type).double()
        print(f"Fitting HMM: window={window}, K={states}, samples={count}, seed={fitting_seed}",
              flush=True)

        def progress(row):
            if row["iteration"] == 1 or row["iteration"] % 100 == 0:
                print(f"  EM iteration {row['iteration']}, gain={row['gain']:.6g}", flush=True)

        try:
            report = fit_shared_hmm(hmm, sample, seed=fitting_seed, progress=progress, **options)
            state = {name: value.detach().cpu().float() for name, value in hmm.state_dict().items()}
            save_checkpoint(path, dict(settings=settings, state=state, fitting_seed=fitting_seed))
            print(f"HMM fitted after {report['iterations']} iterations", flush=True)
            return state
        except HMMConvergenceError as error:
            budget_exhausted = str(error).startswith("Shared HMM did not converge in ")
            if fitting_seed != seed or not budget_exhausted:
                raise
            print(f"Retrying the HMM with seed {seed + 100}", flush=True)
    raise RuntimeError("HMM fitting failed")


def make_model(case, steps, state, training, hmm_options=None):
    seed_all(case["seed"])
    fitting = config.HMM_OPTIONS if hmm_options is None else hmm_options
    options = dict(steps=steps, width=training["width"], heads=training["heads"],
                   layers=training["layers"], dropout=training["dropout"],
                   hmm_states=case["K"], gate_strength=case["alpha"],
                   hmm_ar_order=1, hmm_state=state, train_hmm=False,
                   feedforward_dim=training.get("feedforward_dim"),
                   hmm_covariance_floor=fitting.get("covariance_floor", 0.001),
                   hmm_covariance_type=fitting.get("covariance_type", "full"))
    if case["method"].startswith("V"):
        options["hmm_state"] = None
    model = TinyTransformerHMMGate(**options)
    if case["method"].startswith("V"):
        initialization = {name: getattr(model, name).state_dict()
                          for name in ("backbone", "qkv_hmm", "relations")}
        model = AblationModel(case["method"], steps, initialization, state,
                              width=training["width"], heads=training["heads"],
                              layers=training["layers"], dropout=training["dropout"],
                              feedforward_dim=training.get("feedforward_dim"), states=case["K"],
                              gate_strength=case["alpha"], additive_strength=case.get("beta", 0.5),
                              covariance_floor=fitting.get("covariance_floor", 0.001),
                              covariance_type=fitting.get("covariance_type", "full"))
    return model


def donors(count, seed, device):
    if count < 2:
        raise ValueError("V7 requires at least two samples in each split.")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(count)
    mapping = np.empty(count, dtype=np.int64)
    mapping[order] = np.roll(order, 1)
    return torch.as_tensor(mapping, device=device)


@torch.no_grad()
def predict_logits(model, features, batch_size, mapping=None):
    model.eval()
    logits = []
    for start in range(0, len(features), batch_size):
        inputs = features[start:start + batch_size]
        arguments = {}
        if mapping is not None:
            arguments["hmm_input"] = features[mapping[start:start + batch_size]]
        logits.append(model(inputs, **arguments).cpu().numpy())
    return np.concatenate(logits)


def train_hmformer(case, data, directory, output, training, device, calibrated, hmm_options=None):
    state = None
    if case["method"] != "V1":
        state = fit_hmm(data["Train"][0], case["window"], case["K"], case["N_fit"],
                        case["seed"], 0 if case["method"] == "V6" else 1, output, hmm_options)
    model = make_model(case, case["window"] // 5, state, training, hmm_options).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=training["lr"], weight_decay=training["weight_decay"],
                                  betas=training.get("betas", (0.9, 0.999)), eps=training.get("eps", 1e-8))
    features = {split: torch.as_tensor(np.array(pair[0], copy=True), device=device)
                for split, pair in data.items()}
    labels = noisy_labels(data["Train"][1], case["noise_rate"], case["seed"])
    targets = torch.as_tensor(labels, device=device)
    mappings = {}
    if case["method"] == "V7":
        mappings = {split: donors(len(inputs), case["seed"], device)
                    for split, inputs in features.items()}
    generator = torch.Generator().manual_seed(case["seed"])
    batches = DataLoader(TensorDataset(torch.arange(len(labels))),
                         batch_size=training["batch_size"], shuffle=True, generator=generator)
    seed_all(case["seed"])
    best_rank = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, training["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for (indices,) in batches:
            indices = indices.to(device)
            arguments = {}
            if "Train" in mappings:
                arguments["hmm_input"] = features["Train"][mappings["Train"][indices]]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features["Train"][indices], **arguments), targets[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite classification loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"],
                                           error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        logits = predict_logits(model, features["Validation"], training["batch_size"],
                                 mappings.get("Validation"))
        validation = scores(data["Validation"][1], logits.argmax(axis=1))
        rank = (validation["macro_f1"], validation["accuracy"])
        history.append(dict(epoch=epoch, train_loss=total_loss / len(labels),
                            validation_accuracy=validation["accuracy"],
                            validation_macro_f1=validation["macro_f1"]))
        if best_rank is None or rank > best_rank:
            best_rank, best_epoch, stale = rank, epoch, 0
            save_checkpoint(directory / "model.pt",
                            dict(state={key: value.detach().cpu() for key, value in model.state_dict().items()},
                                 case=case, epoch=epoch))
        else:
            stale += 1
        write_csv(directory / "history.csv", history)
        print(f"{directory.name}: epoch={epoch}, validation Macro-F1={rank[0]:.2f}", flush=True)
        if epoch >= training["earliest_stop_epoch"] and stale >= training["patience"]:
            break
    checkpoint = torch.load(directory / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state"])
    logits = predict_logits(model, features["Test"], training["batch_size"], mappings.get("Test"))
    predictions = logits.argmax(axis=1)
    np.save(directory / "test_predictions.npy", predictions, allow_pickle=False)
    np.save(directory / "test_logits.npy", logits, allow_pickle=False)
    result = dict(best_epoch=best_epoch, metrics=scores(data["Test"][1], predictions))
    if calibrated:
        validation = predict_logits(model, features["Validation"], training["batch_size"])
        bias = calibration_bias(data["Validation"][1], validation)
        adjusted = logits.copy()
        adjusted[:, 2] += np.float32(bias)
        result["calibration_bias"] = bias
        result["calibrated_metrics"] = scores(data["Test"][1], adjusted.argmax(axis=1))
    return result


def train_baseline(case, data, directory, epochs, batch_size, device, training=None, baseline_options=None):
    method = case["method"]
    options = dict(cpnn_passes=config.CPNN_PASSES, cpnn_cap=config.CPNN_CAP,
                   final_lr=config.BASELINE_FINAL_LR, warmup_epochs=config.BASELINE_WARMUP_EPOCHS)
    options.update(baseline_options or {})
    seed_all(case["seed"])
    labels = noisy_labels(data["Train"][1], case["noise_rate"], case["seed"])
    if method == "CPNN-adapt":
        trainer = CPNN(case["seed"], case["window"] // 5 * 3, cap=options["cpnn_cap"])
        epochs = min(epochs, options["cpnn_passes"])
    else:
        trainer = Trainer(dict(method=method, train_seed=case["seed"],
                               noise_rate=case["noise_rate"]),
                           case["window"] // 5, device, batch_size, training=training,
                           total_epochs=epochs, final_lr=options["final_lr"],
                           warmup_epochs=options["warmup_epochs"])
    best_rank = None
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        values = trainer.epoch(data["Train"][0], labels, epoch)
        validation = scores(data["Validation"][1], trainer.predict(data["Validation"][0]).argmax(axis=1))
        rank = (validation["macro_f1"], validation["accuracy"])
        history.append(dict(epoch=epoch, train_loss=values["train_loss"],
                            validation_accuracy=validation["accuracy"],
                            validation_macro_f1=validation["macro_f1"]))
        # Select neural checkpoints after warmup, or at the final epoch of a shorter run.
        if method == "CPNN-adapt" or epoch >= min(options["warmup_epochs"] + 1, epochs):
            if best_rank is None or rank > best_rank:
                best_rank, best_epoch = rank, epoch
                save_checkpoint(directory / "model.pt",
                                dict(state=trainer.state(), case=case, epoch=epoch))
        write_csv(directory / "history.csv", history)
        print(f"{directory.name}: epoch={epoch}, validation Macro-F1={rank[0]:.2f}", flush=True)
        if method == "CPNN-adapt" and trainer.stable >= 2:
            break
    selected = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    trainer.load(selected["state"])
    predictions = trainer.predict(data["Test"][0]).argmax(axis=1)
    np.save(directory / "test_predictions.npy", predictions, allow_pickle=False)
    return dict(best_epoch=best_epoch, metrics=scores(data["Test"][1], predictions))


def run_case(case, data, output, training, baseline_epochs, baseline_batch, device, calibrated,
             *, hmm_options=None, baseline_options=None):
    identifier = (f"w{case['window']}_{case['method']}_s{case['seed']}_noise{round(100 * case['noise_rate'])}"
                  f"_k{case['K']}_a{case['alpha']:g}_n{case['N_fit']}")
    directory = Path(output) / "runs" / identifier
    directory.mkdir(parents=True, exist_ok=True)
    fitting = dict(config.HMM_OPTIONS if hmm_options is None else hmm_options)
    baseline = dict(cpnn_passes=config.CPNN_PASSES, cpnn_cap=config.CPNN_CAP,
                    final_lr=config.BASELINE_FINAL_LR, warmup_epochs=config.BASELINE_WARMUP_EPOCHS)
    baseline.update(baseline_options or {})
    settings = dict(case=case, training=training, hmm_options=fitting, baseline_options=baseline,
                    baseline_epochs=baseline_epochs, baseline_batch=baseline_batch,
                    calibrated=calibrated, device=str(device))
    settings_path = directory / "config.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError(f"Settings changed. Choose a new output directory: {directory}")
    write_json(settings_path, settings)
    result_path = directory / "metrics.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    if case["method"] in config.BASELINE_METHODS:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result = train_baseline(case, data, directory, baseline_epochs, baseline_batch, device,
                                 training=training, baseline_options=baseline)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        result = train_hmformer(case, data, directory, output, training, device, calibrated, fitting)
    result.update(case=case, elapsed_seconds=time.perf_counter() - started)
    write_json(result_path, result)
    return result
