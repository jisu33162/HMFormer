"""Build CA-TID-90 and CA-TID-120 from raw XTraffic observations or a 120-minute population."""
import argparse
import calendar
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURES = "zscore_speed_occ_flow.npy"
CHANNELS = ["speed", "occupancy", "flow"]
CHUNK_ROWS = 8192
POINTS_PER_DAY = 288
MONTH_PATHS = [f"year_2023/year_2023/2023_p{month:02d}.npy" for month in range(1, 13)]


def load(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_selections():
    parts = {split: np.load(ROOT / "selections" / f"{split}.npy", allow_pickle=False)
             for split in ("Train", "Validation", "Test")}
    for split, selected in parts.items():
        if selected.ndim != 2 or selected.shape[1] != 4 or np.any(selected < 0):
            raise ValueError(f"Invalid row selections for {split}")
        if not np.isin(selected[:, 3], [0, 1, 2]).all():
            raise ValueError(f"Invalid labels in {split} selections")
    indices = np.concatenate([selected[:, 0] for selected in parts.values()])
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Split selections contain overlapping population rows")
    return parts


def read_monthly(archive, selected):
    nodes = load(archive / "node_order.npy")
    coordinates = selected[:, 1]
    if nodes.ndim != 1 or np.any(coordinates >= len(nodes)):
        raise ValueError("Invalid sensor node index")
    slots = selected[:, 2, None] + np.arange(24, dtype=np.int32)
    if np.any(slots >= 365 * POINTS_PER_DAY):
        raise ValueError("Selected observations extend beyond 2023")
    raw = np.empty((len(selected), 24, 3), dtype="<f4")
    month_start = 0
    for month, name in enumerate(MONTH_PATHS, 1):
        points = calendar.monthrange(2023, month)[1] * POINTS_PER_DAY
        traffic = load(archive / name)
        row, step = np.where((slots >= month_start) & (slots < month_start + points))
        local = slots[row, step] - month_start
        node = coordinates[row]
        if traffic.shape == (len(nodes), points, 3):
            order = np.lexsort((local, node))
            values = traffic[node[order], local[order]]
        elif traffic.shape == (points, len(nodes), 3):
            order = np.lexsort((node, local))
            values = traffic[local[order], node[order]]
        else:
            raise ValueError(f"Unexpected shape in {name}: {traffic.shape}")
        raw[row[order], step[order]] = values[:, [2, 1, 0]]
        month_start += points
        print(f"Read month {month:02d}", flush=True)
    return raw


def read_population(directory, selected):
    files = ("raw_speed_occ_flow.npy", "sample_metadata.npy")
    if not (directory / files[0]).is_file():
        files = ("raw_windows.npy", "samples.npy")
    raw, metadata = (load(directory / name) for name in files)
    if raw.shape != (len(metadata), 24, 3):
        raise ValueError(f"Invalid 120-minute population shape: {raw.shape}")
    indices = selected[:, 0]
    for column, field in enumerate(("node_index", "start_index", "label"), 1):
        if not np.array_equal(metadata[field][indices], selected[:, column]):
            raise ValueError(f"Population rows differ in {field}")
    return np.ascontiguousarray(raw[indices], dtype="<f4")


def rebuild_population(archive, directory):
    from population_core import (MonthlyTraffic, START, build_normal_pauses, match_all_sources,
                                 prepare_records, select_all)

    dictionary = json.loads((ROOT / "description_to_type.json").read_text(encoding="utf-8"))
    nodes = load(archive / "node_order.npy")
    records, stations, coverage, _, _ = prepare_records(archive, dictionary, nodes)
    stations = stations.sort_values("station_id", kind="stable").reset_index(drop=True)
    foundations, index, _, _ = match_all_sources(records, stations, include_previous_targets=True)
    matched = foundations.loc[foundations.event_start >= START].reset_index(drop=True)
    pause_index, _ = build_normal_pauses(records, index, foundations)
    traffic = MonthlyTraffic(archive / "year_2023/year_2023", nodes)
    try:
        select_all(traffic, matched, index, coverage, stations, 120, directory, pause_index)
    finally:
        traffic.close()


def fit_scaler(raw):
    total = np.zeros(3, dtype=np.float64)
    square = np.zeros(3, dtype=np.float64)
    for start in range(0, len(raw), CHUNK_ROWS):
        block = np.asarray(raw[start:start + CHUNK_ROWS], dtype=np.float64, order="C")
        total += block.sum(axis=(0, 1))
        square += np.square(block).sum(axis=(0, 1))
    points = len(raw) * raw.shape[1]
    mean = total / points
    std = np.sqrt(np.maximum(square / points - mean * mean, 0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid training means or standard deviations")
    return mean, std


def export_dataset(directory, minutes, raw_parts, selections):
    points = minutes // 5
    directory.mkdir()
    mean, std = fit_scaler(raw_parts["Train"][:, :points])
    scaler = dict(method="global_per_channel_zscore", fit_split="Train", ddof=0,
                  channel_order=CHANNELS, mean=mean.tolist(), std=std.tolist())
    (directory / "scaler.json").write_text(json.dumps(scaler, indent=2) + "\n", encoding="utf-8")
    for split, raw in raw_parts.items():
        folder = directory / split
        folder.mkdir()
        shape = (len(raw), points, 3)
        values = np.lib.format.open_memmap(folder / FEATURES, mode="w+", dtype="<f4", shape=shape)
        for start in range(0, len(raw), CHUNK_ROWS):
            block = np.asarray(raw[start:start + CHUNK_ROWS, :points], dtype=np.float64, order="C")
            values[start:start + len(block)] = ((block - mean) / std).astype("<f4")
        values.flush()
        del values
        np.save(folder / "labels.npy", selections[split][:, 3].astype("<i8"), allow_pickle=False)
        print(f"Saved {directory.name}/{split}: {shape}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Raw XTraffic v8 directory.")
    source.add_argument("--population", type=Path, help="Existing common/120min population directory.")
    parser.add_argument("--rebuild-population", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "Data_rebuilt")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if args.rebuild_population:
        scratch = args.rebuild_population.resolve()
        if args.population or scratch.exists() or scratch.is_relative_to(output) or output.is_relative_to(scratch):
            parser.error("Use a raw archive and separate new population/output directories.")
        rebuild_population(args.archive, scratch)
        args.population = scratch
    parts = load_selections()
    selected = np.concatenate(list(parts.values()))
    raw = read_population(args.population, selected) if args.population else read_monthly(args.archive, selected)
    sizes = np.cumsum([len(part) for part in parts.values()])
    raw_parts = dict(zip(parts, np.split(raw, sizes[:-1])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".catid-", dir=output.parent) as temporary:
        release = Path(temporary) / "release"
        release.mkdir()
        for minutes in (90, 120):
            export_dataset(release / f"CA-TID-{minutes}", minutes, raw_parts, parts)
        release.rename(output)
    print(f"Saved both datasets to {output}", flush=True)


if __name__ == "__main__":
    main()
