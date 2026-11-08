#!/usr/bin/env python3
"""Archived XTraffic candidate-selection rules used to rebuild the 120-minute population."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
import sqlite3

import numpy as np
import pandas as pd


START, END = pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01')
STEP = pd.Timedelta(minutes=5)
ZONE = 'America/Los_Angeles'
LABELS = {'hazard': 0, 'accident': 1, 'normal': 2}



def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def read_table(path, **kwargs):
    """Detect the actual delimiter; do not trust a .csv filename."""
    with Path(path).open(encoding='utf-8-sig') as handle:
        header = handle.readline()
    sep = '\t' if header.count('\t') > header.count(',') else ','
    table = pd.read_csv(path, sep=sep, low_memory=False, **kwargs)
    return table, sep


def parse_times(values):
    """Explicit US dates with 2/4-digit years; ambiguous day/month is not guessed."""
    text = values.astype('string').str.strip()
    out = pd.Series(pd.NaT, index=values.index, dtype='datetime64[ns]')
    for fmt in ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M',
                '%m/%d/%y %H:%M:%S', '%m/%d/%y %H:%M'):
        missing = out.isna()
        out.loc[missing] = pd.to_datetime(text.loc[missing], format=fmt, errors='coerce')
    return out


def haversine(lat, lon, lat2, lon2):
    a, b, c, d = map(np.radians, (lat, lon, lat2, lon2))
    v = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(v, 0, 1)))



class MonthlyTraffic:
    """Read-only monthly maps; each row is bound to an explicit civil-time index.

    The spring-DST hour is an explicit unavailable interval, never an elapsed
    UTC slot. No array concatenation, interpolation or padding is performed.
    """
    def __init__(self, folder, nodes, output=None):
        self.nodes = nodes
        self.arrays, self.starts, self.ends, self.layouts = {}, {}, {}, {}
        self.audit = []
        for m in range(1, 13):
            path = folder / f'2023_p{m:02}.npy'
            source = np.load(path, mmap_mode='r', allow_pickle=False)
            if source.ndim != 3 or source.shape[-1] != 3:
                raise ValueError(f'{path}: not a 3-channel array')
            if source.shape[0] == len(nodes) and source.shape[1] != len(nodes):
                arr, layout = source.transpose(1, 0, 2), 'node-time-channel'
            elif source.shape[1] == len(nodes) and source.shape[0] != len(nodes):
                arr, layout = source, 'time-node-channel'
            else:
                raise ValueError(f'{path}: ambiguous or incorrect node axis')
            start = pd.Timestamp(2023, m, 1)
            end = start + pd.offsets.MonthBegin(1)
            civil = pd.date_range(start, end, freq='5min', inclusive='left')
            if len(arr) != len(civil):
                raise ValueError(f'{path}: monthly time length disagrees with civil grid')
            # Check every index against the geographical timezone. This detects
            # the nonexistent local hour without interpreting array length as proof.
            zoned = civil.tz_localize(ZONE, ambiguous='NaT', nonexistent='NaT')
            unavailable = zoned.isna()
            nonexistent = civil.tz_localize(ZONE, ambiguous=True, nonexistent='NaT').isna()
            ambiguous = unavailable & ~nonexistent
            if nonexistent.any() and np.isfinite(arr[nonexistent]).any():
                raise ValueError('DST-missing civil slots contain data: require timestamp sidecar before building')
            self.arrays[m], self.starts[m], self.ends[m] = arr, start, end
            self.layouts[m] = layout
            if output is not None:
                pd.DataFrame({'time_index': np.arange(len(civil)), 'source_local_timestamp': civil,
                              'valid_local_time': ~unavailable, 'nonexistent': nonexistent, 'ambiguous': ambiguous}).to_csv(output / f'time_index_2023_p{m:02}.csv', index=False)
            self.audit.append({'file': path.name, 'source_shape': list(source.shape),
                'time_node_channel_shape': list(arr.shape), 'layout': layout, 'dtype': str(arr.dtype),
                'first_civil_timestamp': str(civil[0]), 'last_civil_timestamp': str(civil[-1]),
                'nonexistent_local_slots': int(nonexistent.sum()), 'ambiguous_local_slots': int(ambiguous.sum()), 'ambiguous_finite_values_excluded': int(np.isfinite(arr[ambiguous]).sum()),
                'nonexistent_slots_all_nonfinite': bool(not nonexistent.any() or not np.isfinite(arr[nonexistent]).any()),
                'index_basis': 'Explicit monthly local civil grid, corroborated by upstream naive date_range and spring-DST missing-slot pattern'})
        self.unavailable_intervals = [(pd.Timestamp('2023-03-12 02:00'), pd.Timestamp('2023-03-12 03:00')), (pd.Timestamp('2023-11-05 01:00'), pd.Timestamp('2023-11-05 02:00'))]

    def extract(self, node, start, points):
        end = start + points * STEP
        if start < START or end > END:
            return None, 'outside_full_year', []
        if any(start < b and end > a for a, b in self.unavailable_intervals):
            return None, 'dst_unavailable_or_ambiguous_observation_support', []
        if start.value % STEP.value:
            return None, 'off_five_minute_grid', []
        blocks, segments, cursor = [], [], start
        while cursor < end:
            m = cursor.month
            if m not in self.arrays:
                return None, 'missing_required_month', segments
            stop = min(end, self.ends[m])
            first = int((cursor - self.starts[m]) / STEP)
            count = int((stop - cursor) / STEP)
            block = self.arrays[m][first:first+count, node, :]
            if block.shape != (count, 3):
                return None, 'incomplete_window', segments
            blocks.append(block[:, [2, 1, 0]])
            segments.append({'month': m, 'first_index': first, 'points': count})
            cursor = stop
        values = np.concatenate(blocks, axis=0).astype(np.float32)
        if values.shape != (points, 3):
            return None, 'incomplete_window', segments
        if not np.isfinite(values).all():
            return None, 'nonfinite_traffic_window', segments
        return values, None, segments

    def close(self):
        for array in self.arrays.values():
            if isinstance(array, np.memmap):
                array._mmap.close()
        self.arrays.clear()


def prepare_records(archive, dictionary, nodes):
    frames, coverage, csv_info = [], [], []
    for year in (2022, 2023):
        path = archive / f'incidents_y{year}.csv'
        frame, sep = read_table(path, dtype={'incident_id': 'string', 'Abs PM': 'string'})
        frame['source_postmile_text'] = frame['Abs PM'].str.strip()
        frame['source_file'], frame['source_row'] = path.name, np.arange(len(frame))
        frame['incident_id'] = frame.incident_id.str.strip()
        frame['parsed_start'] = parse_times(frame.dt)
        frames.append(frame)
        # Published calendar-year file coverage plus conservative earliest-record
        # lower boundary; absence of reports still does not establish completeness.
        coverage.append((max(pd.Timestamp(year, 1, 1), frame.parsed_start.min()), pd.Timestamp(year+1, 1, 1)))
        csv_info.append({'file': path.name, 'delimiter': sep, 'rows': len(frame),
                         'first_record_time': str(frame.parsed_start.min()),
                         'last_record_time': str(frame.parsed_start.max())})
    records = pd.concat(frames, ignore_index=True)
    records['record_key'] = records.source_file + ':' + records.source_row.astype(str)
    records['mapped_type'] = records.DESCRIPTION.map(dictionary).fillna('unmapped')
    records['direction'] = records.Freeway_direction.astype('string').str.strip().str.upper()
    for col in ('duration', 'Fwy', 'Abs PM', 'Latitude', 'Longitude'):
        records[col] = pd.to_numeric(records[col], errors='coerce').astype('float64')
    local = records.parsed_start.dt.tz_localize(ZONE, ambiguous='NaT', nonexistent='NaT')
    duration_valid = np.isfinite(records.duration) & (records.duration >= 0)
    # Duration is elapsed minutes. Localise only to calculate an end correctly
    # across DST; published event and array timestamps remain local civil times.
    safe_duration = records.duration.where(duration_valid, 0)
    end_local = (local + pd.to_timedelta(safe_duration, unit='min')).dt.tz_localize(None)
    records['known_interval'] = duration_valid & local.notna() & end_local.notna()
    records['start_time_valid'] = local.notna()
    records['block_start'] = records.parsed_start.where(records.start_time_valid)
    records['block_end'] = end_local.where(records.known_interval)
    records['duplicate_incident_id'] = records.incident_id.duplicated(False) | records.incident_id.isna()
    # Unknown intervals in 2022 may still extend into 2023. Keep all such records.
    # Keep all previous-year complete records: they can reset a pause in 2022.
    # Unknown ends have no fabricated background end, and invalid starts have
    # no fabricated full-year interval. Both remain reviewable source records.
    records = records.reset_index(drop=True)
    records['record_position'] = np.arange(len(records))
    stations, sep = read_table(archive / 'sensor_meta_feature.csv', dtype={'Abs PM': 'string'})
    stations['sensor_postmile_text'] = stations['Abs PM'].str.strip()
    stations['direction'] = stations.Direction.astype('string').str.strip().str.upper()
    for col in ('station_id', 'Fwy', 'Abs PM', 'Lat', 'Lng'):
        stations[col] = pd.to_numeric(stations[col], errors='coerce').astype('float64')
    if stations.station_id.duplicated().any() or len(np.unique(nodes)) != len(nodes):
        raise ValueError('Duplicate station IDs or node IDs')
    lookup = {int(s): i for i, s in enumerate(nodes)}
    stations['node_index'] = stations.station_id.map(lookup)
    valid = stations.node_index.notna() & np.isfinite(stations.Fwy) & np.isfinite(stations['Abs PM'])
    excluded_station_count = int((~valid).sum())
    stations = stations.loc[valid].copy()
    stations[['station_id', 'node_index']] = stations[['station_id', 'node_index']].astype(np.int64)
    review = records.loc[(records.mapped_type == 'unmapped') | ~records.known_interval
                         | records.duplicate_incident_id].copy()
    review['review_reason'] = [';'.join(r for r, flag in (
        ('unmapped_description', t == 'unmapped'), ('unknown_or_invalid_interval', not k),
        ('missing_or_duplicate_id', d)) if flag)
        for t, k, d in zip(review.mapped_type, review.known_interval, review.duplicate_incident_id)]
    summary = {'source_files': csv_info, 'metadata_delimiter': sep,
        'relevant_background_rows': len(records), 'unknown_interval_rows': int((~records.known_interval).sum()),
        'previous_year_background_rows': int((records.source_file == 'incidents_y2022.csv').sum()),
        'usable_sensors': len(stations), 'excluded_metadata_rows': excluded_station_count,
        'raw_type_is_not_used_as_label': True,
        'normal_coverage_intervals': [[str(a), str(b)] for a, b in coverage]}
    return records, stations, coverage, review, summary


def spatial_match(records, stations, pm_limit, geo_limit):
    """Public compatibility entry point for the exact full-year matcher."""
    return match_all_sources(records, stations, pm_limit, geo_limit)


def overlaps(index, station, start, end, except_record=None):
    if station not in index:
        return np.empty(0, dtype=np.int64), False
    starts, ends, ids, unknown = index[station]
    stop = np.searchsorted(starts, end.value, side='left')
    active = (ends[:stop] >= start.value) & ~unknown[:stop]
    if except_record is not None:
        active &= ids[:stop] != except_record
    return ids[:stop][active], False


def covered(start, end, intervals):
    cursor = start
    for a, b in sorted(intervals):
        if a <= cursor < b:
            cursor = max(cursor, b)
        if cursor >= end:
            return True
    return False


SLOTS = int((END - START) / STEP)
WINDOWS = (60, 90, 120)
META_DTYPE = np.dtype([('node_index', '<i4'), ('station_id', '<i4'),
    ('start_index', '<i4'), ('label', 'i1'), ('event_index', '<i4')])
CODES = {0: 'selected', 1: 'outside_full_year', 2: 'incident_source_coverage_gap',
    3: 'dst_unavailable_or_ambiguous_support', 4: 'normal_paused_pending_qualified_event',
    5: 'background_incident_overlap', 6: 'nonfinite_traffic_window',
    7: 'same_sensor_spacing_to_retained_event_or_selected_normal',
    8: 'exact_feature_duplicate_with_retained_sample'}


def canonical_bytes(values):
    out = np.asarray(values, dtype='<f4').copy()
    out[out == 0] = 0  # np.unique treats signed zero as numerically equal.
    return out.tobytes()


def support_conflict(start, end, intervals):
    return any(start < b and end > a for a, b in intervals)


def ranges_for_overlaps(starts, ends, points, closed_end=True):
    """Integer starts whose full support intersects each source interval.

    Incident ends use conservative contact (>=). DST ends are half-open (>).
    Non-grid incident times are handled by floor division of nanoseconds.
    """
    starts, ends = np.asarray(starts, np.int64), np.asarray(ends, np.int64)
    lo = (starts - START.value) // STEP.value - points + 1
    hi = ((ends - START.value) // STEP.value + 1 if closed_end else
          -(-(ends - START.value) // STEP.value))
    return np.clip(lo, 0, SLOTS), np.clip(hi, 0, SLOTS)


def interval_mask(starts, ends, points, closed_end=True):
    difference = np.zeros(SLOTS + 1, np.int32)
    lo, hi = ranges_for_overlaps(starts, ends, points, closed_end)
    valid = hi > lo
    np.add.at(difference, lo[valid], 1)
    np.add.at(difference, hi[valid], -1)
    return np.cumsum(difference[:-1]) > 0


def normal_stage_codes(station, index, coverage, values, points, unavailable_intervals, pause_index=None):
    """Exhaustive station x civil-grid first-failure ledger, before scheduling.

    Every start index including end-of-year overflow receives exactly one code.
    NumPy interval differences and cumulative finite counts evaluate ALL starts;
    this does not sample or skip proposals.
    """
    code = np.zeros(SLOTS, np.uint8)
    code[SLOTS-points+1:] = 1
    covered_mask = np.zeros(SLOTS, bool)
    for a, b in coverage:
        lo = max(0, int(-(-(a.value-START.value) // STEP.value)))
        hi = min(SLOTS, int((b.value-START.value) // STEP.value)-points+1)
        if hi > lo:
            covered_mask[lo:hi] = True
    code[(code == 0) & ~covered_mask] = 2
    a = np.asarray([x.value for x, _ in unavailable_intervals], np.int64)
    b = np.asarray([x.value for _, x in unavailable_intervals], np.int64)
    code[(code == 0) & interval_mask(a, b, points, False)] = 3
    if pause_index and station in pause_index:
        a, b = pause_index[station]
        code[(code == 0) & interval_mask(a, b, points, False)] = 4
    if station in index:
        starts, ends, _, unknown = index[station]
        code[(code == 0) & interval_mask(starts[~unknown], ends[~unknown], points)] = 5
    finite = np.isfinite(values).all(axis=1)
    cumulative = np.concatenate(([0], np.cumsum(~finite, dtype=np.int32)))
    valid_window = np.zeros(SLOTS, bool)
    valid_window[:SLOTS-points+1] = cumulative[points:] == cumulative[:-points]
    code[(code == 0) & ~valid_window] = 6
    return code


def match_all_sources(records, stations, pm_limit=.5, geo_limit=1., include_previous_targets=False):
    """Exact source-decimal half-mile inclusion and postmile priority.

    Binary floating-point subtraction and endpoint arithmetic can respectively
    reject 4.4-3.9 or 3.9>=4.4-.5. Float search ONLY enumerates a safe superset;
    every final background/target inclusion and postmile sort uses original CSV
    Decimal values. The geographic threshold is unchanged.
    """
    decimal_limit = Decimal(str(pm_limit))
    roads = {}
    for road, part in stations.groupby('Fwy', sort=True):
        p = part.sort_values(['Abs PM', 'station_id'], kind='stable')
        roads[road] = (p['Abs PM'].to_numpy(), p.station_id.to_numpy(),
            p.node_index.to_numpy(), p.direction.to_numpy(), p.Lat.to_numpy(), p.Lng.to_numpy(),
            [Decimal(text) for text in p.sensor_postmile_text], p.sensor_postmile_text.to_numpy())
    links, matches, rejected, unlocated = defaultdict(list), [], [], []
    cols = ['record_position', 'record_key', 'incident_id', 'source_file', 'source_row',
        'mapped_type', 'parsed_start', 'block_end', 'duration', 'Fwy', 'Abs PM',
        'direction', 'Latitude', 'Longitude', 'known_interval', 'duplicate_incident_id', 'DESCRIPTION', 'source_postmile_text']
    for row in records[cols].itertuples(index=False, name=None):
        (pos, key, incident, source, source_row, kind, start, end, duration, road, pm,
         direction, lat, lon, known, duplicated, description, pm_text) = row
        lower_year = pd.Timestamp('2022-01-01') if include_previous_targets else START
        target = kind in ('hazard', 'accident') and pd.notna(start) and lower_year <= start < END
        reason = None
        if not np.isfinite(road) or not np.isfinite(pm) or road not in roads:
            unlocated.append({'record_position': pos, 'record_key': key,
                              'reason': 'no_usable_road_postmile_or_sensor'})
            reason = 'missing_road_postmile_or_sensor'
            nearby = np.empty(0, np.int64)
        else:
            pms, ids, nodes, directions, lats, lons, decimal_pms, sensor_pm_texts = roads[road]
            event_pm = Decimal(pm_text)
            # Round each exact Decimal endpoint outwards by one binary ULP.
            # This is a candidate enumeration bound, never the final threshold.
            lower = np.nextafter(float(event_pm-decimal_limit), -np.inf)
            upper = np.nextafter(float(event_pm+decimal_limit), np.inf)
            lo = np.searchsorted(pms, lower, side='left')
            hi = np.searchsorted(pms, upper, side='right')
            nearby = np.asarray([candidate for candidate in range(lo, hi)
                if abs(decimal_pms[candidate]-event_pm) <= decimal_limit], dtype=np.int64)
            for station in ids[nearby]:
                links[int(station)].append(pos)
        if not target:
            continue
        if reason is None and not known:
            reason = 'unknown_or_invalid_interval'
        if reason is None and duplicated:
            reason = 'missing_or_duplicate_incident_id'
        if reason is None and (pd.isna(direction) or direction not in ('N', 'S', 'E', 'W')):
            reason = 'missing_or_invalid_direction'
        if reason is None:
            candidates = nearby[directions[nearby] == direction]
            if not len(candidates):
                reason = 'no_same_direction_postmile_candidate'
        if reason is None and (not np.isfinite(lat) or not np.isfinite(lon) or
                               not (-90 <= lat <= 90 and -180 <= lon <= 180)):
            reason = 'missing_or_invalid_coordinates'
        if reason is None:
            geo = haversine(lat, lon, lats[candidates], lons[candidates])
            valid = (np.isfinite(geo) & (np.abs(lats[candidates]) <= 90) &
                     (np.abs(lons[candidates]) <= 180) & (geo <= geo_limit))
            candidates, geo = candidates[valid], geo[valid]
            if not len(candidates):
                reason = 'no_candidate_within_geographical_threshold'
        if reason is not None:
            rejected.append({'record_position': pos, 'record_key': key,
                'incident_id': incident, 'event_type': kind, 'reason': reason})
            continue
        decimal_distance = [abs(decimal_pms[candidate]-event_pm) for candidate in candidates]
        chosen_order = sorted(range(len(candidates)), key=lambda i: (decimal_distance[i], geo[i], ids[candidates[i]]))
        choice, chosen = chosen_order[0], candidates[chosen_order[0]]
        matches.append({'record_position': pos, 'record_key': key,
            'source_incident_id': incident, 'source_file': source, 'source_row': source_row,
            'event_type': kind, 'event_start': start, 'event_end': end, 'duration_minutes': duration,
            'station_id': int(ids[chosen]), 'node_index': int(nodes[chosen]),
            'postmile_distance': float(decimal_distance[choice]),
            'postmile_distance_decimal': str(decimal_distance[choice].normalize()),
            'source_postmile_text': pm_text, 'sensor_postmile_text': sensor_pm_texts[chosen],
            'geo_distance_km': float(geo[choice]),
            'candidate_count': len(candidates), 'description': description,
            'source_fwy': int(road), 'source_direction': direction})
    starts, ends = records.block_start.astype('int64').to_numpy(), records.block_end.astype('int64').to_numpy()
    unknown, result = ~records.known_interval.to_numpy(), {}
    for station, positions in links.items():
        ids = np.asarray(positions, np.int64)
        ids = ids[np.argsort(starts[ids], kind='stable')]
        result[station] = (starts[ids], ends[ids], ids, unknown[ids])
    return pd.DataFrame(matches), result, pd.DataFrame(rejected), pd.DataFrame(unlocated)


def build_normal_pauses(records, index, foundations):
    """Per-sensor next strictly later source start, before window selection.

    Foundations already passed original target matching and record checks.
    Simultaneous starts are ordered by source ID and row key. Pause intervals
    are half-open through the strictly-next-five-minute recovery grid.
    """
    eligible = np.zeros(len(records), bool)
    if not foundations.empty:
        eligible[foundations.record_position.to_numpy(np.int64)] = True
    source_starts = records.block_start.astype('int64').to_numpy()
    source_ends = records.block_end.astype('int64').to_numpy()
    valid_starts = records.start_time_valid.to_numpy(bool)
    ids = records.incident_id.fillna('').astype(str).to_numpy()
    keys = records.record_key.to_numpy()
    rows, union = [], {}
    for station in sorted(index):
        _, _, positions, unknown = index[station]
        resets = [int(pos) for pos in positions if eligible[pos]]
        resets.sort(key=lambda pos: (source_starts[pos], ids[pos], keys[pos]))
        reset_starts = np.asarray([source_starts[pos] for pos in resets], np.int64)
        intervals = []
        for pos in positions[unknown & valid_starts[positions]]:
            first = int(source_starts[pos])
            next_index = int(np.searchsorted(reset_starts, first, side='right'))
            reset = resets[next_index] if next_index < len(resets) else None
            resume = ((int(source_ends[reset]) // STEP.value + 1) * STEP.value
                      if reset is not None else END.value)
            if resume <= first:
                raise ValueError('Recovery must follow the pause source start')
            row = {'source_record_key': keys[pos], 'source_incident_id': ids[pos],
                'record_position': int(pos), 'station_id': int(station),
                'pause_start': str(pd.Timestamp(first)),
                'reset_record_key': keys[reset] if reset is not None else '',
                'reset_event_id': ids[reset] if reset is not None else '',
                'reset_event_start': str(pd.Timestamp(source_starts[reset])) if reset is not None else '',
                'reset_event_end': str(pd.Timestamp(source_ends[reset])) if reset is not None else '',
                'resume_grid_start': str(pd.Timestamp(resume)) if reset is not None else '',
                'pause_end_exclusive': str(pd.Timestamp(min(resume, END.value))),
                'never_resumed_in_observation_range': bool(resume >= END.value)}
            for minutes in WINDOWS:
                lo, hi = ranges_for_overlaps([first], [min(resume, END.value)], minutes//5, False)
                # Counts concern full-year candidate starts, before other rules;
                # overlapping source-specific counts must not be summed.
                row[f'affected_normal_candidates_{minutes}min'] = int(max(0, hi[0]-lo[0]))
            rows.append(row)
            if first < END.value and resume > START.value:
                intervals.append((first, min(resume, END.value)))
        merged = []
        for a, b in sorted(intervals):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        if merged:
            union[station] = (np.asarray([a for a, _ in merged], np.int64),
                              np.asarray([b for _, b in merged], np.int64))
    return union, pd.DataFrame(rows)


def select_events(traffic, matched, index, coverage, points):
    """Regenerate every target from raw records, then conflict/dedup/space."""
    decisions, candidates, features = [], [], {}
    for ei, row in matched.iterrows():
        start = pd.Timestamp(row.event_start).floor('5min') - 2 * STEP
        end = start + points * STEP
        ids, unknown = np.empty(0, np.int64), False
        reason = None
        if start < START or end > END:
            reason = 'outside_full_year'
        elif not covered(start, end, coverage):
            reason = 'incident_source_coverage_gap'
        if reason is None:
            ids, unknown = overlaps(index, int(row.station_id), start, end, int(row.record_position))
            if len(ids):
                reason = 'other_incident_overlap'
        if reason is None:
            block, reason, _ = traffic.extract(int(row.node_index), start, points)
        decision = {'event_index': int(ei), 'record_key': row.record_key,
            'source_incident_id': row.source_incident_id, 'station_id': int(row.station_id),
            'node_index': int(row.node_index), 'label': LABELS[row.event_type],
            'window_start': str(start), 'start_index': int((start-START)/STEP),
            'reason': reason or 'selected', 'overlapping_record_count': len(ids),
            'overlapping_record_positions': json.dumps(ids.tolist())}
        decisions.append(decision)
        if reason is None:
            candidates.append(ei)
            features[ei] = block
    order = sorted(candidates, key=lambda i: (decisions[i]['station_id'], decisions[i]['start_index'],
        decisions[i]['label'], str(matched.iloc[i].source_incident_id), str(matched.iloc[i].record_key)))
    key_groups = defaultdict(list)
    for i in order:
        key_groups[(decisions[i]['station_id'], decisions[i]['start_index'])].append(i)
    for group in key_groups.values():
        if len({decisions[i]['label'] for i in group}) > 1:
            for i in group:
                decisions[i]['reason'] = 'same_sensor_window_label_conflict'
        else:
            for i in group[1:]:
                decisions[i]['reason'] = 'same_sensor_window_duplicate'
    feature_groups = defaultdict(list)
    for i in order:
        if decisions[i]['reason'] == 'selected':
            feature_groups[canonical_bytes(features[i])].append(i)
    for group in feature_groups.values():
        if len({decisions[i]['label'] for i in group}) > 1:
            for i in group:
                decisions[i]['reason'] = 'identical_features_label_conflict'
        else:
            for i in group[1:]:
                decisions[i]['reason'] = 'identical_features_same_label_duplicate'
    previous = {}
    for i in order:
        if decisions[i]['reason'] != 'selected':
            continue
        station, start = decisions[i]['station_id'], decisions[i]['start_index']
        if station in previous and start-previous[station] < points:
            decisions[i]['reason'] = 'same_sensor_start_gap_under_minimum'
        else:
            previous[station] = start
    accepted = [i for i in order if decisions[i]['reason'] == 'selected']
    return accepted, features, pd.DataFrame(decisions)


class ExactFeatureRegistry:
    """Bounded-memory disk index. Digest hits are checked against full bytes."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('PRAGMA synchronous=OFF')
        self.db.execute('PRAGMA cache_size=-65536')
        self.db.execute('CREATE TABLE features (digest BLOB PRIMARY KEY, payload BLOB NOT NULL) WITHOUT ROWID')
        self.duplicates, self.collisions = 0, 0

    def add(self, block):
        payload = canonical_bytes(block)
        digest = hashlib.sha256(payload).digest()
        prior = self.db.execute('SELECT payload FROM features WHERE digest=?', (digest,)).fetchone()
        if prior is not None:
            if prior[0] != payload:
                self.collisions += 1
                raise RuntimeError('SHA256 collision: stop rather than silently discard a distinct window')
            self.duplicates += 1
            return False
        self.db.execute('INSERT INTO features VALUES (?,?)', (digest, payload))
        return True

    def close(self):
        count = self.db.execute('SELECT count(*) FROM features').fetchone()[0]
        self.db.commit()
        self.db.close()
        return count


def bin_to_npy(binary, destination, dtype, shape):
    source = np.memmap(binary, dtype=dtype, mode='r', shape=shape)
    target = np.lib.format.open_memmap(destination, mode='w+', dtype=dtype, shape=shape)
    for start in range(0, shape[0], 50000):
        target[start:start+50000] = source[start:start+50000]
    target.flush()
    del source, target


def select_all(traffic, matched, index, coverage, stations, minutes, folder, pause_index=None):
    """Events first, then ALL independent background starts without a quota.

    This retains the current random-normal rebuild's event priority and order of
    spacing-before-normal-feature-dedup. Exhaustive enumeration replaces random
    proposals; historical previous-day keys are eligible on their own merits.
    """
    folder.mkdir(parents=True)
    points = minutes // 5
    event_ids, event_raw, event_decisions = select_events(traffic, matched, index, coverage, points)
    event_decisions.to_csv(folder/'event_candidate_decisions.csv', index=False)
    by_station = defaultdict(list)
    for ei in event_ids:
        by_station[int(matched.iloc[ei].station_id)].append(ei)
    codes = np.lib.format.open_memmap(folder/'normal_candidate_codes.npy', mode='w+',
        dtype=np.uint8, shape=(len(stations), SLOTS))
    stations.to_csv(folder/'candidate_station_order.csv', index=False)
    registry = ExactFeatureRegistry(folder/'feature_index.sqlite')
    for ei in event_ids:
        if not registry.add(event_raw[ei]):
            raise AssertionError('Event global feature deduplication failed')
    raw_binary, meta_binary = folder/'raw.dat', folder/'meta.dat'
    sample_count, normal_count, stage_counts, final_counts = 0, 0, Counter(), Counter()
    station_counts = []
    begun = time.perf_counter()
    with raw_binary.open('wb') as raw_handle, meta_binary.open('wb') as meta_handle:
        for station_row, station in enumerate(stations.itertuples(index=False)):
            node, sid = int(station.node_index), int(station.station_id)
            # Source node slices are contiguous on disk. A single node's entire
            # year occupies 1.26 MB; never concatenate all source nodes.
            values = np.concatenate([traffic.arrays[m][:, node, :] for m in range(1, 13)], axis=0)[:, [2, 1, 0]]
            code = normal_stage_codes(sid, index, coverage, values, points, traffic.unavailable_intervals, pause_index)
            single_counts = np.bincount(code, minlength=len(CODES))
            stage_counts.update({CODES[c]: int(v) for c, v in enumerate(single_counts)})
            events = by_station.get(sid, [])
            event_starts = np.asarray([event_decisions.iloc[ei].start_index for ei in events], np.int32)
            if len(event_starts):
                event_starts.sort()
                diff = np.zeros(SLOTS+1, np.int32)
                np.add.at(diff, np.maximum(0, event_starts-points+1), 1)
                np.add.at(diff, np.minimum(SLOTS, event_starts+points), -1)
                code[(code == 0) & (np.cumsum(diff[:-1]) > 0)] = 7
            starts = np.flatnonzero(code == 0)
            normal_starts, candidate_offset = [], 0
            while candidate_offset < len(starts):
                first = starts[candidate_offset]
                block = values[first:first+points]
                if not registry.add(block):
                    code[first] = 8
                    candidate_offset += 1
                    continue
                normal_starts.append(int(first))
                next_offset = int(np.searchsorted(starts, first+points, side='left'))
                code[starts[candidate_offset+1:next_offset]] = 7
                candidate_offset = next_offset
            codes[station_row] = code
            final_station_counts = np.bincount(code, minlength=len(CODES))
            final_counts.update({CODES[c]: int(v) for c, v in enumerate(final_station_counts)})
            station_counts.append({'station_id': sid, 'node_index': node,
                'single_window_eligible': int(single_counts[0]),
                **{CODES[c]: int(v) for c, v in enumerate(final_station_counts)}})
            entries = [(int(event_decisions.iloc[ei].start_index), LABELS[matched.iloc[ei].event_type], ei) for ei in events]
            entries.extend((first, 2, -1) for first in normal_starts)
            entries.sort()
            if entries:
                metadata = np.empty(len(entries), META_DTYPE)
                raw = np.empty((len(entries), points, 3), np.float32)
                for j, (first, label, ei) in enumerate(entries):
                    metadata[j] = (node, sid, first, label, ei)
                    raw[j] = values[first:first+points]
                raw.tofile(raw_handle)
                metadata.tofile(meta_handle)
                sample_count += len(entries)
                normal_count += len(normal_starts)
            registry.db.commit()
            if (station_row+1) % 500 == 0 or station_row == len(stations)-1:
                print(f'{minutes}min: {station_row+1:,}/{len(stations):,} stations, '
                      f'{normal_count:,} normal, {time.perf_counter()-begun:.1f}s', flush=True)
                codes.flush()
    codes.flush()
    del codes  # release the Windows memory-map handle before later readers
    unique_count = registry.close()
    if unique_count != sample_count:
        raise AssertionError('Global exact-feature index and output count disagree')
    bin_to_npy(raw_binary, folder/'raw_windows.npy', '<f4', (sample_count, points, 3))
    bin_to_npy(meta_binary, folder/'samples.npy', META_DTYPE, (sample_count,))
    raw_binary.unlink()
    meta_binary.unlink()
    # No retained feature index is needed after its complete counts, collision
    # checks and sample audit are recorded. The generator can reconstruct it.
    (folder/'feature_index.sqlite').unlink()
    pd.DataFrame(station_counts).to_csv(folder/'normal_candidate_counts_by_station.csv', index=False)
    report = {'status': 'passed', 'selection_window_minutes': minutes,
        'candidate_grid_shape': [len(stations), SLOTS], 'all_grid_candidates': len(stations)*SLOTS,
        'codes': CODES, 'normal_single_window_stage_counts': dict(stage_counts),
        'normal_final_counts': dict(final_counts), 'normal_samples': normal_count,
        'target_samples': len(event_ids), 'total_samples': sample_count,
        'target_decisions': dict(Counter(event_decisions.reason)),
        'exact_sha256_digest_hits_checked_against_full_float32_bytes': registry.duplicates,
        'sha256_collisions': registry.collisions,
        'selection_order': 'All events: key conflicts/dedup, feature conflicts/dedup, spacing. '
            'All normals: station_id/start ascending; complete single-window rules, retained-event spacing, '
            'normal spacing, exact feature dedup against retained events and normals.',
        'normal_count_cap': None, 'normal_previous_day_pairing': False,
        'partitions_generated': False, 'standardization_generated': False}
    if sum(final_counts.values()) != len(stations)*SLOTS:
        raise AssertionError('Not every normal candidate was accounted for')
    write_json(folder/'selection_report.json', report)
    return report


def sample_ids_for(metadata):
    # Format the finite civil grid once. Repeated starts retain exactly the
    # original pandas representation while avoiding millions of strftime calls.
    starts = metadata['start_index'].astype(np.int64)
    if len(starts) == 0 or ((starts >= 0) & (starts < SLOTS)).all():
        if not hasattr(sample_ids_for, 'civil_stamps'):
            civil_grid = pd.to_datetime(START.value + np.arange(SLOTS, dtype=np.int64)*STEP.value)
            sample_ids_for.civil_stamps = civil_grid.strftime('%Y%m%dT%H%M').to_numpy(str)
        stamp = sample_ids_for.civil_stamps[starts]
    else:
        # Preserve the old function's behaviour for diagnostic out-of-year rows.
        timestamps = pd.to_datetime(START.value + starts*STEP.value)
        stamp = timestamps.strftime('%Y%m%dT%H%M').to_numpy(str)
    station = metadata['station_id'].astype(str)
    ids = np.char.add(np.char.add(station, '_'), stamp)
    event = metadata['label'] != 2
    ids = np.where(event,
        np.char.add(np.char.add('fye_v8_', ids), np.char.add('_', metadata['label'].astype(str))),
        np.char.add('fyn_v8_', ids))
    return ids.astype('S48')
