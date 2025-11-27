#!/usr/bin/env python3
"""Evaluate discovered models (Alpha, Heuristics, Inductive)

Produces `outputs/model_metrics.csv` and `outputs/model_summary.md`.
"""
import os
import csv
import argparse
import pandas as pd
import time

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils

# miners
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter

# conformance / evaluation (try to import optional evaluators)
try:
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
except Exception:
    alignments = None

try:
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
except Exception:
    precision_evaluator = None


def detect_separator(path, default=','):
    try:
        import csv as _csv
        with open(path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            if not sample:
                return default
            sniffer = _csv.Sniffer()
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
    except Exception:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                header = f.readline()
                if ';' in header and header.count(';') > header.count(','):
                    return ';'
        except Exception:
            pass
        return default


def load_and_prepare(csv_path, sep=None):
    if sep is None:
        sep = detect_separator(csv_path)
    df = pd.read_csv(csv_path, sep=sep)

    # Prepare for pm4py
    df = df.rename(columns={
        'STUDIENR': 'case:concept:name',
        'KURSTXT': 'concept:name',
        'SEMESTER_END': 'time:timestamp'
    })
    if 'time:timestamp' in df.columns:
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')

    df = df.sort_values(['case:concept:name'] + (['time:timestamp'] if 'time:timestamp' in df.columns else []))
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    log = log_converter.apply(df)
    return df, log


def run_alpha(log):
    t0 = time.time()
    net, im, fm = alpha_miner.apply(log)
    # Manually construct initial and final markings if missing
    def manual_initial_marking(net):
        return {p: 1 for p in net.places if not any(a.target == p for a in net.arcs)}
    def manual_final_marking(net):
        return {p: 1 for p in net.places if not any(a.source == p for a in net.arcs)}
    if im is None:
        im = manual_initial_marking(net)
    if fm is None:
        fm = manual_final_marking(net)
    t1 = time.time()
    return {'net': net, 'im': im, 'fm': fm, 'time': t1 - t0}


def run_heuristics(log, dep_thresh=0.8, and_thresh=0.65, loop_two=0.5):
    t0 = time.time()
    res = heuristics_miner.apply(
        log,
        parameters={
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: dep_thresh,
            heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: and_thresh,
            heuristics_miner.Variants.CLASSIC.value.Parameters.LOOP_LENGTH_TWO_THRESH: loop_two
        }
    )
    t1 = time.time()
    # res may be heuristics net or (heuristics_net, dep_matrix, dfg)
    heur_net = res[0] if isinstance(res, tuple) else res
    print("Heuristics net type:", type(heur_net))
    print("Heuristics net object:", heur_net)
    # If heur_net is already a Petri net, use it directly
    # Manually construct initial and final markings if missing
    def manual_initial_marking(net):
        return {p: 1 for p in net.places if not any(a.target == p for a in net.arcs)}
    def manual_final_marking(net):
        return {p: 1 for p in net.places if not any(a.source == p for a in net.arcs)}
    if hasattr(heur_net, 'places') and hasattr(heur_net, 'transitions'):
        net = heur_net
        im = manual_initial_marking(net)
        fm = manual_final_marking(net)
    else:
        try:
            net, im, fm = hn_converter.apply(heur_net)
            if im is None:
                im = manual_initial_marking(net)
            if fm is None:
                fm = manual_final_marking(net)
        except Exception as e:
            print("Heuristics net to Petri net conversion failed:", e)
            net, im, fm = (None, None, None)
    # If Petri net conversion fails, fallback to DFG metrics if available
    metrics = {}
    if net is None and isinstance(res, tuple) and len(res) > 2:
        dfg = res[2]
        metrics['dfg_edges'] = len(dfg) if dfg else None
    return {'heur_net': heur_net, 'net': net, 'im': im, 'fm': fm, 'time': t1 - t0, **metrics}


def run_inductive(log, noise=0.0):
    t0 = time.time()
    pt = inductive_miner.apply(log, variant=inductive_miner.Variants.IMf, parameters={'noise_threshold': noise})
    t1 = time.time()
    # Manually construct initial and final markings if missing
    def manual_initial_marking(net):
        return {p: 1 for p in net.places if not any(a.target == p for a in net.arcs)}
    def manual_final_marking(net):
        return {p: 1 for p in net.places if not any(a.source == p for a in net.arcs)}
    try:
        net, im, fm = pt_converter.apply(pt)
        if im is None:
            im = manual_initial_marking(net)
        if fm is None:
            fm = manual_final_marking(net)
    except Exception:
        net, im, fm = (None, None, None)
    return {'pt': pt, 'net': net, 'im': im, 'fm': fm, 'time': t1 - t0}


def compute_basic_metrics(net):
    if net is None:
        return {'places': None, 'transitions': None, 'arcs': None, 'size': None}
    places = len(net.places)
    transitions = len(net.transitions)
    arcs = len(net.arcs) if hasattr(net, 'arcs') else None
    size = places + transitions
    return {'places': places, 'transitions': transitions, 'arcs': arcs, 'size': size}


def try_alignments(event_log, net, im, fm):
    if alignments is None:
        return {'replay_fitness': None}
    try:
        aligned = alignments.apply_log(event_log, net, im, fm)
        # alignments.apply_log may return a list; statistics depend on pm4py version
        # We'll attempt to compute an approximate fitness: count fitting traces
        if isinstance(aligned, dict):
            # some versions return a dict with 'averageFitness'
            return {'replay_fitness': aligned.get('averageFitness', None)}
        elif isinstance(aligned, list):
            # list of alignment results -- compute fraction of traces with empty cost
            fitted = 0
            total = len(aligned)
            for a in aligned:
                # a is a tuple or dict depending on version
                if isinstance(a, dict):
                    fitness = a.get('fitness')
                    if fitness is not None and fitness >= 1.0:
                        fitted += 1
                elif isinstance(a, tuple) or isinstance(a, list):
                    # heuristic: if last element cost == 0
                    try:
                        cost = a[-1]
                        if cost == 0:
                            fitted += 1
                    except Exception:
                        pass
            return {'replay_fitness': fitted / total if total > 0 else None}
        else:
            return {'replay_fitness': None}
    except Exception:
        return {'replay_fitness': None}


def try_precision(event_log, net, im, fm):
    if precision_evaluator is None:
        return {'precision': None}
    try:
        prec = precision_evaluator.apply(event_log, net, im, fm)
        return {'precision': prec}
    except Exception:
        return {'precision': None}


def save_metrics_csv(rows, out_path):
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def save_summary_md(rows, out_path):
    lines = ["# Model evaluation summary\n"]
    for r in rows:
        lines.append(f"## {r['model']}\n")
        lines.append(f"- Time (s): {r.get('time')}\n")
        lines.append(f"- Places: {r.get('places')}\n")
        lines.append(f"- Transitions: {r.get('transitions')}\n")
        lines.append(f"- Arcs: {r.get('arcs')}\n")
        lines.append(f"- Size (places+trans): {r.get('size')}\n")
        lines.append(f"- Replay fitness: {r.get('replay_fitness')}\n")
        lines.append(f"- Precision: {r.get('precision')}\n")
        lines.append("\n")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([l if l.endswith('\n') else l + '\n' for l in lines])


def build_argparser():
    p = argparse.ArgumentParser(description='Evaluate Alpha/Heuristics/Inductive miners')
    p.add_argument('--input', '-i', default='DTU_Curricula_Data_Filtered.csv')
    p.add_argument('--out', '-o', default='outputs')
    p.add_argument('--sample', type=int, default=None)
    p.add_argument('--top-activities', type=int, default=None)
    p.add_argument('--noise', type=float, default=0.0)
    # Heuristics Miner thresholds (dependency/AND/loop-two)
    p.add_argument('--heur-dependency', type=float, default=0.8, help='Dependency threshold for Heuristics Miner')
    p.add_argument('--heur-and', type=float, default=0.65, help='AND measure threshold for Heuristics Miner')
    p.add_argument('--heur-loop-two', type=float, default=0.5, help='Loop-two threshold for Heuristics Miner')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df, log = load_and_prepare(args.input)

    # optional sampling and filtering
    if args.sample is not None:
        unique = df['case:concept:name'].unique()
        if len(unique) > args.sample:
            sampled = pd.Series(unique).sample(n=args.sample, random_state=1).tolist()
            df = df[df['case:concept:name'].isin(sampled)]
            log = log_converter.apply(df)

    if args.top_activities is not None:
        before = df['concept:name'].nunique()
        top = df['concept:name'].value_counts().nlargest(args.top_activities).index.tolist()
        df = df[df['concept:name'].isin(top)]
        log = log_converter.apply(df)
        after = df['concept:name'].nunique()
        print(f"Filtered activities: kept top {after} of {before} activities by frequency.")

    results = []

    print('\nRunning Alpha Miner...')
    alpha_res = run_alpha(log)
    metrics = compute_basic_metrics(alpha_res['net'])
    align = try_alignments(log, alpha_res['net'], alpha_res['im'], alpha_res['fm'])
    prec = try_precision(log, alpha_res['net'], alpha_res['im'], alpha_res['fm'])
    row = {'model': 'alpha', 'time': alpha_res['time'], **metrics, **align, **prec}
    results.append(row)

    print('\nRunning Heuristics Miner...')
    heur_res = run_heuristics(
        log,
        dep_thresh=args.heur_dependency,
        and_thresh=args.heur_and,
        loop_two=args.heur_loop_two,
    )
    metrics = compute_basic_metrics(heur_res['net'])
    # If Petri net conversion failed, add DFG metrics if available
    if metrics['places'] is None and 'dfg_edges' in heur_res:
        metrics['dfg_edges'] = heur_res['dfg_edges']
    align = try_alignments(log, heur_res['net'], heur_res['im'], heur_res['fm'])
    prec = try_precision(log, heur_res['net'], heur_res['im'], heur_res['fm'])
    row = {'model': 'heuristics', 'time': heur_res['time'], **metrics, **align, **prec}
    results.append(row)

    print('\nRunning Inductive Miner...')
    ind_res = run_inductive(log, noise=args.noise)
    metrics = compute_basic_metrics(ind_res['net'])
    align = try_alignments(log, ind_res['net'], ind_res['im'], ind_res['fm'])
    prec = try_precision(log, ind_res['net'], ind_res['im'], ind_res['fm'])
    row = {'model': 'inductive', 'time': ind_res['time'], **metrics, **align, **prec}
    results.append(row)

    csv_path = os.path.join(args.out, 'model_metrics.csv')
    md_path = os.path.join(args.out, 'model_summary.md')
    save_metrics_csv(results, csv_path)
    save_summary_md(results, md_path)

    print(f"\nSaved metrics CSV to: {csv_path}")
    print(f"Saved summary MD to: {md_path}")


if __name__ == '__main__':
    main()
