"""Runs the paper's experiments in order and writes every result to runs/ (or $FEDPROOF_RESULTS). Safe to stop and
restart: finished rows are skipped.

  python3 experiments/run.py prepare --raw /path/to/csvs      # once: datasets -> data/prepared
  python3 experiments/run.py all                              # tune, E1..E9, then report
  python3 experiments/run.py E2 E4                            # selected experiments
  python3 experiments/run.py all --quick                      # a few minutes: checks that everything runs
  python3 experiments/run.py report --results results         # tables from the paper's result files
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # before numpy: the harness forks
    os.environ.setdefault(_v, '1')

import os
import sys
import time
import argparse
if '--results' in sys.argv[:-1]:
    os.environ['FEDPROOF_RESULTS'] = sys.argv[sys.argv.index('--results') + 1]
sys.path.insert(0, '.')
from experiments import common as C

RAW = {   # file names inside --raw, as distributed by the dataset authors
    'edge-iiotset': 'DNN-EdgeIIoT-dataset.csv',
    'wustl-iiot-2021': 'wustl_iiot_2021.csv',
    'x-iiotid': 'X-IIoTID dataset.csv',
}
NATURAL = {}   # dataset -> device or scenario column for a natural split (none of the three datasets has one)


def prepare(raw_dir, quick=False):
    from experiments import prepare_data as D
    os.makedirs(C.DATA, exist_ok=True)
    for ds, fname in RAW.items():
        if ds not in C.DATASETS:
            continue
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            cands = [f for f in os.listdir(raw_dir) if f.lower().endswith('.csv') and ds.split('-')[0][:4] in f.lower().replace('-', '')]
            if not cands:
                print('prepare: %s not found in %s (expected %s)' % (ds, raw_dir, fname)); continue
            path = os.path.join(raw_dir, cands[0])
        splits = ['iid', 'dir1.0', 'dir0.3'] + (['natural:' + NATURAL[ds]] if ds in NATURAL else []) + [C.SCALE['tag']]
        for sp in splits:
            if C.available(ds, sp, 0):
                continue
            if sp == C.SCALE['tag']:                  # the 50-plant variant of the privacy experiment
                base, meta = D.prepare(ds, path, C.DATA, plants=C.SCALE['plants'], per_plant=C.SCALE['per_plant'] if not quick else 1024,
                                       split='iid', tag=sp)
            else:
                base, meta = D.prepare(ds, path, C.DATA, plants=C.PLANTS, per_plant=C.PER_PLANT if not quick else 2048, split=sp)
            print('prepared %s %s: %d plants of %d, %d features, %d classes, %d rows after dedup' % (ds, sp, meta['plants'], meta['per_plant'], meta['n_features'], len(meta['classes']), meta['rows_after_dedup']), flush=True)
            if sp == 'iid':
                print('  label %s; classes %s' % (meta['label_column'], meta['class_counts']))
                print('  dropped %s' % meta['dropped_columns'])
                print('  kept %s' % meta['features_kept'])
                print('  alone separate normal from attack: %s' % (meta['separating_columns'] or 'none'), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='+')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--netns', action='store_true', help='kernel-shaped links for E6 (root, experiments/netns_links.sh)')
    ap.add_argument('--e8-rounds', type=int, default=40)
    ap.add_argument('--results', default=None, help='result folder (default runs/; the paper\'s results are in results/)')
    a = ap.parse_args()
    from experiments import simulation as SE, system as YE, network as NE
    steps = dict(tune=lambda: [SE.tune(ds, a.quick) for ds in C.DATASETS if C.available(ds, 'iid', 0)],
                 E1=lambda: SE.e1(a.quick), E2=lambda: SE.e2(a.quick), E3=lambda: YE.e3(a.quick), E4=lambda: SE.e4(a.quick),
                 E5=lambda: YE.e5(a.quick), E6=lambda: NE.e6(a.quick, a.netns), E7=lambda: NE.e7(a.quick),
                 E8=lambda: YE.e8(a.quick, rounds=a.e8_rounds), E9=lambda: SE.e9(a.quick),
                 report=lambda: __import__('experiments.report', fromlist=['x']).main())
    order = ['tune', 'E1', 'E2', 'E4', 'E9', 'E5', 'E3', 'E6', 'E7', 'E8', 'report']
    todo = order if a.what == ['all'] else a.what
    os.makedirs(C.RESULTS, exist_ok=True)
    log = open(os.path.join(C.RESULTS, 'run.log'), 'a')
    for w in todo:
        if w == 'prepare':
            prepare(a.raw, a.quick); continue
        t0 = time.time()
        print('=== %s' % w, flush=True)
        steps[w]()
        msg = '%s %s done in %.1f min (code %s)' % (time.strftime('%Y-%m-%d %H:%M'), w, (time.time() - t0) / 60, C.code_version())
        print(msg, flush=True)
        if w != 'report':                     # drawing tables leaves the result folder unchanged
            log.write(msg + '\n'); log.flush()


if __name__ == '__main__':
    main()
