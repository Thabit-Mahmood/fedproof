"""Tables and quick-look figures drawn only from the result files (runs/ or results/). Writes <result folder>/report/:
tables as Markdown and CSV, figures as PDF and PNG, and summary.json with every number quoted in the text. The
figures of the paper itself are drawn by figures/make_figures.py."""
import os
import sys
import json
import collections
sys.path.insert(0, '.')
import numpy as np
from experiments import common as C

OUT = os.path.join(C.RESULTS, 'report')


def rows(name):
    p = os.path.join(C.RESULTS, name + '.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def ms(vals):
    vals = [v for v in vals if v is not None and v == v]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float('nan'), float('nan'), 0)


def table(path, header, body):
    with open(path + '.md', 'w') as f:
        f.write('| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header) + '\n')
        for r in body:
            f.write('| ' + ' | '.join(str(x) for x in r) + ' |\n')
    with open(path + '.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for r in body:
            f.write(','.join('"%s"' % x for x in r) + '\n')


def fig(name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'figure.dpi': 200, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def save(plt, name):
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), bbox_inches='tight')
    plt.close('all')


def main():
    try:
        _main()
    except Exception as e:
        print('report: stopped at', repr(e))
        raise


def _main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    # ---- E1 table
    r1 = rows('E1_detection')
    g = collections.defaultdict(list)
    for r in r1:
        g[(r['dataset'], r['split'], r['system'])].append(r['macro_f1'])
    body = [(ds, sp, sy, '%.4f +- %.4f' % ms(v)[:2], ms(v)[2]) for (ds, sp, sy), v in sorted(g.items())]
    table(os.path.join(OUT, 'table_E1_detection'), ['dataset', 'split', 'system', 'macro-F1', 'seeds'], body)
    summary['E1'] = {'|'.join(k): ms(v)[0] for k, v in g.items()}
    # ---- E2 figure 4: macro-F1 against epsilon
    r2 = rows('E2_privacy')
    if r2:
        plt = fig('fig4')
        dss = sorted({r['dataset'] for r in r2})
        f, axs = plt.subplots(1, len(dss), figsize=(2.4 * len(dss), 2.0), squeeze=False)
        for ax, ds in zip(axs[0], dss):
            for sy, sp, mk, lab in (('ours', 'iid', 'o-', 'ours, 10 plants'), ('ours', C.SCALE['tag'], '^-', 'ours, 50 plants'),
                                    ('trusted_dp', 'iid', 's--', 'trusted aggregator, 10 plants')):
                pts = collections.defaultdict(list)
                for r in r2:
                    if r['dataset'] == ds and r['system'] == sy and r['split'] == sp and r.get('eps') and r['eps'] == r['eps'] and r['eps'] > 0:
                        pts[(r['r'], r['uses'])].append((r['eps'], r['macro_f1']))
                xs = sorted((np.mean([p[0] for p in v]), np.mean([p[1] for p in v]), np.std([p[1] for p in v])) for v in pts.values())
                if xs:
                    ax.errorbar([x[0] for x in xs], [x[1] for x in xs], yerr=[x[2] for x in xs], fmt=mk, ms=3, lw=1, label=lab)
            ref = [r['macro_f1'] for r in r1 if r['dataset'] == ds and r['system'] == 'float_central' and r['split'] == 'iid']
            if ref:
                ax.axhline(np.mean(ref), color='gray', lw=0.8, ls=':', label='floating point, central')
            if ax.lines or ax.collections:
                ax.set_xscale('log')
            ax.set_xlabel('privacy budget epsilon (delta = 1e-5)'); ax.set_title(ds)
        axs[0][0].set_ylabel('macro-F1'); axs[0][0].legend(frameon=False, fontsize=6)
        save(plt, 'detection_vs_privacy')
        summary['E2_rows'] = len(r2)
    # ---- E2 table: epsilon against the number of honest plants (two uses)
    rh = rows('E2_hmin')
    if rh:
        Hs = sorted({r['H'] for r in rh})
        body = []
        for ds in sorted({r['dataset'] for r in rh}):
            for rr in sorted({r['r'] for r in rh}):
                e = {r['H']: r['eps'] for r in rh if r['dataset'] == ds and r['r'] == rr and r['uses'] == 2}
                body.append([ds, rr] + ['%.3g' % e[H] if H in e else '' for H in Hs])
        table(os.path.join(OUT, 'table_E2_eps_vs_honest'), ['dataset', 'r'] + ['H=%d' % H for H in Hs], body)
        summary['E2_hmin'] = {r['key']: r['eps'] for r in rh}
    # ---- E4 figure 5
    r4 = rows('E4_poisoning')
    if r4:
        plt = fig('fig5')
        atts = [a for a in ('attack_to_normal', 'backdoor', 'label_flip', 'garbage') if any(r['attack'] == a for r in r4)]
        f, axs = plt.subplots(2, len(atts), figsize=(2.2 * len(atts), 3.6), squeeze=False)
        for j, at in enumerate(atts):
            for sy, mk in (('plain_fedavg', 'x-'), ('norm_bound_only', 's--'), ('krum', 'v:'), ('median', 'd:'), ('trimmed_mean', 'p:'), ('ours', 'o-')):
                for i, metric in enumerate(('attack_success', 'macro_f1')):
                    pts = collections.defaultdict(list)
                    for r in r4:
                        if r['attack'] == at and r['system'] == sy and r.get(metric) is not None:
                            pts[r['share']].append(r[metric])
                    if pts:
                        xs = sorted(pts)
                        axs[i][j].errorbar([10 * x for x in xs], [np.mean(pts[x]) for x in xs], yerr=[np.std(pts[x]) for x in xs],
                                           fmt=mk, ms=3, lw=1, label=sy.replace('_', ' '))
            axs[0][j].set_title(at.replace('_', ' ')); axs[1][j].set_xlabel('poisoned plants (%)')
        axs[0][0].set_ylabel('attack success'); axs[1][0].set_ylabel('clean macro-F1'); axs[1][0].legend(frameon=False, fontsize=6)
        save(plt, 'poisoning')
    # ---- E5 figure 6 and table
    r5 = rows('E5_proofcost')
    if r5:
        body = [(r['model'], r['d'], r['b'], r['level'], r['prove_s'], r['verify_s'], r['msg_kB'], r['peak_rss_MB']) for r in r5]
        table(os.path.join(OUT, 'table_E5_proofcost'), ['model', 'parameters', 'chunk', 'level', 'prove s', 'verify s', 'message kB', 'peak MB'], body)
        plt = fig('fig6')
        f, axs = plt.subplots(1, 3, figsize=(6.6, 1.9))
        for lv, mk in (('P128', 'o-'), ('P256', 's--')):
            sel = sorted((r for r in r5 if r['level'] == lv and r['b'] == 16), key=lambda r: r['d'])
            for ax, k, lab in zip(axs, ('prove_s', 'verify_s', 'msg_kB'), ('proving (s)', 'verification (s)', 'message (kB)')):
                ax.plot([r['d'] for r in sel], [r[k] for r in sel], mk, ms=3, lw=1, label=lv); ax.set_ylabel(lab); ax.set_xlabel('parameters')
        axs[0].legend(frameon=False)
        save(plt, 'proof_cost')
    # ---- E6 figures 7 and 8
    r6 = rows('E6_network')
    if r6:
        live = [r for r in r6 if r['mode'] == 'live' and r.get('profile') != 'unshaped']
        body = [(r['profile'], r['dropout'], r['shaping'], '%.2f' % np.mean(r['round_s']), '%.2f' % np.mean(r['emulated_same_cores']),
                 '%.1f' % np.mean(r['error_pct']), '%.2f' % np.mean(r['emulated_one_machine_per_party'])) for r in live]
        table(os.path.join(OUT, 'table_E6_rounds'), ['profile', 'dropout', 'shaping', 'live s', 'emulated s (same cores)', 'error %', 'one machine per party s'], body)
        em = [r for r in r6 if r['mode'] == 'emulated']
        if em:
            plt = fig('fig8')
            f, ax = plt.subplots(figsize=(3.2, 2.2))
            for dl in sorted({r['delay_ms'] for r in em}):
                sel = sorted((r for r in em if r['delay_ms'] == dl and r['dataset'] == em[0]['dataset']), key=lambda r: r['down'])
                ax.plot([r['down'] for r in sel], [np.mean(r['round_s']) for r in sel], 'o-', ms=2, lw=1, label='%g ms' % dl)
            ax.set_xscale('log'); ax.set_xlabel('plant downlink (Mbit/s; uplink half)'); ax.set_ylabel('round time (s)')
            ax.legend(frameon=False, fontsize=6, title='one-way delay', title_fontsize=6)
            save(plt, 'round_time_links')
        un = [r for r in r6 if r.get('profile') == 'unshaped']
        if un:
            body = [(r['dataset'], int(r['bytes_by_kind']['proofs']), int(r['bytes_by_kind']['other']), int(r['bytes_down_per_party']),
                     int(r['bytes_up_per_party'])) for r in un]
            table(os.path.join(OUT, 'table_E6_traffic'), ['dataset', 'log bytes: plant messages', 'log bytes: other', 'mean down per party', 'mean up per party'], body)
    # ---- E7 figure 9
    r7 = rows('E7_scaling')
    if r7:
        plt = fig('fig9')
        f, ax = plt.subplots(figsize=(3.2, 2.2))
        em = sorted((r for r in r7 if r['key'].endswith('emulated')), key=lambda r: r['plants'])
        ax.plot([r['plants'] for r in em], [np.mean(r['emulated_round_s']) for r in em], 'o-', ms=3, lw=1, label='emulated')
        rp = sorted((r for r in r7 if 'replay_round_s' in r), key=lambda r: r['plants'])
        ax.plot([r['plants'] for r in rp], [np.mean(r['replay_round_s']) for r in rp], 's', ms=4, label='replayed over sockets')
        ax.set_xlabel('plants'); ax.set_ylabel('round time (s)'); ax.legend(frameon=False, fontsize=6)
        save(plt, 'scaling')
    # ---- E8 table
    r8 = rows('E8_endtoend')
    if r8:
        body = [(r['dataset'], r['level'], r['rounds'], '%.1f' % r['round_s_mean'], r['exact_vectors'], r['mismatches'], r['audit_same'],
                 '%.4f' % r['macro_f1']) for r in r8]
        table(os.path.join(OUT, 'table_E8_endtoend'), ['dataset', 'level', 'rounds', 'mean round s', 'vectors checked bit for bit', 'mismatches', 'audit reproduces model', 'macro-F1'], body)
        summary['E8'] = body
    # ---- E9 table
    r9 = rows('E9_membership')
    if r9:
        epsmap = {('%s_%s_r%d_u%d_%d_%s' % (r['dataset'], r['split'], r['r'], r['uses'], r['seed'], r['system'])): r['eps'] for r in r2}
        body = sorted((r['dataset'], r['system'], r['r'], '%.3g' % epsmap.get(r['model'], float('nan')), '%.4f' % r['advantage']) for r in r9)
        table(os.path.join(OUT, 'table_E9_membership'), ['dataset', 'system', 'noise bits r', 'epsilon', 'attack advantage'], body)
    # ---- figure 10: per-class detection on the most imbalanced dataset
    if r1:
        imb = {}
        for ds in {r['dataset'] for r in r1}:
            try:
                m = C.load(ds, 'iid', 0)['meta']; cc = list(m['class_counts'].values()); imb[ds] = max(cc) / max(1, min(cc))
            except Exception:
                pass
        if imb:
            ds = max(imb, key=imb.get)
            m = C.load(ds, 'iid', 0)['meta']
            plt = fig('fig10')
            f, ax = plt.subplots(figsize=(3.4, 2.2))
            for sy, off in (('float_central', -0.2), ('int_no_noise', 0.2)):
                rec = [r['per_class_recall'] for r in r1 if r['dataset'] == ds and r['system'] == sy and r['split'] == 'iid']
                if rec:
                    v = np.mean(rec, 0); ax.bar(np.arange(len(v)) + off, v, 0.4, label=sy.replace('_', ' '))
            r2s = [r['per_class_recall'] for r in r2 if r['dataset'] == ds and r['system'] == 'ours' and r['split'] == 'iid' and r['r'] == 12 and r['uses'] == 2]
            ax.set_xticks(range(len(m['classes']))); ax.set_xticklabels(m['classes'], rotation=70, fontsize=5)
            ax.set_ylabel('detection rate (recall)'); ax.legend(frameon=False, fontsize=6); ax.set_title(ds)
            save(plt, 'per_class')
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=str)
    print('report written to', OUT)


if __name__ == '__main__':
    main()
