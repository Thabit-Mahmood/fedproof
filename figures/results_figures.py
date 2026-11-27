"""Figures 5 to 9 of the paper, drawn from the result files (600 dpi PNG).

Usage: python3 figures/results_figures.py [result folder, default results] [output folder, default figures/out]"""
import json, os, sys, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

R = sys.argv[1] if len(sys.argv) > 1 else 'results'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'figures/out'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'lines.linewidth': 1.1, 'lines.markersize': 3.5, 'axes.grid': True, 'grid.linewidth': 0.4, 'grid.alpha': 0.35,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 600, 'legend.frameon': False})
DS = [('edge-iiotset', 'Edge-IIoTset'), ('wustl-iiot-2021', 'WUSTL-IIoT-2021'), ('x-iiotid', 'X-IIoTID')]
C_OURS, C_TRUST = '#1f4e8c', '#d9822b'
SYS = [('plain_fedavg', 'Plain FedAvg', '#8c8c8c', 'x', '-'), ('norm_bound_only', 'Norm bound only', '#8c564b', 's', '--'),
       ('krum', 'Krum', '#7b4fa0', 'v', ':'), ('median', 'Median', '#2e8b57', 'D', ':'),
       ('trimmed_mean', 'Trimmed mean', '#17a3a3', 'P', ':'), ('ours', 'FedProof', C_OURS, 'o', '-')]


def rows(name):
    p = os.path.join(R, name + '.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def panel(w, h, left, bottom, right=0.06, top=0.06):
    """One subfigure with a fixed plot area, so that panels placed side by side line up."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h])
    return fig, ax


def save_panel(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), facecolor='white')
    plt.close(fig)


def save_legend(handles, labels, ncol, name, **kw):
    """The legend shared by the subfigures of one figure, drawn as its own image."""
    fig = plt.figure(figsize=(7.16, 0.4))
    fig.legend(handles, labels, loc='center', ncol=ncol, **kw)
    save(fig, name)


# ------------------------------------------------------------------ Fig. 5: detection against privacy, one image per dataset
e1, e2 = rows('E1_detection'), rows('E2_privacy')
for (ds, title), name in zip(DS, ('fig5a_edge', 'fig5b_wustl', 'fig5c_xiiotid')):
    fig, ax = panel(2.32, 2.00, left=0.44, bottom=0.42)
    for sy, sp, u, col, mk, ls, lab in (('ours', 'iid', 2, C_OURS, 'o', '-', 'FedProof, 10 plants'),
                                        ('trusted_dp', 'iid', 2, C_TRUST, 's', '-', 'Trusted aggregator, 10 plants'),
                                        ('ours', 'iid50', 8, C_OURS, '^', '--', 'FedProof, 50 plants'),
                                        ('trusted_dp', 'iid50', 8, C_TRUST, 'v', '--', 'Trusted aggregator, 50 plants')):
        pts = collections.defaultdict(list)
        for r in e2:
            if r['dataset'] == ds and r['system'] == sy and r['split'] == sp and r['uses'] == u:
                pts[r['r']].append((r['eps'], r['macro_f1']))
        xs = sorted((np.mean([p[0] for p in v]), np.mean([p[1] for p in v]), np.std([p[1] for p in v])) for v in pts.values())
        ax.errorbar([x[0] for x in xs], [x[1] for x in xs], yerr=[x[2] for x in xs], color=col, marker=mk, ls=ls,
                    capsize=1.5, elinewidth=0.6, label=lab, mfc='white' if ls == '--' else col)
    nn = np.mean([r['macro_f1'] for r in e1 if r['dataset'] == ds and r['split'] == 'iid' and r['system'] == 'int_no_noise'])
    fc = np.mean([r['macro_f1'] for r in e1 if r['dataset'] == ds and r['split'] == 'iid' and r['system'] == 'float_central'])
    ax.axhline(nn, color=C_OURS, lw=0.8, ls=(0, (1, 1.5)), label='FedProof, no noise')
    ax.axhline(fc, color='#555555', lw=0.8, ls=(0, (4, 2)), label='Centralized model')
    ax.set_xscale('log'); ax.set_xlabel(r'Privacy budget $\varepsilon$ ($\delta = 10^{-5}$)')
    ax.set_ylim(0.2, 1.0); ax.set_ylabel('Macro-F1')
    h, l = ax.get_legend_handles_labels()
    save_panel(fig, name)
save_legend(h, l, 3, 'fig5_legend')

# ------------------------------------------------------------------ Fig. 6: poisoning (mean over datasets and seeds), one image per attack
e4 = rows('E4_poisoning')
panels = (('label_flip', 'macro_f1', 'fig6a_label_flip', 'Macro-F1'), ('garbage', 'macro_f1', 'fig6b_garbage', 'Macro-F1'),
          ('attack_to_normal', 'attack_success', 'fig6c_attack_to_normal', 'Attack success'),
          ('backdoor', 'attack_success', 'fig6d_backdoor', 'Attack success'))
for at, met, name, yl in panels:
    fig, ax = panel(1.74, 1.68, left=0.40, bottom=0.40)
    for si, (sy, lab, col, mk, ls) in enumerate(SYS):
        off = (si - (len(SYS) - 1) / 2) * 0.9      # small horizontal offset per system, same values
        ys, es = [], []
        for sh in (1, 2, 3):
            v = [r[met] for r in e4 if r['attack'] == at and r['share'] == sh and r['system'] == sy and r[met] is not None]
            ys.append(np.mean(v)); es.append(np.std(v) / np.sqrt(max(1, len(v))))
        ax.errorbar([10 + off, 20 + off, 30 + off], ys, yerr=es, color=col, marker=mk, ls=ls, capsize=1.5, elinewidth=0.6,
                    label=lab, ms=4.2 if sy == 'ours' else 3.4, lw=1.7 if sy == 'ours' else 0.9, zorder=5 if sy == 'ours' else 2)
    ax.set_xticks([10, 20, 30]); ax.set_xlabel('Poisoned plants (%)'); ax.set_ylabel(yl)
    ax.set_ylim(-0.03, 1.03) if met == 'attack_success' else ax.set_ylim(0.0, 0.65)
    h, l = ax.get_legend_handles_labels()
    save_panel(fig, name)
save_legend(h, l, 6, 'fig6_legend')

# ------------------------------------------------------------------ Fig. 7: proof cost against model size, one image per chunk size
e5 = rows('E5_proofcost')
for b, name in ((16, 'fig7a_b16'), (128, 'fig7b_b128')):
    fig, ax = panel(1.72, 1.61, left=0.38, bottom=0.40)
    for lvl, col, mk in (('P128', C_OURS, 'o'), ('P256', '#b03a2e', 's')):
        pts = sorted((r['d'], r['prove_s'], r['verify_s']) for r in e5 if r['b'] == b and r['level'] == lvl)
        xs = [p[0] / 1000 for p in pts]
        ax.plot(xs, [p[1] for p in pts], color=col, ls='-', marker=mk, ms=3.5, label='Prove, %s' % lvl)
        ax.plot(xs, [p[2] for p in pts], color=col, ls='--', marker=mk, ms=3.5, mfc='white', label='Verify, %s' % lvl)
    ax.set_yscale('log'); ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(['0.5', '1', '2', '5', '10', '20'])
    ax.set_ylim(0.4, 32); ax.set_xlabel('Parameters (thousands)'); ax.set_ylabel('Time (s)')
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    h, l = ax.get_legend_handles_labels()
    save_panel(fig, name)
save_legend(h, l, 4, 'fig7_legend', columnspacing=1.0, handlelength=1.6)

# ------------------------------------------------------------------ Fig. 8: round time on kernel-shaped links
e6 = rows('E6_network')
prof = (('industrial-ethernet', 'Industrial\nEthernet'), ('private-5g', 'Private\n5G'), ('constrained-wan', 'Constrained\nwide-area'))
get = lambda p, drop, loss, f: np.mean(next(r for r in e6 if r['key'] == 'edge-iiotset|live|%s|drop=%s|loss=%g|kernel' % (p, drop, loss))[f])
fig, ax = plt.subplots(figsize=(3.5, 1.9))
x = np.arange(len(prof)); w = 0.2
series = ((('live', 0), 'Live, no loss', C_OURS, None), (('em', 0), 'Emulated, no loss', '#9db7d8', None),
          (('live', 1), 'Live, 1% loss', '#b03a2e', None), (('one', 0), 'Emulated, one machine per party', '#cfcfcf', '////'))
for i, ((kind, loss), lab, col, hatch) in enumerate(series):
    f = {'live': 'round_s', 'em': 'emulated_same_cores', 'one': 'emulated_one_machine_per_party'}[kind]
    ys = [get(p, False, loss, f) for p, _ in prof]
    ax.bar(x + (i - 1.5) * w, ys, w, color=col, edgecolor='#333333', linewidth=0.4, hatch=hatch, label=lab, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([p[1] for p in prof]); ax.set_ylabel('Round time (s)'); ax.grid(axis='x', visible=False)
ax.legend(loc='upper left', ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
ax.set_ylim(0, 205)
fig.tight_layout()
save(fig, 'fig8_links')

# ------------------------------------------------------------------ Fig. 9: scaling
e7 = rows('E7_scaling')
em = sorted({(r['plants'], np.mean(r['emulated_round_s'])) for r in e7})
rp = sorted({(r['plants'], np.mean(r['replay_round_s'])) for r in e7 if r.get('replay_round_s')})
fig, ax = plt.subplots(figsize=(3.5, 1.7))
ax.plot([p[0] for p in em], [p[1] for p in em], color=C_OURS, marker='o', label='Emulated, one machine per party')
ax.plot([p[0] for p in rp], [p[1] for p in rp], ls='none', marker='s', ms=6, mfc='none', mec='#b03a2e', mew=1.0,
        label='Replayed over sockets')
ax.set_xlabel('Plants'); ax.set_ylabel('Round time (s)'); ax.set_xticks([10, 25, 50, 100]); ax.legend(loc='upper left')
fig.tight_layout()
save(fig, 'fig9_scaling')
print('figures written to', OUT)
