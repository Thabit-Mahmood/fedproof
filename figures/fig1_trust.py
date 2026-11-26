"""Fig. 1: what each approach trusts, and what a corrupted plant can do in each."""
import sys
from figkit import Canvas, C

W, H = 7.16, 2.42
cv = Canvas(W, H)
PW, GAP = 2.32, 0.10
rows = ('Individual updates hidden', 'Training step proved', 'Noise verified', 'Post-quantum')
panels = [
    ('(a) Federated averaging', (False, False, False, False)),
    ('(b) Secure aggregation with norm checks', (True, False, False, False)),
    ('(c) Proof-carrying training (this work)', (True, True, True, True)),
]


def plants(x0, y_top, bad_y):
    ys = [y_top, y_top - 0.36, bad_y]
    for k, y in enumerate(ys):
        bad = y == bad_y
        cv.icon('factory', x0, y, 0.25, C['bad'] if bad else C['plant'])
        if bad:
            cv.icon('incognito', x0 + 0.13, y + 0.12, 0.13, C['bad'], z=7)
    return ys


for p, (title, marks) in enumerate(panels):
    x = p * (PW + GAP)
    if p:                                                        # thin rule between panels, no cards
        cv.ax.plot([x - GAP / 2, x - GAP / 2], [0.10, H - 0.08], color='#cfcfcf', lw=0.6, zorder=1)
    cv.text(x + PW / 2, H - 0.15, title, size=7.6, weight='bold')
    top, bad = 1.86, 1.08
    ys = plants(x + 0.25, top, bad)
    if p == 0:
        sx, sy = x + 1.85, 1.50
        cv.icon('server', sx, sy, 0.34, C['public'])
        cv.icon('magnify', sx + 0.17, sy + 0.17, 0.15, C['bad'], z=7)
        cv.text(sx, sy - 0.40, 'server reads\nevery update', size=6.2, color=C['bad'])
        for y in ys[:2]:
            cv.arrow(x + 0.42, y, sx - 0.2, sy + 0.04, color=C['plant'])
        cv.arrow(x + 0.42, bad, sx - 0.2, sy - 0.06, color=C['bad'], lw=1.1)
        cv.text(x + 1.05, 1.92, 'update in clear', size=6.2, color=C['plant'], rotation=-9)
        cv.text(x + 1.05, 1.08, 'any vector: accepted', size=6.2, color=C['bad'], rotation=13)
    elif p == 1:
        sx, sy = x + 1.85, 1.50
        cv.icon('server', sx, sy, 0.34, C['public'])
        cv.icon('sigma', sx + 0.17, sy + 0.17, 0.14, C['node'], z=7)
        cv.text(sx + 0.06, sy - 0.46, 'sees only the sum;\nnoise must be added\nhonestly by someone', size=6.2, color=C['public'])
        for y in ys[:2]:
            cv.arrow(x + 0.42, y, sx - 0.2, sy + 0.04, color=C['plant'])
            cv.icon('lock', x + 0.95, y - 0.05 if y == ys[0] else y + 0.03, 0.11, C['plant'], z=7)
        cv.arrow(x + 0.42, bad, sx - 0.2, sy - 0.06, color=C['bad'], lw=1.1)
        cv.icon('lock', x + 0.95, bad + 0.12, 0.11, C['bad'], z=7)
        cv.text(x + 0.88, 1.01, 'any update in bound: accepted', size=6.2, color=C['bad'], rotation=13)
    else:
        lx = x + 1.08
        for k in range(5):                                       # the public log as a strip of entries
            cv.box(lx - 0.12, 1.22 + k * 0.13, 0.24, 0.11, fc=C['log_bg'], ec=C['log'], lw=0.5, r=0.015, z=4)
        cv.text(lx + 0.08, 2.06, 'public log', size=6.0, color=C['log'])
        for y in ys[:2]:
            cv.arrow(x + 0.42, y, lx - 0.15, 1.6, color=C['plant'])
        cv.icon('file-certificate', x + 0.70, 1.84, 0.13, C['plant'], z=7)
        cv.text(x + 0.50, 2.06, 'shares + proof', size=6.0, color=C['plant'])
        cv.arrow(x + 0.42, bad, lx - 0.15, 1.30, color=C['bad'], lw=1.1)
        cv.icon('close-circle', x + 0.80, 1.20, 0.14, C['bad'], z=7)
        cv.text(x + 0.66, 0.97, 'proof fails:\nrejected', size=6.2, color=C['bad'])
        for k, ny in enumerate((1.86, 1.55, 1.24)):
            cv.icon('server-security', x + 1.92, ny, 0.22, C['node'])
            cv.arrow(lx + 0.14, 1.55, x + 1.79, ny, color=C['node'], lw=0.6, ms=5)
        cv.text(x + 1.92, 2.04, 'committee', size=6.2, color=C['node'])
        cv.text(x + 1.92, 1.02, 'opens one sum', size=6.2, color=C['node'])
        cv.icon('magnify-scan', x + 1.40, 0.95, 0.18, C['aud'])
        cv.text(x + 1.40, 0.79, 'auditor replays', size=6.0, color=C['aud'])
    # what the approach guarantees
    cv.ax.plot([x + 0.08, x + PW - 0.08], [0.70, 0.70], color='#d0d0d0', lw=0.5, zorder=2)
    for k, (lab, ok) in enumerate(zip(rows, marks)):
        y = 0.56 - k * 0.135
        cv.text(x + 0.14, y, lab, size=6.4, ha='left', color=C['ink'])
        cv.icon('check-circle' if ok else 'close-circle', x + PW - 0.20, y, 0.12, C['good'] if ok else C['bad'])
cv.save(sys.argv[1] if len(sys.argv) > 1 else 'fig1_trust.png')
