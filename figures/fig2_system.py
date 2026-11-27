"""Fig. 2: parties, the public log and the steps of a round, in three outlined sections."""
import sys
from figkit import Canvas, C

W, H = 7.16, 3.00
cv = Canvas(W, H)
BY0, BY1 = 0.04, 2.94                                            # bottom and top of the three section boxes
HEAD = 2.76                                                      # heading line, inside each box
SECT = [(0.04, 2.14, 'Industrial plants', C['plant'], C['plant_bg']),
        (2.48, 2.20, 'Relay and public log', C['log'], C['log_bg']),
        (4.98, 2.14, 'Committee at edge sites', C['node'], C['node_bg'])]
for x0, w, title, col, bg in SECT:
    cv.box(x0, BY0, w, BY1 - BY0, fc=bg, ec=col, lw=1.0, r=0.06, z=1)
    cv.text(x0 + w / 2, HEAD, title, size=9.0, weight='bold', color=col)

ROWS = (2.34, 1.70, 0.92)                                        # first, second and last party in a section
DOTS = 1.31
LAB = 0.21                                                       # label offset below an icon row


# ------------------------------------------------------------------ plants
def plant(y, name, bad=False):
    col = C['bad'] if bad else C['plant']
    cv.icon('factory', 0.36, y + 0.06, 0.38, col)
    cv.text(0.36, y - LAB, name, size=7.8, color=col, weight='bold')
    if bad:
        cv.icon('incognito', 0.55, y + 0.25, 0.17, C['bad'], z=7)
    for x, ic, lab, c in ((0.90, 'database', 'records', C['secret']), (1.38, 'chip', 'prover', col),
                          (1.86, 'email-lock', 'message', col)):
        cv.icon(ic, x, y + 0.06, 0.25, c)
        cv.text(x, y - LAB, lab, size=7.2, color=c)
    cv.arrow(1.04, y + 0.06, 1.24, y + 0.06, color='#8a8a8a', lw=0.7, ms=6)
    cv.arrow(1.52, y + 0.06, 1.72, y + 0.06, color='#8a8a8a', lw=0.7, ms=6)


plant(ROWS[0], 'Plant 1')
plant(ROWS[1], 'Plant 2')
cv.text(1.11, DOTS, '⋮', size=13, color=C['plant'])
plant(ROWS[2], 'Plant N', bad=True)
cv.text(1.11, 0.45, 'any number of plants', size=7.2, color=C['bad'], style='italic')
cv.text(1.11, 0.31, 'may be corrupted', size=7.2, color=C['bad'], style='italic')

# ------------------------------------------------------------------ relay and the log
cv.icon('router-network', 2.76, 2.45, 0.28, C['relay'])
cv.text(2.96, 2.45, 'untrusted relay', size=7.8, ha='left', color=C['relay'], weight='bold')
cv.text(3.58, 2.22, 'may drop, delay, duplicate, reorder', size=7.0, color=C['relay'], style='italic')
cv.text(3.58, 2.09, 'or pad; never rewrites', size=7.0, color=C['relay'], style='italic')
entries = [('1', 'beacon$_t$'), ('2', 'round keys$_t$'), ('3', 'plant messages'), ('5', 'complaints'),
           ('6', 'seed reveals'), ('7', 'certificate'), ('8', 'sums')]
EX, EW, EH = 2.86, 1.20, 0.19
ER = EX + EW
y0 = 1.86
ey = [y0 - k * EH for k in range(len(entries))]
for y, (b, lab) in zip(ey, entries):
    cv.box(EX, y - EH / 2, EW, EH, fc='white', ec=C['log'], lw=0.6, r=0.0, z=4)
    cv.text(EX + EW / 2, y, lab, size=7.3, color=C['ink'], z=6)
    cv.badge(EX - 0.15, y, b, r=0.075, size=6.4)
ybot = ey[-1] - EH / 2
cv.text(3.58, ybot - 0.13, 'append-only; head $=H(\\mathrm{head},\\mathrm{entry})$', size=7.0, color=C['log'])
cv.icon('magnify-scan', 2.70, 0.255, 0.24, C['aud'])
cv.text(2.86, 0.32, 'every party and the auditor replay', size=7.0, ha='left', color=C['aud'])
cv.text(2.86, 0.19, 'the log; each replica runs steps 4 and 9', size=7.0, ha='left', color=C['aud'])


# ------------------------------------------------------------------ committee
def node(y, name, bad=False):
    col = C['bad'] if bad else C['node']
    cv.icon('server-security', 5.30, y + 0.06, 0.36, col)
    cv.text(5.30, y - LAB, name, size=7.8, color=col, weight='bold')
    if bad:
        cv.icon('incognito', 5.48, y + 0.25, 0.16, C['bad'], z=7)
    for x, ic, lab, c in ((5.86, 'key-variant', 'round key', C['secret']), (6.36, 'clipboard-check-outline', 'row check', col),
                          (6.84, 'sigma', 'one sum', col)):
        cv.icon(ic, x, y + 0.06, 0.23, c)
        cv.text(x, y - LAB, lab, size=7.2, color=c)


node(ROWS[0], 'Node 1')
node(ROWS[1], 'Node 2')
cv.text(6.05, DOTS, '⋮', size=13, color=C['node'])
node(ROWS[2], 'Node K', bad=True)
cv.text(6.05, 0.45, 'at most $T_n$ corrupted per round;', size=7.2, color=C['node'], style='italic')
cv.text(6.05, 0.31, 'round keys erased at close', size=7.2, color=C['node'], style='italic')

# ------------------------------------------------------------------ flows between the sections
cv.arrow(2.02, ROWS[0] + 0.02, EX - 0.26, ey[2] + 0.05, color=C['plant'], lw=1.0, ms=8)      # messages (3)
cv.arrow(2.02, ROWS[1] + 0.03, EX - 0.26, ey[2] - 0.02, color=C['plant'], lw=1.0, ms=8)
cv.arrow(2.02, ROWS[2] + 0.04, EX - 0.26, ey[4], color=C['plant'], lw=1.0, ms=8)           # seed reveal (6)
cv.arrow(5.08, ROWS[0] + 0.02, ER + 0.03, (ey[0] + ey[1]) / 2, color=C['node'], lw=1.0, ms=8)  # beacon, keys (1, 2)
cv.arrow(ER + 0.03, ey[2], 5.08, ROWS[1] + 0.02, color=C['node'], lw=1.0, ms=8, ls='--')     # reads own row
cv.text(4.37, ey[3] - 0.04, 'reads its', size=6.9, color=C['node'], style='italic')
cv.text(4.37, ey[3] - 0.16, 'own row', size=6.9, color=C['node'], style='italic')
cv.arrow(5.08, ROWS[2] + 0.04, ER + 0.03, (ey[5] + ey[6]) / 2, color=C['node'], lw=1.0, ms=8)  # certificate, sums (7, 8)
cv.save(sys.argv[1] if len(sys.argv) > 1 else 'fig2_system.png')
