"""Fig. 2: parties, the public log and the steps of a round, drawn as one open layout."""
import sys
from figkit import Canvas, C

W, H = 7.16, 3.05
cv = Canvas(W, H)

# ------------------------------------------------------------------ column headings
cv.text(1.02, 2.92, 'Industrial plants', size=7.8, weight='bold', color=C['plant'])
cv.text(3.58, 2.92, 'Relay and public log', size=7.8, weight='bold', color=C['log'])
cv.text(6.10, 2.92, 'Committee at edge sites', size=7.8, weight='bold', color=C['node'])
for x0, x1, col in ((0.10, 1.94, C['plant']), (2.62, 4.54, C['log']), (5.08, 7.12, C['node'])):
    cv.ax.plot([x0, x1], [2.80, 2.80], color=col, lw=0.8, zorder=1)


# ------------------------------------------------------------------ plants
def plant(y, name, bad=False):
    col = C['bad'] if bad else C['plant']
    cv.icon('factory', 0.30, y + 0.05, 0.30, col)
    cv.text(0.30, y - 0.17, name, size=6.4, color=col, weight='bold')
    if bad:
        cv.icon('incognito', 0.45, y + 0.21, 0.14, C['bad'], z=7)
    for x, ic, lab, c in ((0.80, 'database', 'records', C['secret']), (1.22, 'chip', 'prover', col),
                          (1.64, 'email-lock', 'message', col)):
        cv.icon(ic, x, y + 0.06, 0.19, c)
        cv.text(x, y - 0.13, lab, size=5.8, color=c)
    cv.arrow(0.91, y + 0.06, 1.11, y + 0.06, color='#8a8a8a', lw=0.6, ms=5)
    cv.arrow(1.33, y + 0.06, 1.53, y + 0.06, color='#8a8a8a', lw=0.6, ms=5)


plant(2.42, 'Plant 1')
plant(1.82, 'Plant 2')
cv.text(1.02, 1.43, '⋮', size=11, color=C['plant'])
plant(0.98, 'Plant N', bad=True)
cv.text(1.02, 0.60, 'any number of plants may be corrupted', size=6.0, color=C['bad'], style='italic')

# ------------------------------------------------------------------ relay and the log
cv.icon('router-network', 3.06, 2.56, 0.24, C['relay'])
cv.text(3.24, 2.56, 'untrusted relay', size=6.2, ha='left', color=C['relay'])
cv.text(3.58, 2.37, 'may drop, delay, duplicate, reorder or pad; never rewrites', size=5.6, color=C['relay'],
        style='italic')
entries = [('1', 'beacon$_t$'), ('2', 'round keys$_t$'), ('3', 'plant messages'), ('5', 'complaints'),
           ('6', 'seed reveals'), ('7', 'certificate'), ('8', 'sums')]
EX, EW, EH = 3.03, 1.10, 0.19
y0 = 2.12
for k, (b, lab) in enumerate(entries):
    y = y0 - k * EH
    cv.box(EX, y - EH / 2, EW, EH, fc='white', ec=C['log'], lw=0.5, r=0.0, z=4)
    cv.text(EX + EW / 2, y, lab, size=6.1, color=C['ink'], z=6)
    cv.badge(EX - 0.12, y, b, r=0.065, size=5.6)
ybot = y0 - (len(entries) - 1) * EH - EH / 2
cv.text(EX + EW / 2, ybot - 0.13, 'append-only; head $=H(\\mathrm{head},\\mathrm{entry})$', size=5.7, color=C['log'])
cv.icon('magnify-scan', 3.10, ybot - 0.42, 0.22, C['aud'])
cv.text(3.26, ybot - 0.37, 'every party and the auditor replay the log;', size=5.7, ha='left', color=C['aud'])
cv.text(3.26, ybot - 0.50, 'steps 4 and 9 are computed by each replica', size=5.7, ha='left', color=C['aud'])


# ------------------------------------------------------------------ committee
def node(y, name, bad=False):
    col = C['bad'] if bad else C['node']
    cv.icon('server-security', 5.30, y + 0.05, 0.28, col)
    cv.text(5.30, y - 0.17, name, size=6.4, color=col, weight='bold')
    if bad:
        cv.icon('incognito', 5.45, y + 0.21, 0.13, C['bad'], z=7)
    for x, ic, lab, c in ((5.84, 'key-variant', 'round key', C['secret']), (6.36, 'clipboard-check-outline', 'row check', col),
                          (6.86, 'sigma', 'one sum', col)):
        cv.icon(ic, x, y + 0.06, 0.17, c)
        cv.text(x, y - 0.13, lab, size=5.8, color=c)


node(2.42, 'Node 1')
node(1.82, 'Node 2')
cv.text(6.10, 1.43, '⋮', size=11, color=C['node'])
node(0.98, 'Node K', bad=True)
cv.text(6.10, 0.60, 'at most $T_n$ corrupted per round; keys erased at close', size=6.0, color=C['node'],
        style='italic')

# ------------------------------------------------------------------ flows
ER = EX + EW
cv.arrow(1.80, 2.40, EX - 0.20, 1.76, color=C['plant'], lw=0.9)                       # messages (3)
cv.arrow(1.80, 1.84, EX - 0.20, 1.73, color=C['plant'], lw=0.9)
cv.arrow(1.80, 1.00, EX - 0.20, 1.36, color=C['plant'], lw=0.9)                       # seed reveal (6)
cv.arrow(5.12, 2.44, ER + 0.03, 2.03, color=C['node'], lw=0.9)                        # beacon, keys (1, 2)
cv.arrow(ER + 0.03, 1.74, 5.12, 1.86, color=C['node'], lw=0.9, ls='--')               # reads own row
cv.text(4.62, 1.92, 'reads its own row', size=5.5, color=C['node'], style='italic')
cv.arrow(5.12, 1.00, ER + 0.03, 1.02, color=C['node'], lw=0.9)                        # certificate, sums (7, 8)
cv.ax.set_ylim(0.30, H); cv.fig.set_size_inches(W, H - 0.30)
cv.save(sys.argv[1] if len(sys.argv) > 1 else 'fig2_system.png')
