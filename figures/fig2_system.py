"""Fig. 2: parties, links, the public log, trust and the nine steps of a round."""
import sys
from figkit import Canvas, C

W, H = 7.16, 3.45
cv = Canvas(W, H)

# ------------------------------------------------------------------ zones
cv.box(0.02, 0.86, 2.18, 2.57, fc=C['plant_bg'], ec=C['plant'], lw=0.6, r=0.06)
cv.pill(1.11, 3.30, 'Industrial plants (N)', C['plant'])
cv.box(2.50, 0.86, 2.06, 2.57, fc=C['log_bg'], ec=C['log'], lw=0.6, r=0.06)
cv.pill(3.53, 3.30, 'Relay and public log', C['log'])
cv.box(4.86, 0.86, 2.28, 2.57, fc=C['node_bg'], ec=C['node'], lw=0.6, r=0.06)
cv.pill(6.00, 3.30, 'Committee at edge sites (K)', C['node'])

# ------------------------------------------------------------------ plants
def plant(y, name, bad=False):
    col = C['bad'] if bad else C['plant']
    cv.box(0.10, y - 0.29, 2.02, 0.58, fc='white', ec=col, lw=0.6, r=0.05, z=2)
    cv.icon('factory', 0.33, y + 0.04, 0.30, col)
    cv.text(0.33, y - 0.20, name, size=6.4, color=col, weight='bold')
    cv.icon('database', 0.78, y + 0.06, 0.19, C['secret'])
    cv.text(0.78, y - 0.15, 'records', size=5.8, color=C['secret'])
    cv.icon('chip', 1.20, y + 0.06, 0.19, col)
    cv.text(1.20, y - 0.15, 'prover', size=5.8, color=col)
    cv.icon('email-lock', 1.62, y + 0.06, 0.19, col)
    cv.text(1.62, y - 0.15, 'message', size=5.8, color=col)
    cv.arrow(0.89, y + 0.06, 1.09, y + 0.06, color='#777777', lw=0.6, ms=5)
    cv.arrow(1.31, y + 0.06, 1.51, y + 0.06, color='#777777', lw=0.6, ms=5)
    if bad:
        cv.icon('incognito', 0.47, y + 0.20, 0.14, C['bad'], z=7)


plant(2.82, 'Plant 1')
plant(2.12, 'Plant 2')
cv.text(1.11, 1.66, '⋮', size=11, color=C['plant'])
plant(1.32, 'Plant N', bad=True)
cv.text(1.11, 0.925, 'any number may be corrupted', size=6.0, color=C['bad'], style='italic')

# ------------------------------------------------------------------ relay and log
cv.icon('router-network', 3.53, 2.98, 0.26, C['relay'])
cv.text(3.53, 2.725, 'untrusted relay: may drop, delay, duplicate,\nreorder or pad; never rewrites', size=5.9, color=C['relay'])
entries = [('1', 'beacon$_t$'), ('2', 'round keys$_t$'), ('3', 'messages of plants'), ('5', 'complaints'),
           ('6', 'seed reveals'), ('7', 'certificate'), ('8', 'sums')]
y0, dy = 2.42, 0.205
for k, (b, lab) in enumerate(entries):
    y = y0 - k * dy
    cv.box(2.98, y - 0.08, 1.10, 0.16, fc='white', ec=C['log'], lw=0.5, r=0.02, z=4)
    cv.text(3.56, y, lab, size=6.2, color=C['ink'], z=6)
    cv.badge(2.88, y, b, r=0.065, size=5.6)
    if k:
        cv.ax.plot([4.14, 4.14], [y + 0.08, y + dy - 0.08], color=C['log'], lw=0.6, zorder=3)
    cv.icon('link-variant', 4.24, y, 0.10, C['log'])
cv.text(3.53, 0.97, 'append-only, canonical bytes;\nhead$\\,=H(\\mathrm{head},\\mathrm{entry})$', size=5.9, color=C['log'])

# ------------------------------------------------------------------ committee
def node(y, name, bad=False):
    col = C['bad'] if bad else C['node']
    cv.box(4.94, y - 0.26, 2.12, 0.52, fc='white', ec=col, lw=0.6, r=0.05, z=2)
    cv.icon('server-security', 5.17, y + 0.03, 0.27, col)
    cv.text(5.17, y - 0.19, name, size=6.2, color=col, weight='bold')
    cv.icon('key-variant', 5.60, y + 0.05, 0.16, C['secret'])
    cv.text(5.60, y - 0.13, 'round key', size=5.6, color=C['secret'])
    cv.icon('clipboard-check-outline', 6.10, y + 0.05, 0.16, col)
    cv.text(6.10, y - 0.13, 'row check', size=5.6, color=col)
    cv.icon('sigma', 6.62, y + 0.05, 0.16, col)
    cv.text(6.62, y - 0.13, 'one sum', size=5.6, color=col)
    if bad:
        cv.icon('incognito', 5.31, y + 0.19, 0.13, C['bad'], z=7)


node(2.82, 'Node 1')
node(2.20, 'Node 2')
cv.text(6.00, 1.80, '⋮', size=11, color=C['node'])
node(1.36, 'Node K', bad=True)
cv.text(6.00, 0.98, 'at most $T_n$ corrupted per round;\nkeys erased when the round closes', size=5.9,
        color=C['node'], style='italic')

# ------------------------------------------------------------------ flows (steps)
cv.arrow(2.13, 2.55, 2.79, 2.04, color=C['plant'], lw=0.9)                 # messages
cv.arrow(2.13, 2.05, 2.79, 1.99, color=C['plant'], lw=0.9)
cv.arrow(2.13, 1.36, 2.79, 1.59, color=C['plant'], lw=0.9)                 # seed reveal
cv.arrow(4.93, 2.80, 4.32, 2.44, color=C['node'], lw=0.9)                  # beacon, keys
cv.arrow(4.32, 2.00, 4.93, 2.20, color=C['node'], lw=0.9, ls='--')         # read own row
cv.arrow(4.93, 1.36, 4.32, 1.20, color=C['node'], lw=0.9)                  # certificate, sums
cv.text(4.70, 1.86, 'reads its\nown row', size=5.4, color=C['node'], style='italic')

# ------------------------------------------------------------------ bottom band: every replica and the auditor
cv.box(0.02, 0.02, 7.12, 0.78, fc=C['aud_bg'], ec=C['aud'], lw=0.6, r=0.06)
cv.icon('cogs', 0.30, 0.41, 0.30, C['aud'])
cv.text(0.55, 0.53, 'Every party, the driver and the auditor run one deterministic replica of the log', size=6.6,
        ha='left', weight='bold', color=C['aud'])
cv.text(0.55, 0.30, 'It counts only signed first copies (4), verifies every proof, applies strikes only for faults '
        'anyone can check, opens the sum with\nerror correction and applies the same public update, giving '
        '$W_{t+1}$ (9). An auditor who replays the log reproduces the final model exactly.', size=6.1, ha='left', color=C['ink'])
cv.icon('magnify-scan', 6.88, 0.41, 0.30, C['aud'])
cv.save(sys.argv[1] if len(sys.argv) > 1 else 'fig2_system.png')
