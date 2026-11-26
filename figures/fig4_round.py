"""Fig. 4: the steps of one round as a flow in three phases, colored by the party that acts."""
import sys
from matplotlib.patches import Circle
from figkit import Canvas, C

W, H = 7.16, 2.62
cv = Canvas(W, H)
ACT = {'node': (C['node'], 'committee nodes'), 'plant': (C['plant'], 'plants'), 'all': (C['aud'], 'every replica')}
FLOW = '#8a8a8a'

# step: number, title, two description lines, actor, icon
main = [('1', 'Beacon', 'reveal the next', 'hash-chain value', 'node', 'dice-5'),
        ('2', 'Round keys', 'post fresh signed', 'ML-KEM-768 keys', 'node', 'key-variant'),
        ('3', 'Messages', 'post proof, encrypted', 'rows and digests', 'plant', 'email-lock'),
        ('4', 'Counting', 'verify every proof', 'against own inputs', 'all', 'magnify-scan'),
        ('5', 'Share check', 'each node checks', 'its own row', 'node', 'clipboard-check-outline'),
        ('7', 'Certificate', '$q$ nodes sign', 'the valid set $V$', 'node', 'file-certificate'),
        ('8', 'Sums', 'one sum per node;', 'robust decoding', 'node', 'sigma'),
        ('9', 'Close', 'same model $W_{t+1}$;', 'secrets erased', 'all', 'cube-outline')]
X0, DX, Y, R = 0.46, 0.893, 1.30, 0.20
xs = [X0 + k * DX for k in range(len(main))]


def node(x, y, n, ic, actor, r=R):
    col = ACT[actor][0]
    cv.ax.add_patch(Circle((x, y), r, fc='white', ec=col, lw=1.1, zorder=4))
    cv.icon(ic, x, y, r * 1.05, col, z=5)
    cv.badge(x - r * 0.78, y + r * 0.78, n, r=0.068, size=5.8, z=8)


for x, (n, title, d1, d2, actor, ic) in zip(xs, main):
    node(x, Y, n, ic, actor)
    cv.text(x, Y - 0.33, title, size=6.9, weight='bold', color=ACT[actor][0])
    cv.text(x, Y - 0.50, d1, size=5.9, color='#333333')
    cv.text(x, Y - 0.63, d2, size=5.9, color='#333333')
for a, b in zip(xs[:-1], xs[1:]):
    cv.arrow(a + R + 0.02, Y, b - R - 0.02, Y, color=FLOW, lw=0.9, ms=6)
cv.text((xs[4] + xs[5]) / 2, Y + 0.07, 'no complaint', size=5.3, color=FLOW, style='italic')

# ------------------------------------------------------------------ branch: resolution of a complaint
bx, by = (xs[4] + xs[5]) / 2, 2.00
node(bx, by, '6', 'lock-open-variant', 'plant', r=0.17)
cv.arrow(xs[4] + 0.10, Y + R - 0.01, bx - 0.17, by - 0.03, color=C['bad'], lw=0.9, ms=6, ls=(0, (3, 1.6)), rad=-0.25)
cv.arrow(bx + 0.17, by - 0.03, xs[5] - 0.10, Y + R - 0.01, color=FLOW, lw=0.9, ms=6, ls=(0, (3, 1.6)), rad=-0.25)
cv.text(xs[4] - 0.02, by - 0.06, 'complaint', size=5.7, color=C['bad'], style='italic', ha='right')
cv.text(xs[5] + 0.14, by + 0.08, 'Resolution', size=6.9, weight='bold', color=C['plant'], ha='left')
cv.text(xs[5] + 0.14, by - 0.07, 'plant reveals the seed; anyone rebuilds the row', size=5.9, color='#333333', ha='left')
cv.text(xs[5] + 0.14, by - 0.21, 'bad row: plant struck; good row: node removed', size=5.9, color=C['bad'], ha='left')

# ------------------------------------------------------------------ phases
phases = [('Round setup', xs[0], xs[1]), ('Messages and checks', xs[2], xs[4]),
          ('Certification and opening', xs[5], xs[7])]
for lab, a, b in phases:
    cv.ax.plot([a - 0.38, b + 0.38], [2.42, 2.42], color='#bdbdbd', lw=0.7, zorder=1)
    cv.text((a + b) / 2, 2.53, lab, size=7.2, weight='bold', color=C['ink'])

# ------------------------------------------------------------------ legend: who acts
lx = 1.60
cv.text(lx - 0.10, 0.22, 'Acting party:', size=6.0, ha='right', color='#333333')
for k, key in enumerate(('node', 'plant', 'all')):
    col, lab = ACT[key]
    x = lx + k * 1.15
    cv.ax.add_patch(Circle((x, 0.22), 0.05, fc='white', ec=col, lw=1.1, zorder=4))
    cv.text(x + 0.10, 0.22, lab, size=6.0, ha='left', color=col)
cv.text(5.95, 0.22, 'every step is an entry on the public log', size=5.8, color=FLOW, style='italic')
cv.ax.set_ylim(0.10, H); cv.fig.set_size_inches(W, H - 0.10)
cv.save(sys.argv[1] if len(sys.argv) > 1 else 'fig4_round.png')
