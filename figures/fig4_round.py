"""Fig. 4: one round as entries of the log, with what every replica checks and what a fault costs."""
import sys
from figkit import Canvas, C

W, H = 7.16, 2.62
cv = Canvas(W, H)
LX, X0 = 0.02, 0.66
CW = (W - X0 - 0.02) / 9
steps = [
    ('Beacon', 'nodes', 'Hash-chain values;\n$\\geq T_n{+}1$ must\nhash forward', 'Missing or wrong\nvalue: node sits\nout the round'),
    ('Round keys', 'nodes', 'Signature and\nFIPS 203 key\ncheck', 'Key forged in a\nnode\'s name: node\ndisputes, removed'),
    ('Messages', 'plants', 'Proof verified on\ninputs each replica\ncomputes itself', 'Signed proof that\nfails: plant\nstruck'),
    ('Counting', 'replica', 'First signed copy\nwith expected\ncontext only', 'Replays, copies\nand junk are\nignored'),
    ('Share check', 'nodes', 'Each node: its own\nrow against the\nmasked digests', 'Bad row: node\nsigns a\ncomplaint'),
    ('Resolution', 'plants', 'Seed revealed;\nanyone rebuilds\nthe ciphertext', 'Bad row: plant\nstruck. Good row:\nnode removed'),
    ('Certificate', 'nodes', '$|V|\\geq A_{\\min}$; $q$ nodes\nsign $H(t,V,\\mathrm{head})$\nfor their own $V$', 'Fewer than $A_{\\min}$\nplants: round\naborts'),
    ('Sums', 'nodes', 'Robust decoding\nof the sums\nover $V$', 'Wrong sum: node\nnamed; too many:\nabort, no blame'),
    ('Close', 'replica', 'Same update,\nstrikes and\nremovals', 'Node seized later:\nround keys already\nerased'),
]
icon_of = {'nodes': ('server-security', C['node']), 'plants': ('factory', C['plant']), 'replica': ('cogs', C['aud'])}
# row geometry
ys = dict(head=2.42, who=2.08, check=1.55, fault=0.90)
cv.text(LX + 0.02, ys['who'], 'Acts', size=6.6, ha='left', weight='bold')
cv.text(LX + 0.02, ys['check'], 'What is\nchecked', size=6.6, ha='left', weight='bold')
cv.text(LX + 0.02, ys['fault'], 'A fault\ncosts', size=6.6, ha='left', weight='bold', color=C['bad'])
# the log as a timeline of entries
cv.ax.plot([X0 - 0.02, W - 0.02], [ys['head'] - 0.19, ys['head'] - 0.19], color=C['log'], lw=0.6, zorder=1)
for k, (name, who, chk, fault) in enumerate(steps):
    x = X0 + k * CW + CW / 2
    cv.box(x - CW / 2 + 0.02, ys['head'] - 0.11, CW - 0.04, 0.22, fc=C['log_bg'], ec=C['log'], lw=0.5, r=0.03, z=3)
    cv.badge(x - CW / 2 + 0.11, ys['head'], str(k + 1), r=0.07, size=5.8)
    cv.text(x + 0.06, ys['head'], name, size=6.5, weight='bold', z=6)
    ic, col = icon_of[who]
    cv.icon(ic, x, ys['who'] + 0.02, 0.21, col)
    cv.text(x, ys['who'] - 0.17, {'nodes': 'committee', 'plants': 'plants', 'replica': 'every party'}[who], size=5.7,
            color=col)
    cv.box(x - CW / 2 + 0.02, ys['check'] - 0.27, CW - 0.04, 0.54, fc='white', ec='#c8c8c8', lw=0.5, r=0.03, z=2)
    cv.text(x, ys['check'], chk, size=5.75, linespacing=1.15)
    cv.box(x - CW / 2 + 0.02, ys['fault'] - 0.27, CW - 0.04, 0.54, fc='#fbecea', ec='#e2b3ad', lw=0.5, r=0.03, z=2)
    cv.text(x, ys['fault'], fault, size=5.75, color='#7d2a21', linespacing=1.15)
    if k < 8:
        cv.arrow(x + CW / 2 - 0.035, ys['head'], x + CW / 2 + 0.035, ys['head'], color=C['log'], lw=0.6, ms=4, z=4)
# bottom statement
cv.box(0.02, 0.04, W - 0.04, 0.42, fc=C['log_bg'], ec=C['log'], lw=0.5, r=0.05)
cv.icon('router-network', 0.24, 0.25, 0.22, C['relay'])
cv.text(0.45, 0.25, 'A relay that withholds an entry can only delay the round or cost a plant its turn: missing messages and '
        'reveals are never penalised,\nbecause anyone can check a signed fault in the log but nobody can tell who '
        'caused an omission. No fault changes an opened sum.', size=6.2, ha='left', color=C['ink'])
cv.save(sys.argv[1] if len(sys.argv) > 1 else 'fig4_round.png')
