"""Fig. 3: what one argument proves, and what the proof-carrying message contains."""
import sys
from figkit import Canvas, C

W, H = 7.16, 2.86
cv = Canvas(W, H)
SEC, PUB = C['secret'], '#5a5a5a'

# ------------------------------------------------------------------ inputs
cv.text(0.66, 2.74, 'Inputs', size=7.2, weight='bold')
inputs = [('database', SEC, 'Committed chunk', '$b$ records, committed\nbefore training'),
          ('cube-outline', PUB, 'Public model $W_t$', 'identical for every\nplant (in ctx)'),
          ('dice-5', SEC, 'Noise bits $\\rho$', 'committed before\nthe beacon'),
          ('access-point-network', PUB, 'Beacon bits $\\beta$', '$\\mathrm{SHAKE}(\\mathrm{beacon}_t, i)$')]
for k, (ic, col, a, b) in enumerate(inputs):
    y = 2.36 - k * 0.52
    cv.box(0.04, y - 0.22, 1.24, 0.44, fc=C['secret_bg'] if col == SEC else 'white', ec=col, lw=0.6, r=0.04)
    cv.icon(ic, 0.22, y, 0.22, col)
    cv.text(0.38, y + 0.09, a, size=6.5, ha='left', weight='bold', color=col)
    cv.text(0.38, y - 0.09, b, size=5.7, ha='left', color='#333333', linespacing=1.05)

cv.ax.plot([1.31, 1.35, 1.35, 1.31], [2.58, 2.58, 0.88, 0.88], color='#777777', lw=0.7, zorder=2)
cv.arrow(1.35, 1.73, 1.42, 1.73, color='#777777', lw=0.8, ms=5)
# ------------------------------------------------------------------ the argument
FX, FW = 1.42, 3.86
cv.box(FX, 0.60, FW, 2.14, fc='#fafcff', ec=C['plant'], lw=0.9, ls=(0, (4, 2)), r=0.06)
cv.pill(FX + FW / 2, 2.74, 'One zero-knowledge argument over $\\mathbb{F}_p$  (P128 or P256)', C['plant'], size=6.6)
train = [('Forward', 'clamped ReLU'), ('Error', 'smoothed hinge'), ('Backward', 'exact floors'),
         ('Clip each record', '$\\|s_jG_j\\|\\leq\\hat C$ exactly'), ('Sum, truncate', '$\\lfloor\\sum_j s_j G_j/2^{sh}\\rfloor$'),
         ('Add noise', '$\\nu=\\sum 2^{r\'}(\\rho\\oplus\\beta)$')]
bw, gap = 0.565, 0.066
for k, (a, b) in enumerate(train):
    x = FX + 0.07 + k * (bw + gap)
    cv.box(x, 2.00, bw, 0.46, fc='white', ec=C['plant'], lw=0.6, r=0.04, z=3)
    cv.text(x + bw / 2, 2.34, a, size=5.6 if len(a) > 12 else 5.9, weight='bold', color=C['plant'])
    cv.text(x + bw / 2, 2.13, b, size=5.3, color='#333333')
    if k < len(train) - 1:
        cv.arrow(x + bw, 2.23, x + bw + gap, 2.23, color=C['plant'], lw=0.7, ms=5, z=4)
cv.text(FX + 0.09, 1.86, 'Training step in exact integers: every rounding a floor, every bound checked by bit decomposition',
        size=5.8, ha='left', style='italic', color='#333333')
# sharing row
share = [('$u$', 'update vector'), ('Packed shares', 'blocks of $k_p$,\ndegree $<w$'),
         ('Row for node $m$', 'shares $s_m$ and\none-time masks $r_m$'),
         ('Masked digests', '$D_{m,g}=\\langle\\gamma_g,s_m\\rangle+r_{m,g}$')]
xs = [4.67, 3.75, 2.75, 1.49]
ws = [0.54, 0.84, 0.92, 1.18]
for (a, b), x, w in zip(share, xs, ws):
    cv.box(x, 1.08, w, 0.52, fc='white', ec=C['node'], lw=0.6, r=0.04, z=3)
    cv.text(x + w / 2, 1.47, a, size=6.2, weight='bold', color=C['node'])
    cv.text(x + w / 2, 1.24, b, size=5.6, color='#333333', linespacing=1.05)
for k in range(3):
    cv.arrow(xs[k], 1.34, xs[k + 1] + ws[k + 1], 1.34, color=C['node'], lw=0.7, ms=5, z=4)
cv.arrow(4.94, 2.00, 4.94, 1.60, color=C['plant'], lw=0.7, ms=5)
cv.text(1.51, 0.84, 'Also proved: opened data equal the\ncommitted chunk, re-committed under\nfresh salts for later rounds',
        size=5.6, ha='left', style='italic', color='#333333', linespacing=1.1)
cv.text(3.12, 0.84, 'Challenge $\\gamma$ is derived only after the\nencrypted rows are absorbed into the\ntranscript, so rows cannot be fitted to it',
        size=5.6, ha='left', style='italic', color='#333333', linespacing=1.1)

# ------------------------------------------------------------------ encryption (outside the argument)
cv.box(2.43, 0.10, 1.04, 0.40, fc=C['secret_bg'], ec=SEC, lw=0.6, r=0.04, z=3)
cv.icon('lock', 2.55, 0.30, 0.15, SEC)
cv.text(2.99, 0.37, 'ML-KEM-768 +', size=5.9, weight='bold', color=SEC)
cv.text(2.99, 0.21, 'AES-256-GCM per node', size=5.5, color='#333333')
cv.arrow(2.95, 1.08, 2.95, 0.51, color=SEC, lw=0.7, ms=5)
cv.text(3.57, 0.30, 'outside the argument; a disputed\nrow is settled by revealing its seed', size=5.6,
        ha='left', color='#555555', style='italic')

# ------------------------------------------------------------------ the message
MX = 5.42
cv.box(MX, 0.60, W - MX - 0.02, 2.14, fc='#f6f9fd', ec=C['plant'], lw=0.6, r=0.06)
cv.pill(MX + (W - MX) / 2, 2.74, 'Proof-carrying message', C['plant'], size=6.6)
items = [('file-certificate', C['plant'], 'Proof $\\pi$'), ('email-lock', SEC, '$K$ encrypted share rows'),
         ('clipboard-check-outline', C['node'], 'Masked digests $D_{m,g}$'), ('dice-5', SEC, 'Root of next noise bits'),
         ('draw-pen', C['plant'], 'ML-DSA-65 signature on ctx')]
for k, (ic, col, lab) in enumerate(items):
    y = 2.40 - k * 0.30
    cv.icon(ic, MX + 0.18, y, 0.17, col)
    cv.text(MX + 0.33, y, lab, size=6.3, ha='left')
cv.text(MX + 0.10, 0.76, 'ctx binds registry, $t$, $i$, $H(W_t)$, beacon,\nchunk index and the round keys',
        size=5.6, ha='left', color='#444444', linespacing=1.1)
cv.arrow(FX + FW, 1.95, MX, 1.95, color=C['plant'], lw=0.9)
# measured cost, from Table III
cv.box(MX, 0.10, W - MX - 0.02, 0.40, fc='white', ec='#bdbdbd', lw=0.5, r=0.04)
cv.text(MX + (W - MX) / 2, 0.30, 'Measured, three intrusion models ($b=128$,\nP128): message 2.1 to 2.3 MB; prove 3.4\nto 3.8 s, verify 1.8 to 2.1 s on one core',
        size=5.5, color='#333333', linespacing=1.1)
# legend
cv.box(0.04, 0.08, 0.16, 0.12, fc=C['secret_bg'], ec=SEC, lw=0.6, r=0.02)
cv.text(0.25, 0.14, 'secret to the plant', size=5.8, ha='left')
cv.box(0.04, 0.27, 0.16, 0.12, fc='white', ec=PUB, lw=0.6, r=0.02)
cv.text(0.25, 0.33, 'public', size=5.8, ha='left')
cv.save(sys.argv[1] if len(sys.argv) > 1 else 'fig3_message.png')
