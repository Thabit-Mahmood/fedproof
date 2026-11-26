"""Fig. 3: what one argument proves, and what the proof-carrying message contains."""
import sys
from figkit import Canvas, C

W, H = 7.16, 2.30
cv = Canvas(W, H)
SEC, PUB = C['secret'], '#5a5a5a'

# ------------------------------------------------------------------ inputs (tight boxes; orange fill marks secret data)
cv.text(0.62, 2.20, 'Inputs', size=7.4, weight='bold')
inputs = [('database', SEC, 'Committed chunk', '$b$ records'),
          ('cube-outline', PUB, 'Public model $W_t$', 'same for every plant'),
          ('dice-5', SEC, 'Noise bits $\\rho$', 'fixed before the beacon'),
          ('access-point-network', PUB, 'Beacon bits $\\beta$', '$\\mathrm{SHAKE}(\\mathrm{beacon}_t, i)$')]
for k, (ic, col, a, b) in enumerate(inputs):
    y = 1.86 - k * 0.42
    cv.box(0.04, y - 0.17, 1.16, 0.34, fc=C['secret_bg'] if col == SEC else 'white', ec=col, lw=0.6, r=0.03)
    cv.icon(ic, 0.19, y, 0.19, col)
    cv.text(0.33, y + 0.07, a, size=6.4, ha='left', weight='bold', color=col)
    cv.text(0.33, y - 0.08, b, size=5.7, ha='left', color='#333333')
cv.ax.plot([1.24, 1.28, 1.28, 1.24], [1.94, 1.94, 0.53, 0.53], color='#8a8a8a', lw=0.7, zorder=2)
cv.arrow(1.28, 1.25, 1.40, 1.25, color='#8a8a8a', lw=0.8, ms=5)

# ------------------------------------------------------------------ the argument (dashed outline, white)
FX, FW, FY, FH = 1.42, 3.90, 0.62, 1.50
cv.box(FX, FY, FW, FH, fc='white', ec=C['plant'], lw=0.8, ls=(0, (4, 2)), r=0.04)
cv.text(FX + 0.08, FY + FH - 0.12, 'One zero-knowledge argument over $\\mathbb{F}_p$ (P128 or P256)', size=6.8,
        ha='left', weight='bold', color=C['plant'])
train = [('Forward', 'clamped ReLU'), ('Error', 'smoothed hinge'), ('Backward', 'exact floors'),
         ('Clip', '$\\|s_jG_j\\|\\leq\\hat C$'), ('Sum, truncate', '$\\lfloor\\sum_j s_jG_j/2^{F_u}\\rfloor$'),
         ('Add noise', '$\\sum 2^{r\'}(\\rho\\oplus\\beta)$')]
bw, gap = 0.565, 0.066
ty = 1.58
for k, (a, b) in enumerate(train):
    x = FX + 0.07 + k * (bw + gap)
    cv.box(x, ty - 0.19, bw, 0.38, fc='white', ec=C['plant'], lw=0.6, r=0.03, z=3)
    cv.text(x + bw / 2, ty + 0.08, a, size=6.0, weight='bold', color=C['plant'])
    cv.text(x + bw / 2, ty - 0.09, b, size=5.5, color='#333333')
    if k < len(train) - 1:
        cv.arrow(x + bw, ty, x + bw + gap, ty, color=C['plant'], lw=0.7, ms=5, z=4)
share = [('$u$', 'update'), ('Packed shares', 'blocks of $k_p$'), ('Row for node $m$', 'shares, one-time masks'),
         ('Masked digests', '$D_{m,g}=\\langle\\gamma_g,\\phi_m\\rangle+\\psi_{m,g}$')]
xs = [4.67, 3.75, 2.72, 1.49]
ws = [0.54, 0.84, 0.95, 1.13]
sy = 0.98
for (a, b), x, w in zip(share, xs, ws):
    cv.box(x, sy - 0.19, w, 0.38, fc='white', ec=C['node'], lw=0.6, r=0.03, z=3)
    cv.text(x + w / 2, sy + 0.08, a, size=6.1, weight='bold', color=C['node'])
    cv.text(x + w / 2, sy - 0.09, b, size=5.5, color='#333333')
for k in range(3):
    cv.arrow(xs[k], sy, xs[k + 1] + ws[k + 1], sy, color=C['node'], lw=0.7, ms=5, z=4)
cv.arrow(4.94, ty - 0.19, 4.94, sy + 0.19, color=C['plant'], lw=0.7, ms=5)

# ------------------------------------------------------------------ encryption, outside the argument
ex = 2.72
cv.box(ex, 0.12, 0.95, 0.34, fc='white', ec=SEC, lw=0.6, r=0.03, z=3)
cv.icon('lock', ex + 0.13, 0.29, 0.15, SEC)
cv.text(ex + 0.55, 0.36, 'ML-KEM-768 +', size=5.9, weight='bold', color=SEC)
cv.text(ex + 0.55, 0.21, 'AES-256-GCM', size=5.7, color=SEC)
cv.arrow(ex + 0.47, sy - 0.19, ex + 0.47, 0.47, color=SEC, lw=0.7, ms=5)
cv.text(ex + 1.03, 0.29, 'one row per node, outside the argument', size=5.8, ha='left', style='italic', color='#555555')

# ------------------------------------------------------------------ the message (list with a bracket, no box)
MX = 5.55
cv.arrow(FX + FW, 1.37, MX - 0.08, 1.37, color=C['plant'], lw=0.9)
cv.text(MX, 2.20, 'Proof-carrying message', size=7.4, weight='bold', ha='left', color=C['plant'])
cv.ax.plot([MX - 0.04, MX - 0.04], [0.62, 2.06], color=C['plant'], lw=0.8, zorder=2)
items = [('file-certificate', C['plant'], 'Proof $\\pi$'), ('email-lock', SEC, '$K$ encrypted share rows'),
         ('clipboard-check-outline', C['node'], 'Masked digests $D_{m,g}$'), ('dice-5', SEC, 'Root of the next noise bits'),
         ('draw-pen', C['plant'], 'ML-DSA-65 signature on ctx')]
for k, (ic, col, lab) in enumerate(items):
    y = 1.90 - k * 0.30
    cv.icon(ic, MX + 0.14, y, 0.17, col)
    cv.text(MX + 0.30, y, lab, size=6.3, ha='left')

# ------------------------------------------------------------------ legend
cv.box(0.04, 0.17, 0.15, 0.11, fc=C['secret_bg'], ec=SEC, lw=0.6, r=0.02)
cv.text(0.24, 0.225, 'secret to the plant', size=5.8, ha='left')
cv.box(1.18, 0.17, 0.15, 0.11, fc='white', ec=PUB, lw=0.6, r=0.02)
cv.text(1.38, 0.225, 'public', size=5.8, ha='left')
cv.save(sys.argv[1] if len(sys.argv) > 1 else 'fig3_message.png')
