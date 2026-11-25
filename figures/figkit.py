"""Drawing helpers for the design figures: inch coordinates, Apache-2.0 icons (Pictogrammers Material Design Icons)."""
import io, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import cairosvg
from PIL import Image

ICONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 7.5, 'savefig.dpi': 600})

# one palette for every figure in the paper
C = dict(plant='#1f4e8c', plant_bg='#e8eef7', node='#1d7a5a', node_bg='#e5f3ec', log='#6b6b6b', log_bg='#f2f2f2',
         relay='#b03a2e', aud='#6a3d9a', aud_bg='#f0eaf6', secret='#d9822b', secret_bg='#fbeedd', public='#4d4d4d',
         pub_bg='#ffffff', bad='#b03a2e', good='#2e7d32', ink='#1a1a1a', soft='#8c8c8c', step='#d9822b')

_cache = {}


def icon_img(name, color='#1a1a1a', px=256):
    key = (name, color, px)
    if key not in _cache:
        svg = open(os.path.join(ICONS, name + '.svg')).read()
        svg = re.sub(r'<path ', '<path fill="%s" ' % color, svg)
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=px, output_height=px)
        _cache[key] = np.asarray(Image.open(io.BytesIO(png)).convert('RGBA'))
    return _cache[key]


class Canvas:
    """A figure whose axes coordinates are inches from the bottom-left corner."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis('off')

    def icon(self, name, x, y, s, color='#1a1a1a', z=5):
        """Icon of size s inches centred at (x, y)."""
        self.ax.imshow(icon_img(name, color), extent=(x - s / 2, x + s / 2, y - s / 2, y + s / 2), zorder=z,
                       interpolation='lanczos')

    def box(self, x, y, w, h, fc='white', ec='#333333', lw=0.7, r=0.05, ls='-', z=1, alpha=1.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % r, fc=fc, ec=ec, lw=lw, ls=ls,
                           zorder=z, alpha=alpha)
        self.ax.add_patch(p)
        return p

    def text(self, x, y, s, size=7.5, color='#1a1a1a', ha='center', va='center', weight='normal', style='normal',
             z=6, bg=None, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, style=style, zorder=z, **kw)
        if bg:
            t.set_bbox(dict(fc=bg, ec='none', pad=0.6))
        return t

    def arrow(self, x0, y0, x1, y1, color='#333333', lw=0.9, style='-|>', ms=7, ls='-', rad=0.0, z=3, shrink=0):
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                            connectionstyle='arc3,rad=%g' % rad, zorder=z, shrinkA=shrink, shrinkB=shrink)
        self.ax.add_patch(a)

    def badge(self, x, y, s, fc=None, r=0.075, size=6.2, z=8):
        fc = fc or C['step']
        self.ax.add_patch(Circle((x, y), r, fc=fc, ec='white', lw=0.6, zorder=z))
        self.text(x, y - 0.004, s, size=size, color='white', weight='bold', z=z + 1)

    def pill(self, x, y, s, fc, size=6.8, color='white', padx=0.07, h=0.16, z=7, weight='bold'):
        t = self.text(x, y, s, size=size, color=color, weight=weight, z=z + 1)
        self.fig.canvas.draw()
        bb = t.get_window_extent().transformed(self.ax.transData.inverted())
        w = bb.width + 2 * padx
        self.box(x - w / 2, y - h / 2, w, h, fc=fc, ec='none', r=h / 2, z=z)
        return w

    def save(self, path):
        self.fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.015, facecolor='white')
        plt.close(self.fig)
