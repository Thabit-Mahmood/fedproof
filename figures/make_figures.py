"""Draws every figure of the paper into figures/out/.

Usage: python3 figures/make_figures.py [result folder, default results]
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
results = sys.argv[1] if len(sys.argv) > 1 else 'results'
for name in ('fig1_trust', 'fig2_system', 'fig3_message', 'fig4_round'):
    subprocess.run([sys.executable, os.path.join(HERE, name + '.py'), os.path.join(OUT, name + '.png')], check=True, cwd=HERE)
subprocess.run([sys.executable, os.path.join(HERE, 'results_figures.py'), results, OUT], check=True)
