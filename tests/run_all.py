"""Run every test file and write a summary to runs/tests_summary.txt."""
import os, subprocess, sys, time
files = ['test_field.py', 'test_hashing.py', 'test_ligero.py', 'test_sharing_pq.py', 'test_dp.py', 'test_step.py', 'test_step_variants.py',
         'test_step_attacks.py', 'test_malformed.py', 'test_fallback.py', 'test_protocol.py', 'test_protocol_attacks.py',
         'test_erasure.py', 'test_step_iiot_shapes.py', 'test_parameter_checks.py', 'test_relay_attacks.py',
         'test_log_driven_parties.py', 'test_public_surface.py', 'test_wire_and_harness.py', 'test_honest_liveness.py',
         'test_output_stage_cheats.py', 'test_norm_bound.py', 'test_constraint_coverage.py']
files += [f for f in sys.argv[1:]]
os.makedirs('runs', exist_ok=True)
out = []
for f in files:
    t0 = time.time()
    r = subprocess.run([sys.executable, 'tests/' + f], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ['(no output)'])[-1]
    out.append('%s  %-28s %6.1fs  %s' % ('PASS' if r.returncode == 0 else 'FAIL', f, time.time() - t0, last[:110]))
    if r.returncode != 0:
        out.append('      ' + (r.stderr.strip().splitlines() or [''])[-1][:200])
    print(out[-1], flush=True)
open('runs/tests_summary.txt', 'w').write('\n'.join(out) + '\n')
sys.exit(1 if any(l.startswith('FAIL') for l in out) else 0)
