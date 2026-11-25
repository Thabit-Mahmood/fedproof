# Results of the paper

These files hold every number reported in the paper. Each line of a `.jsonl` file is one run, with its settings, its
measurements, the version of the code that produced it and the time it finished. `run.log` lists the order and
duration of the experiments.

| File | Contents |
|---|---|
| `tuning_<dataset>.json` | Clipping threshold and learning rates chosen on the public 5% of each dataset |
| `E1_detection.jsonl` | Detection without privacy noise: float and integer baselines and our system (Table II) |
| `E2_privacy.jsonl`, `E2_hmin.jsonl`, `eps_cache.json` | Detection against the privacy budget, the honest-plant sweep and the privacy accountant's values (Fig. 5) |
| `E3_integrity.jsonl` | The test suite: files passed and protocol attack scenarios (Section VI) |
| `E4_poisoning.jsonl` | Poisoning by corrupted plants against six aggregation rules (Fig. 6) |
| `E5_proofcost.jsonl` | Proving time, verification time and message size (Table III, Fig. 7) |
| `E6_network.jsonl`, `trace_<dataset>.json` | Live and emulated rounds on industrial links, with the traces the emulator replays (Fig. 8) |
| `E7_scaling.jsonl` | Round time from 10 to 100 plants (Fig. 9) |
| `E8_endtoend.jsonl` | End-to-end training with real proofs, the bit-for-bit check and the audit (Table IV) |
| `E9_membership.jsonl` | Membership inference against the trained models (Section VI) |

The code versions recorded in the rows refer to the research tree before it was reorganised for publication; the
reorganisation changed file names and layout only. Rows marked `996c864e2cb0` (all of E9 and the E6 runs on
kernel-shaped links) come from exactly this code. Rows marked `e44d98934a74` come from the version before it, which
differed only in how the membership attack draws its member and non-member records.
