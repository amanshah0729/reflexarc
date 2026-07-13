"""Smoke test: one undisturbed rollout, with the tactile channel logged.

Checks the three things that would invalidate everything downstream:
  1. the policy still solves the task through this runner (not just Faultline's)
  2. the fingerpads report force during the carry, not zeros
  3. the friction-cone ratio is finite and below 1 while the grasp holds
"""

import sys

import numpy as np

from reflexarc.runner import PolicyRunner

CKPT = sys.argv[1] if len(sys.argv) > 1 else "ishandotsh/act_libero_spatial_test"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

r = PolicyRunner(checkpoint=CKPT, n_action_steps=20, max_steps=400)
roll = r.run(seed=SEED)
r.close()

fn = roll.tactile.array("fn")
ncon = roll.tactile.array("n_contact")
cone = roll.tactile.array("cone_ratio")
h = roll.tactile.array("oracle_obj_height")
held = ncon > 0

print(f"\nsuccess={roll.success}  steps={roll.steps}  wall={roll.wall_time:.1f}s")
print(f"steps with both pads in contact: {int(held.sum())} / {len(held)}")
if held.any():
    i0, i1 = int(np.argmax(held)), len(held) - 1 - int(np.argmax(held[::-1]))
    print(f"grasp window: steps {i0}..{i1}")
    print(f"grip normal force  min/med/max: "
          f"{fn[held].min():.4f} / {np.median(fn[held]):.4f} / {fn[held].max():.4f} N")
    print(f"cone ratio         min/med/max: "
          f"{cone[held].min():.3f} / {np.median(cone[held]):.3f} / {cone[held].max():.3f}")
    print(f"object height rise: {(h[held].max() - h[held].min())*1000:.1f} mm")
else:
    print("!! no two-pad contact recorded at any step -- sensor model is not reading")

print("\nfirst 6 steps after grasp onset:")
if held.any():
    for s in range(i0, min(i0 + 6, len(held))):
        row = roll.tactile.rows[s]
        print(f"  step {s:3d}  fn={row['fn']:.4f}  ft={row['ft']:.4f}  "
              f"cone={row['cone_ratio']:.3f}  ncon={int(row['n_contact'])}  "
              f"slip={row['oracle_slip_speed']:.4f} m/s")
