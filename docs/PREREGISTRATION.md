# Pre-registration: the breaking-load experiment

Written and committed **before** the experiment ran. This file is not edited
after the fact; if the analysis had to change, that is recorded as an amendment
at the bottom with the reason.

The reason for writing it: this project's findings log already contains three
retractions, all of the same kind — a number was interpreted before what
produced it was checked. F2a read a detector as a slip precursor when it was
reporting a grasp that had already half-failed. F4a reported a control arm that
had welded the gripper shut. The episode-level detection analysis in F1 claimed
950 ms of warning from a detector that was simply always on.

There is also a specific risk in *this* experiment. Five setups have now been
tried before this one, and all five produced nulls. Continuing to change the
setup until a reflex looks effective is the failure mode this design exists to
avoid, so the prediction and the analysis are fixed here in advance.

## Question

Under a disturbance the policy cannot anticipate, does a fast sensorimotor loop
below the policy improve grasp retention, and does its **rate** matter?

## Setup

`libero_object` task 0, SmolVLA, nominal mass and friction. The grasp is
acquired at full strength; once the object is 2 cm clear of the table, its mass
ramps linearly from x1 to x400 over 250 control steps (12.5 s).

This timing is the point. Every earlier setup here fixed the difficulty at
reset, which made the object hard to *acquire* rather than hard to *hold*, and
F10 records that acquisition always failed first as a result. The slip
literature does not do that: it grasps normally and then pours rice into the
container. This is that protocol.

Load is the disturbance and grip force is the response, deliberately on
different channels — a reflex that restores the exact quantity being removed
would be undoing its own perturbation.

## Outcome

**Breaking load**: the mass multiplier in force at the step the gripper loses
contact. Continuous, one number per episode, rather than a bit.

Episodes that never lose contact are **right-censored** at the ramp maximum.
They are not discarded and not counted as failures; survival analysis is the
tool that uses them correctly.

## Analysis, fixed in advance

- Kaplan-Meier estimate of breaking load per arm, censoring at x400.
- Median breaking load from the KM curve.
- Log-rank test of each reflex arm against `policy`, two-sided.
- Significance threshold **p < 0.05**, Holm-corrected across the arms compared
  to `policy`.
- n = 15 seeds per arm, identical seeds across arms.

Task success is reported alongside but is **not** the primary outcome: an arm
can hold on longer and still fail the task, and F4 recorded exactly that when
an arm-arrest reflex converted drops into never-lifted.

## Arms

| arm | what it is |
|---|---|
| `policy` | baseline, no reflex |
| `reflex@1Hz` | grip-force reflex, evaluated once per 500 physics substeps |
| `reflex@5Hz` | once per 100 substeps |
| `reflex@20Hz` | once per 25 substeps — the policy's own action rate |
| `reflex@100Hz` | once per 5 substeps |
| `reflex@500Hz` | every substep — the physics rate |
| `yoked` | same intervention, same number of activations per seed, times drawn without consulting the sensor |
| `oracle` | fires on the true load rather than on the tactile signal — an upper bound on what any detector could achieve |

The rate ladder is one implementation at five decimations, so **rate is the only
thing that varies** across those five. Earlier comparisons here used two
different classes for 20 Hz and 500 Hz, which confounded rate with
implementation.

`yoked` is the arm that matters most. It has already overturned one result in
this project: a grip-force reflex that took success from 5/30 to 20/30 was
matched at 18/30 by the same intervention applied at random times, which
converted "a tactile reflex quadruples success" into "extra grip force at
arbitrary moments does the same thing".

## Predictions

Registered before running.

1. **Breaking load is flat in reflex rate above ~20 Hz.** Specifically, no
   significant difference between `reflex@20Hz`, `reflex@100Hz` and
   `reflex@500Hz` after correction.
2. **`yoked` matches the sensed arms.** No significant difference between
   `yoked` and the best sensed arm.
3. **The sensed arms beat `policy` on breaking load.** This is the one I expect
   to be positive — grip force is the channel that worked in F4, and this design
   finally gives it a disturbance it can act against.
4. **`oracle` beats every measurable arm**, which would locate the limit in the
   detector rather than in the response.

Prediction 3 being positive together with 2 being null is the outcome I think
most likely: the intervention helps, and the sensing does not explain why.

## Stopping rule

If this design returns a null for the sensed arms, that is the end of the
setup-modification sequence. Six setups will have been tried, each fixing an
identified flaw in the previous one, and continuing past that is no longer
debugging the experiment — it is searching for a configuration that produces
the answer I want.

## Known limitation, unchanged

MuJoCo infers contact force from geometry penetration, which the tactile-sim
literature describes as adequate for normal force and sparse for shear, with
slip and rotation the specific operations rigid simulators handle badly. This
design answers where a fast loop belongs in the control stack. It does not
produce a quantitative claim about real slip.
