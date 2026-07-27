# Findings log

Every claim with its status and the evidence behind it. Entries are written
when the measurement happens, including the ones that killed a plan.

Status vocabulary: **CONFIRMED** (measured, holds), **SUPPORTED** (measured,
one condition), **CORRECTED** (claimed, then measured and revised),
**FALSIFIED**, **OPEN**.

Everything below is `libero_spatial` task 0 unless stated, on MuJoCo 3.8.1 /
robosuite 1.4.0 / LeRobot 0.4.4, ACT (52M) unless a policy is named.

---

## Calibration

### C1. LIBERO objects are ~200x too light to make grasp stability testable — **CONFIRMED**

The akita black bowl of `libero_spatial` task 0 has mass **0.0056 kg**. Its
weight is 0.055 N. Measured grip normal force during a carry is 1.8-4.9 N
against a finger friction coefficient of 1.0-2.0, so the grasp can resist
several newtons of tangential load. The friction-cone ratio |Ft| / (mu |Fn|),
which reaches 1.0 at the point of sliding, sits at **~0.08** through a nominal
carry.

Nothing the policy does can lose that object, so no part of the benchmark score
depends on holding on to it. A robustness result measured on this task is
silent about grasp stability by construction.

Scaling every free object's mass, 15 seeds per cell:

| factor | bowl | success | never lifted | lifted then lost | carried, missed |
|---|---|---|---|---|---|
| x1 | 5.6 g | 13/15 | 0 | 1 | 1 |
| x50 | 0.28 kg | 12/15 | 3 | 0 | 0 |
| x100 | 0.56 kg | 12/15 | 3 | 0 | 0 |
| x150 | 0.84 kg | 9/15 | 3 | 2 | 1 |
| x200 | 1.12 kg | 3/15 | 4 | **6** | 2 |
| x300 | 1.68 kg | 0/15 | 7 | 5 | 3 |

x1 giving 13/15 is consistent with the sibling `Faultline` repo's F10, which
measured ACT at 18/20 on this task at the same `n_action_steps`.

At x100 the cone ratio during the carry rises to **0.74-0.97** -- a marginal
grasp -- and the policy still succeeds 12/15. Real failures begin at x200,
where 6/15 episodes lift the object and then lose it.

*Why it matters*: mass is the only perturbation axis in this literature that
leaves the observation untouched. Camera shift, lighting, texture, blur and
instruction corruption all break the policy by corrupting its input. A mass
change leaves every pixel identical and alters only what the world does, which
is the exact failure mode a reflex is supposed to address. It is not fully
unobservable -- a loaded arm tracks its commands differently, so the
information reaches proprioception -- but that is the same channel a human has.

*Operating point chosen*: **x200**, where 40% of episodes are lift-then-drop.

### C1a. No grasp in LIBERO is anywhere near failing — **CONFIRMED across 4 suites**

C1 measured one bowl on one task. This is the general version.

The metric is the **grasp safety factor**: the tangential load the contact can
carry before sliding, over the object's weight.

    S = mu (Fn_left + Fn_right) / (m g)

S = 1 is the point of slipping. Grasp controllers, biological and robotic, are
usually described as operating somewhere around 1.5-3.

SmolVLA, 65 measured carries across 15 tasks in all four suites, nominal
settings, measuring only successful carries (a failed grasp has no margin to
report):

| suite:task | object | mass | Fn total | **S** | pad share |
|---|---|---|---|---|---|
| 10:1 | butter | 5.2 g | 39.1 N | **1350** | 0.98 |
| object:1 | cream cheese | 6.2 g | 37.1 N | **1017** | 0.97 |
| object:3 | bbq sauce | 12.2 g | 33.8 N | **523** | 1.00 |
| object:0 | alphabet soup | 25.9 g | 39.5 N | **303** | 0.99 |
| object:2 | salad dressing | 21.8 g | 34.1 N | **290** | 0.95 |
| goal:2 | wine bottle | 15.4 g | 15.1 N | **199** | 0.81 |
| spatial:0-3 | akita black bowl | 5.6 g | 4.5-5.2 N | **144-178** | 0.82-1.00 |
| goal:1, goal:3 | akita black bowl | 5.6 g | 3.7-4.8 N | **127-157** | 0.85-1.00 |
| 10:2 | moka pot | 68.2 g | 20.9 N | **63** | 1.00 |

**Median 221. Minimum 61. Not one carry in 65 falls below 3.**

![grasp safety factor](figures/grasp_audit.png)

Two independent causes, both worth stating. The objects are 10-100x lighter
than the things they depict -- a wine bottle at 15 g, a moka pot at 68 g, a can
of soup at 26 g. And the gripper closes far harder than the task needs, 34-40 N
of normal force on `libero_object`'s boxes and cans against objects weighing
under a fifth of a newton.

*Why it matters*: a manipulation benchmark whose grasps sit two orders of
magnitude inside the friction cone cannot score a policy on whether it holds
on. Nothing a policy does to the grasp is measurable, so nothing in the
reported number reflects grasp quality. It also explains F9 exactly -- there is
no in-transport slip to detect because there is no in-transport slip.

*Scope*: one policy (SmolVLA), 5 seeds per task, first four tasks of each
suite. Pad share is policy-dependent (C3), so the force distribution would
differ for ACT; the safety factor depends on total normal force and object
mass, which are properties of the gripper and the scene rather than of the
policy, and would move less.

### C2. The four failure modes must be reported separately — **method note**

`success` is not a usable outcome variable here. A reflex that arrests the arm
converts drops into timeouts, which raises no success rate but changes what
failed. Every result reports the breakdown:

- `never_lifted` — the grip never got the object off the table. Not addressable
  by a slip reflex; a floor effect of the mass setting.
- `lifted_lost` — lifted clear, then lost. **The only failure a grasp reflex
  could prevent.**
- `carried_missed` — the grasp held and the policy still failed the task.
- `success`.

---

## The gripper

### C3. LIBERO grasps load the finger shafts, not the fingertips — and it depends on the policy — **CONFIRMED**

Real tactile hardware sits on the fingertip. The Panda model has separate
collision geoms for the finger shaft (`gripper0_finger{1,2}_collision`) and the
pad (`..._pad_collision`), so which one carries the load is measurable.

On `libero_spatial` task 0 with **ACT**, every load-bearing contact during a
successful carry is on the shafts. The pads touch for a single step during
closure and then separate. A sensor restricted to the pads reads **zero
through an entire successful grasp** — the first version of the sensor model
here did exactly that and reported `ncon = 0` for a rollout that lifted the
bowl 103 mm and completed the task.

The bowl rim wedges between the fingers rather than being pinched by their
tips.

Share of grip normal force borne by the pads during the carry phase, **SmolVLA**,
4 seeds per task:

| suite | task | pad share |
|---|---|---|
| libero_spatial | 0 | 0.63, 0.25, 0.82, 1.00 |
| libero_spatial | 1 | 1.00, 0.53, 0.23, 1.00 |
| libero_spatial | 2 | — , 1.00, 0.21, 1.00 |
| libero_spatial | 3 | — , 0.86, 0.86, 1.00 |
| libero_object | 0 | 0.99, 0.97, … |

Two things follow. SmolVLA pinches with the pads on the same task and object
where ACT wedges on the shafts, so **whether a fingertip sensor sees anything
depends on the policy, not only on the object**. And `libero_object`, whose
objects are boxes and cans rather than bowls, gives near-pure fingertip grasps.

*Why it matters*: "add tactile sensing to this benchmark" is not a well-posed
instruction. Where the sensor should go depends on which policy is holding the
object, and a fingertip-only sensor is blind to a large fraction of grasps that
the benchmark scores as successes.

*Consequence for this project*: the sensor model reads all finger collision
geoms and reports the pad share separately, rather than assuming the fingertip
is where the information is.

---

## Methodological

### M1. `mj_setConst` silently destroys the simulation state — **CONFIRMED**

Changing `body_mass` leaves the solver's derived constants (`body_invweight0`,
`body_subtreemass`) stale, and `mj_setConst` is the documented refresh. But it
computes those constants *at* `qpos0` and uses `mjData` as scratch space, so it
overwrites the live state.

Measured: after the call, `qpos == model.qpos0` exactly, and every free object
teleports from z = 970 mm on the table to z = 0.

The episode then runs to completion and reports an ordinary success rate for a
scrambled scene. The first mass ladder run here read 0/3 at x50 and looked like
a real physical result. Fixed by saving `qpos`/`qvel` around the call and
calling `mj_forward` afterwards.

A test that applies the mass change and then re-applies the init state will not
catch this, because re-applying the state repairs the damage. The regression
has to assert on the state immediately after the mutation.

### M2. LIBERO caches absolute paths in `~/.libero/config.yaml` — **CONFIRMED**

Moving the project directory breaks every environment build with a path that
no longer exists. Loud rather than silent, so low severity, but it also breaks
any sibling project sharing the same interpreter.

---

## Detection

### F1. Force magnitude does not predict *when* the object will be lost — **CONFIRMED**

31 drop episodes and 29 hold episodes (mass x150 and x200 pooled, ACT, 80
rollouts). Step-level discrimination between "a drop follows within 500 ms"
and "an ordinary step of a marginal carry", as AUC:

| channel | pooled | x150 | x200 |
|---|---|---|---|
| `cone_ratio` |Ft| / (mu |Fn|) | 0.668 | 0.731 | 0.628 |
| `fn` grip normal force | 0.654 | 0.724 | 0.667 |
| `d_cone` | 0.628 | 0.615 | 0.646 |
| `ft` tangential force | 0.568 | 0.503 | 0.512 |
| `pad_fraction` | 0.542 | 0.541 | 0.497 |
| `d_fn` force collapse rate | 0.508 | 0.541 | 0.476 |

The best measurable channel reaches 0.67. For comparison, the sibling
`Faultline` repo's F9 reached AUC 0.61 on episode-level failure prediction and
was recorded as "real but weak, not usable".

An earlier episode-level pass looked far better -- a threshold on `d_fn`
detected 21/22 drops with a median lead of 19 control steps (950 ms). That was
an artifact. A carry window at this mass is only 13-22 steps, so a lead of 19
means the detector fires at the moment of lift and stays on; its false-alarm
rate was 43% against seven negatives. The step-level AUC is the honest version
of the same data.

*Interpretation*: the fingertip force signal reports that the grasp is loaded
near its friction limit, which at this mass is true for nearly every episode
including the ones that succeed. It says the grasp is marginal, not that a drop
is imminent.

*Caveat, unresolved*: `oracle_slip_speed` scores 0.313, consistently below
chance. That is almost certainly a defect in the oracle rather than a fact
about physics -- these traces differenced `data.cvel`, which is in the body's
own com-centred frame, against a world-frame hand velocity. Fixed in
`sense.py` via `mj_objectVelocity`, but the traces above predate the fix, so
**no claim is made here about what ground truth could predict**. The result
above concerns the measurable channels only.

### F2. The informative signal is where the contact is, not how hard it is — **SUPPORTED**

`pad_fraction` -- the share of grip normal force borne by the fingertips rather
than the finger shafts -- has an unremarkable AUC (0.542), but at a threshold
of 0.97 it catches **26 of 31 drops** with a per-episode false-alarm rate under
20%, at a median lead of **3 control steps (150 ms)**.

The mechanism is legible: these grasps begin as a wedge across the finger
shafts (C3). When the object slides down into a fingertip-only pinch, the load
migrates entirely to the pads, and shortly afterwards it is gone.

Two consequences.

A single force reading per finger cannot express this; it needs enough spatial
resolution to tell shaft contact from tip contact. That is an argument for
taxel arrays over load cells, made from a failure prediction rather than from
first principles.

And 150 ms sits in a specific gap: longer than one control step (50 ms), far
shorter than one action chunk at the default `n_action_steps = 20` (1000 ms).
If a reflex helps anywhere, this is the window where it must, because nothing
operating at the policy's replan rate can act inside it.

*Why only SUPPORTED*: one threshold, one task, one policy, chosen after seeing
the data. It needs a held-out check before it is worth more than a hypothesis.

### F2a. What that signal actually is — **CORRECTS F2**

I read `pad_fraction > 0.97` as the load migrating from the finger shafts onto
the fingertips: a grasp degenerating from a wedge into a pinch. That reading is
wrong, and the check that shows it is one line.

Across both trace sets, **68 steps** exceed the threshold. The number of those
steps at which *both* fingers carry any load is **zero**.

`pad_fraction > 0.97` does not describe a redistribution within an intact
two-finger grasp. It fires when one finger has already stopped touching the
object altogether, so all remaining force is trivially on the other finger's
pad. It is not a precursor to failure. It is a partial failure, roughly 150 ms
before the total one.

The 26/31 detection rate and the 150 ms lead in F2 are unchanged and still
real. What changes is what a reflex could do about it: at the moment the signal
appears, half the grasp is already gone.

This also explains F3 below, and it is the reason F2 keeps its number rather
than being deleted -- the measurement was right and the mechanism I attached to
it was not.

## The reflex

### F3. No reflex recovers any drops — and the first run could not have — **CONFIRMED**

Five arms, mass x200, 30 seeds each, identical seeds and initial states, ACT at
`n_action_steps = 20`. Detector `pad_fraction > 0.97`, hold 6 steps.

| arm | success | lifted_lost | never_lifted | fired/ep | Fisher p (drops) |
|---|---|---|---|---|---|
| policy alone | 5/30 | 17 | 8 | 0.0 | — |
| + reflex (freeze arm) | 6/30 | 15 | 9 | 19.3 | 0.796 |
| + replan trigger | 4/30 | 17 | 9 | 2.8 | 1.000 |
| + grip force x6 | 5/30 | 17 | 8 | 2.1 | 1.000 |
| yoked control | 4/30 | 18 | 8 | 1.7 | 1.000 |

Nothing moves. Not the reflex, not forcing the policy to re-plan, and not
reaching past the action space to raise the finger servo gain six-fold
(verified to work: peak grip force 18.3 N to 33.4 N).

**But the run does not support the conclusion it appears to, because the
detector fired in only 5 of 30 episodes** -- mean 19.3, median 0. `SlipReflex`
gates on `require_contact`, which asks that both fingers be touching the
object, and F2a establishes that the detector's operating point occurs only
when one finger has already let go. The reflex was blind to its own detector by
construction.

A repeat with the gate removed is running. Whatever it shows, the honest
statement about this table is "no effect, from a controller that almost never
ran", not "tactile reflexes do not help".

*A separate run with a chance-level detector* (`d_fn < -5 N/step`, AUC 0.508)
gave the same null across four arms -- 5/30, 5/30, 4/30, 5/30, every p = 1.000
-- while firing 13.9 times per episode. That one is uninformative by design and
is kept as the detector ablation.

*Known defect in the yoked control*: it fired 1.7 times per episode against the
reflex arm's 19.3, so the intervention budgets are not matched as intended.
`yoke()` caps the schedule at the width of the carry window, while the reflex
arm's count includes steps outside it. Since no arm differs from the policy,
this cannot hide a benefit, but the control is weaker than described and needs
fixing before any positive result would stand.

### F4. Ungated, one arm works — and the control shows the sensing did nothing — **CONFIRMED**

Repeat with `require_contact` removed, so the detector can fire when only one
finger is loaded. Same seeds, same mass, same detector.

| arm | success | lifted_lost | never_lifted | fired/ep | p vs policy (success) |
|---|---|---|---|---|---|
| policy alone | 5/30 | 17 | 8 | 0.0 | — |
| + reflex (freeze arm) | 5/30 | 5 | **17** | 179.5 | 1.000 |
| + grip force x6 | **20/30** | 4 | 6 | 22.7 | **0.0000** |
| grip force x6, always on | 0/30 | 0 | **30** | 400.0 | 0.052 |
| grip force x6, **timing shuffled** | **18/30** | 4 | 8 | 21.6 | **0.001** |

Three things, in order of how much they matter.

**The arm-arrest reflex cuts drops from 17 to 5 (p = 0.003) and helps nobody.**
Never-lifted rises from 8 to 17 and success does not move. It prevents drops by
preventing the lift. This is exactly the artifact the outcome taxonomy in C2
exists to catch, and a paper reporting "drops reduced 70%" from this arm would
be reporting a robot that stopped doing the task.

**Grip force works, and it is specifically the part the action space cannot
express: success 5/30 to 20/30, drops 17 to 4.**

The arm labelled `squeeze` does two things at once. It raises the finger servo
gain six-fold, which no policy can request, and it also forces the gripper
command closed on the steps it is active, which any policy could do --
`SqueezeReflex` inherits `close_gripper` from `SlipReflex`. Those two halves
imply opposite conclusions about where the limit lies, so a control at
`force_gain = 1.0` was run: same detector, same force-close, no boost.

| | success | dropped | never lifted |
|---|---|---|---|
| policy alone | 5/30 | 17 | 8 |
| force-close only, no boost | 6/30 | 16 | 8 |
| force-close + servo gain x6 | **20/30** | 4 | 6 |

Forcing the gripper command closed does nothing (6/30 against 5/30). The entire
effect is the servo gain -- an authority that exists in the hardware, is
unreachable through LIBERO's binary gripper command, and that no policy trained
on this benchmark could ever have learned to ask for.

*Caveat*: the firing rates are not matched across these three arms (22.7, 64.0
and 0 activations per episode), because the intervention changes the dynamics
that drive the detector. The `squeeze` vs `yoked_squeeze` pair below is matched
by construction and is the comparison that carries weight.

**And the tactile signal contributed nothing to it.** The timing-shuffled
control -- same boost, same number of activations per seed, times drawn without
looking at the sensor -- scores 18/30 against the sensed arm's 20/30, with the
same 4 drops. The sensed and blind versions are indistinguishable.

So the honest reading of the one positive result in this project is: *the
gripper was under-powered for the object, and intermittently giving it more
force fixes the task*. When the reflex fired was irrelevant. What it was
responding to was irrelevant.

### F4a. The always-on control is invalid — **RETRACTED**

The `always_squeeze` arm scored 0/30 with all 30 episodes never lifting, and a
sweep of continuous gains reproduced it exactly at x1.5, x2.0 and x3.0 --
0/30, 0/30, 0/30, every episode never lifting.

Identical results across a four-fold range of gain is not a physical response,
it is a constant. A 1.5x stiffer position servo cannot prevent a robot from
picking anything up.

The cause is the same inherited `close_gripper` above. With `always=True` the
arm forces the gripper command closed on *every* step of the episode, including
the approach, so the hand can never open to take the object. The arm measures
"a robot whose gripper is welded shut", at any gain.

Both the always-on arm and the gain sweep are withdrawn. The claim they were
supposed to support -- that the intervention must be intermittent -- is
unsupported and currently untested.

This does not touch the `squeeze` versus `yoked_squeeze` comparison, which is
the load-bearing one: both arms carry the identical confound, applied the same
number of times, so the difference between them still isolates timing.

*This is the result the yoked control was built for.* Without it the table
above reads "a tactile reflex quadruples manipulation success", which is false.

### F5. Chunk blindness is not what loses the object — longer chunks are better — **CONFIRMED**

`n_action_steps` sets how long the policy's decision stays frozen. The premise
of this whole project is that a disturbance landing inside that window cannot
be answered by the policy. Measured at mass x200, 20 seeds per cell:

| chunk | open-loop | success | lifted_lost |
|---|---|---|---|
| 5 | 250 ms | 1/20 | 8 |
| 10 | 500 ms | 0/20 | 10 |
| 20 | 1000 ms | 4/20 | 11 |
| 50 | 2500 ms | **10/20** | **4** |

The relationship runs the wrong way. Committing to 2.5 seconds of open-loop
execution is the best setting tested, and replanning five times more often is
the worst. Under a dynamics perturbation the policy's corrections are worse
than no corrections: it observes a scene whose appearance is unchanged but
whose dynamics are not, and steers accordingly.

The sibling `Faultline` repo's F2 measured the opposite ordering on the same
task at nominal mass -- n=50 was the *worst* setting there, costing 27 points.
The reversal is the finding: chunk length interacts with the disturbance, and
"replan more often" is not a safe default.

*Caveat*: the reflex arm in this sweep used the gated detector from F3 and
fired zero times, so this table supports no claim about the reflex, only about
the policy.

### F6. A 25x faster reflex buys nothing — **CONFIRMED**

Every reflex above runs once per control step: 20 Hz, the same rate the
policy's actions are consumed at. Its only advantage over the policy is
re-deciding *within* a chunk. The biological claim is stronger than that, and
robosuite makes it reachable: `MujocoEnv.step` runs 25 physics substeps of 2 ms
per control step, and `_pre_action` is called on each one after `sim.forward()`,
so contact forces are current. Wrapping it gives a controller at **500 Hz
against a policy at 20 Hz**, and against a decision frozen for 1000 ms at the
default chunk length.

Same detector (`cone_ratio > 0.9`), same response (finger servo gain x6), same
seeds, mass x200, 30 seeds per arm. The 20 Hz arm modifies servo gain only
(`--no-force-close`), so the two differ in rate alone.

| arm | rate | success | lifted_lost | never_lifted |
|---|---|---|---|---|
| policy alone | — | 5/30 | 17 | 8 |
| reflex, grip x6 | 20 Hz | 21/30 | 1 | 8 |
| reflex, grip x6 | **500 Hz** | 23/30 | 1 | 6 |
| reflex, grip x1 | 500 Hz | 5/30 | 17 | 8 |

21/30 against 23/30, one drop each. **Rate is not the variable.** Twenty-five
times the control bandwidth is worth nothing measurable here, while the same
loop with its force channel removed reproduces the policy baseline to the
episode (5/30, 17, 8) -- which also serves as a no-op check on the
instrumentation, since a hook that quietly perturbed the simulation could not
land on the baseline exactly.

Taken with F4: the reflex needs an actuator channel, and does not need speed.
The useful framing is not "fast loop versus slow loop" but "is grip force
addressable at all". On a real Panda it is -- the gripper firmware closes a
force loop at roughly 1 kHz. LIBERO's binary command hides it, and no policy
trained through that interface can request it.

### F7. You cannot induce slip in a form-closure grasp by pulling sideways — **CONFIRMED**

The timed external wrench was built as the instrument for a latency question --
a disturbance with a known onset makes detection latency, response latency and
"did the response beat the replan" all measurable -- and then went unused,
because mass scaling produced failures more conveniently. Mass is the wrong
disturbance for that question: it is persistent, so the grasp is marginal from
the moment it closes and there is no onset to measure against. That is F1's
AUC of 0.5 restated as a design error.

Firing the impulse instead, on healthy grasps (mass x50, 12 seeds, ACT):

| impulse | success | lifted_lost |
|---|---|---|
| 0 N | 9/12 | 0 |
| 4 N | 8/12 | 1 |
| 8 N | 9/12 | 0 |
| 16 N | 9/12 | 0 |
| 32 N | 7/12 | 2 |
| 64 N | 5/12 | 4 |

A **64 N** lateral tug on a 0.28 kg object dislodges it in 4 of 12 episodes.
For scale, the grasp's Coulomb capacity is roughly 10-20 N (grip normal force
5 N against pad friction 1-2), and 64 N applied for 200 ms is enough impulse to
give the free object about 45 m/s.

The grasp survives because it is not held by friction. C3 established that the
bowl rim wedges between the finger shafts; a wedge resists lateral load
geometrically, and no tangential force pulls an object out of one. Sliding it
out would require pulling along the wedge axis.

*Consequence*: `libero_spatial` task 0 cannot host a slip experiment at all,
whatever the disturbance. Every negative result above (F3, F4, F6) was measured
on a grasp with no slip mode to detect. They remain correct about what they
tested -- an arm-motion reflex, a replan trigger, and control rate, on a
form-closure grasp losing its geometry -- and they are not evidence about
tactile slip reflexes on friction grasps.

The experiment moves to `libero_object`, where C3 measured a pad share of
0.94-0.99: fingertip pinches held by friction, which is the regime the tactile
literature is about and the only one where "slip" is the right word.

### F8. On a friction grasp the test is underpowered, not negative — **INCONCLUSIVE**

`libero_object` task 0, SmolVLA, fingertip pinches (pad share 0.94-0.99),
friction scaled to an effective coefficient of 0.2, 20 seeds per arm.

| arm | rate | success | lifted_lost | never_lifted |
|---|---|---|---|---|
| policy alone | — | 14/20 | 2 | 4 |
| grip x6 | 20 Hz | 12/20 | 2 | 6 |
| grip x6 | 500 Hz | 13/20 | **0** | 7 |
| grip x1 | 500 Hz | 14/20 | 2 | 4 |

No comparison approaches significance (p = 0.49 to 1.00), and the reason is
design rather than physics: the baseline drops the object in 2 of 20 episodes,
so a reflex that prevented *every* drop would move the metric by 2/20, which
this n cannot resolve. The 500 Hz arm did reach 0 drops, but it converted them
into failed lifts (never_lifted 4 to 7), leaving success flat.

Two things this does establish. The no-boost arm reproduces the policy exactly
again (14/20, 2, 4), so the 500 Hz hook remains a verified no-op without its
force channel. And the failure budget at this friction setting is dominated by
`never_lifted` -- the object escaping during closure rather than during
transport -- which is a slip a grasp reflex might address but only before the
lift, not during the carry the detector was designed for.

A properly powered repeat is running at an effective coefficient of 0.04 with
40 seeds on the policy-versus-500 Hz pair.

*Recorded as inconclusive rather than negative.* An underpowered null is not
evidence of absence, and F3/F4/F6 above are only entitled to their conclusions
because their effects were large or their controls were exact.

### F9. LIBERO grasps do not slip in transport — **CONFIRMED, and it subsumes F3-F8**

The powered repeat at an effective friction coefficient of **0.04** -- a 50x
reduction, a pinch on a surface roughly as slick as wet ice -- moved the
failure mode but not to slip:

| arm | success | lifted_lost | never_lifted |
|---|---|---|---|
| policy alone | 29/40 | **1** | 10 |
| grip x6 @ 500 Hz | 26/40 | **0** | 14 |

One drop in forty. Lowering friction makes the gripper fail to *acquire* the
object, not to keep it.

Pooling every rollout run for this project, 1,292 in total:

| suite | rollouts | lifted then dropped |
|---|---|---|
| `libero_object` — fingertip pinch, friction to x0.02, impulses to 16 N | 280 | **12 (4%)** |
| `libero_spatial` — form-closure wedge, mass to x300 | ~800 | ~300 (38%) |

The only regime that produces in-transport loss at any rate is the one where
the grasp is a rim wedged between the finger shafts, carrying an object 200x
heavier than the benchmark specifies, and losing its *geometry* rather than
sliding (F7: a 64 N lateral tug dislodges it 4 times in 12).

**So LIBERO cannot host a tactile slip-reflex experiment.** Not because the
disturbance was wrong -- mass, friction and a timed external wrench were all
tried, across two suites, two policies and 1,292 episodes -- but because the
carry phase is not where these grasps fail. Failures live at closure, before a
stable grasp exists, which is a grasp-selection problem rather than anything a
reflex can reach.

That is the answer to the question this project set out to ask, and it is a
statement about the benchmark rather than about reflexes. A fair test needs a
simulator or a robot where a held object can slide: deformable or textured
contact, a gripper whose force is commandable, and objects with realistic mass
and friction. None of those are properties of LIBERO.

### F10. Every dial that weakens the grasp breaks acquisition before it breaks retention — **CONFIRMED**

The obvious objection to F9 is: fine, no grasp slips at the default settings --
so turn a dial until one does. There are three dials. All three were turned,
and all three fail the same way.

**Grip strength.** The cleanest lever, because it drives the safety factor to 1
without touching the object, the scene, or the policy's input distribution --
unlike mass, which reaches the slip regime only past physical realism. Scaling
the finger servo gain, `libero_object` task 0, SmolVLA, 10 seeds:

| grip x | servo kp | success | lifted_lost | never_lifted |
|---|---|---|---|---|
| 1 | 1000 | 9/10 | 0 | 1 |
| 0.7 | 700 | 9/10 | 0 | 1 |
| 0.5 | 500 | 7/10 | 2 | 1 |
| 0.35 | 350 | 8/10 | 0 | 2 |
| 0.25 | 250 | 8/10 | 0 | 2 |
| **0.15** | 150 | **0/10** | 1 | **9** |
| 0.1 | 100 | 0/10 | 2 | 8 |
| 0.03 | 30 | 0/10 | 0 | 10 |
| 0.01 | 10 | 0/10 | 0 | 10 |

The transition is a cliff between x0.25 and x0.15: 8/10 success to 0/10, and
the failures land almost entirely in `never_lifted`. **No setting produces more
than 2 drops in 10.** At x0.03 the safety factor is still about 9 -- nine times
what is needed to hold the object -- and the gripper cannot pick it up at all.

That is the mechanism. **Acquiring the object needs far more force than holding
it.** Closure has to overcome approach dynamics and settle the contact; once
closed, geometry and a huge margin do the rest. So the two thresholds are not
separated by a usable band -- weaken the grasp and acquisition fails first,
every time.

The other two dials do the same thing:

| dial | strongest setting reached | what happened |
|---|---|---|
| contact friction | x0.02 (effective mu 0.04) | 1 drop in 40; never_lifted 10 -> 14 |
| object mass | x200 (bowl at 1.12 kg) | 6 drops in 15 -- **but** on `libero_spatial`'s form-closure wedge, and past the mass of the real object |
| grip strength | x0.15 (kp 150) | 1 drop in 10, 9 never lifted |

Only the mass dial produces retention failures in quantity, and only on the
suite where the grasp is a rim wedged between the finger shafts losing its
geometry rather than sliding (F7: a 64 N lateral tug dislodges that 4 times in
12). On the friction-pinch suite, where "slip" is the right word, nothing
produces it.

*This is the complete answer to the question the project set out to ask.* It is
not that LIBERO's default settings happen to be too forgiving -- that would be
fixable with a parameter. It is that **the parameter space contains no point
where these policies acquire an object and then lose it**, because the gripper's
margin for holding is far wider than its margin for grasping. A slip reflex has
nothing to act on anywhere in the reachable configuration space.

### F11. A tactile grip reflex does work — above 100 Hz, and only when the sensor drives it — **CONFIRMED (pre-registered)**

The first positive result in this project, from the first design capable of
showing one. Pre-registered in `docs/PREREGISTRATION.md`, committed before the
run, with two amendments both recorded from policy-only pilots.

Load-to-failure: the grasp is acquired normally, then the object's mass ramps
x1 to x400 over 100 control steps. The outcome is the **load at which the grasp
broke**, one continuous number per rollout, not a bit. `libero_object` task 0,
SmolVLA, 15 seeds per arm, 14 usable (one seed never lifts, in every arm).

| arm | median breaking load | activations/ep | log-rank vs policy (Holm) |
|---|---|---|---|
| policy | 312 | 0 | — |
| reflex @ 1 Hz | 300 | 1 | 1.000 |
| reflex @ 5 Hz | 316 | 7 | 1.000 |
| reflex @ 20 Hz | 320 | 18 | 1.000 |
| **reflex @ 100 Hz** | **392** | 55 | **0.0064** |
| **reflex @ 500 Hz** | **376** | 70 | **0.0298** |
| oracle (true load) | 372 | 1 | **0.0428** |
| **yoked** (same budget, blind timing) | **288** | 56 | 1.000 |

Paired by seed, which is the comparison that shows it is not carried by a few
episodes:

| arm | beats policy on | median delta |
|---|---|---|
| reflex @ 100 Hz | **11 / 14 seeds** | **+82** |
| reflex @ 500 Hz | **12 / 14 seeds** | **+64** |
| yoked | 7 / 14 seeds | +0 |

**Three claims, and the third is the one that matters.**

*Rate matters, with a knee between 20 and 100 Hz.* At the policy's own action
rate of 20 Hz the reflex is indistinguishable from no reflex (320 against 312).
At 100 Hz it holds 25% more load. A fast loop below the policy has to run
faster than the policy's action rate to be worth anything; running *at* that
rate is equivalent to not having one.

*The sensing matters.* `yoked` applies the same intervention with a matched
activation budget (56 against 55) at times drawn without the sensor, and lands
at 288 — at or below baseline, better than policy on exactly half the seeds.
Sensed-versus-yoked is p = 0.0102. **When you squeeze is the whole effect.**

*And the fingertip signal is as good as ground truth.* The oracle, firing on
the true load, reaches 372 against the tactile detector's 392. The detector is
not the bottleneck.

**This directly overturns F6**, which reported that a 25x faster reflex buys
nothing. F6 was measured on a binary outcome at a single fixed load, under a
mass set at reset. That design cannot see a 25% shift in breaking load, and its
disturbance had no onset for a detector to find. The earlier reading was not
wrong about its own data; the data could not answer the question.

*Scope, and it is narrow.* One task, one policy, n=14. Task success is 0/14 in
every arm -- the ramp is harsh by design, so this measures grasp retention and
says nothing about task completion. Six of fourteen episodes in the 100 Hz arm
break at exactly the ramp ceiling, so the true effect is bounded below by what
is reported here. And it remains MuJoCo's penetration-based contact, which the
tactile-sim literature describes as sparse in shear -- the architectural claim
travels, the number does not.

*Predictions 1 and 2 were both wrong*, which is the reason the pre-registration
was worth writing. I predicted breaking load would be flat in rate above 20 Hz
(it is not -- the knee is above 20 Hz) and that yoked would match the sensed
arms (it does not -- the sensing is the effect). Predictions 3 and 4 were right
and half-right: the sensed arms beat policy, and the oracle does not beat them.

## Methodological

### M3. Seeding from `hash()` is not reproducible across processes — **CONFIRMED, and it corrects two runs here**

Two runs of an identical impulse cell -- 8 N, mass x50, seeds 0-11, same code
-- returned 7/12 and 9/12 success with 2 and 0 drops.

`Impulse.unit()` drew its direction from
`np.random.default_rng(abs(hash(("impulse", self.seed))) % 2**32)`. Python
salts `str.__hash__` per interpreter (PEP 456, default since 3.3), so the tug
went a different way in every process. `yoke()` had the same defect.

The failure mode is the quiet one. Within a process everything is consistent,
so any determinism test that constructs two objects and compares them passes;
the property only breaks across a resume, a re-run, or someone else's machine.
Catching it requires a test that starts a second interpreter, which
`tests/test_rng.py` now does.

Fixed with a blake2b-derived seed (`src/reflexarc/rng.py`). Both impulse
calibrations were discarded and re-run.

**This defect is also present in the sibling `Faultline` repo**, at
`src/faultline/perturb/base.py:143`, and it reaches six perturbations there:
camera (rotation axis and shift direction), lighting (sign and direction),
texture (hue direction), noise (the noise field), blur (axis), and brightness
-- where the draw decides the *sign*, i.e. whether the image is brightened or
darkened. Demonstrated: the same `(brightness, magnitude=0.5, seed=0)` gives
sign -1 in one interpreter and +1 in the next.

That matters most for its four-model comparison, because each policy is a
separate CLI invocation and GR00T runs in a different virtual environment
entirely. Models compared under "identical pipeline, seeds and initial states"
were in fact perturbed in different directions.

---

## Prior-art check (2026-08-10)

Run after the experiments, which is the wrong order, and is the same process
failure the sibling `Faultline` repo records twice. Recorded in full because the
outcome changes what this project may claim.

**The headline hypothesis is not novel, and at least one group reports it
working.**

### The core idea — a fast tactile reflex layered under a slow policy

- [**Reactive Slip Control in Multifingered Grasping: Hybrid Tactile Sensing and
  Internal-Force Optimization**](https://arxiv.org/abs/2602.16127) (Feb 2026) is
  this project's hypothesis, built and reported as working: a low-level reflex
  layer driven by fast tactile feedback for multifinger grasp stabilisation,
  combining learned slip detection with model-based internal-force control to
  arrest in-hand slip while preserving the object-level wrench. Slip onset
  detected at **20.4 ± 6 ms**, grasp response ~30 ms, framed explicitly against
  human reflex baselines — the same biological argument this repo opens with.
- [**TactileReflex**](https://arxiv.org/abs/2605.23568) (May 2026) —
  noise-statistics-driven vision-tactile reflex control for force-sensitive
  manipulation.
- [**UniTacVLA**](https://arxiv.org/abs/2606.31723) (Jun 2026) — an
  action-tactile mixed controller supplying high-frequency closed-loop
  corrections *on top of the low-frequency action chunks* a VLA backbone emits.
  This project's architecture, named and published.
- [**TouchWorld**](https://arxiv.org/abs/2607.07287) (Jul 2026), **VLA-Touch**,
  and the [Awesome-Force-Tactile-VLA](https://github.com/OpenHelix-Team/Awesome-Force-Tactile-VLA)
  list, which is long enough to establish a populated field rather than an
  opening.

The README cites only the faster-brain side (real-time chunking, latent
world-model switching, interleaved correction) and needs the reflex side added.
**"Faster brain or faster spinal cord" is an active 2026 question, not an open
one.**

### "Where the contact is, not how hard" (F2, F2a)

Prior art, and older than the robotics work. In human motor control, tangential
loading produces **partial (incipient) slip at the periphery of the contact
patch** before gross slip, and the CNS is understood to use that signal to
modulate grip force ([Perception of partial slips under tangential loading of
the fingertip](https://www.nature.com/articles/s41598-018-25226-w), Sci Rep
2018; [Dynamics of fingertip contact during the onset of tangential
slip](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4191101/)). The
spatial-resolution conclusion in F2 — taxel arrays over load cells — is that
literature's standard position, reached here from a failure prediction instead.
Per-finger reflexive tactile control also exists ([In-Hand Object Stabilization
by Independent Finger Control](https://arxiv.org/abs/1806.05031)).

F2a already retracted the mechanism, so this costs little. It does mean F2's
sensor-design argument should be stated as *converging with* the tactile
literature, not as an independent one.

### Form closure versus force closure (F7)

Textbook grasping theory (Bicchi & Kumar, *Robotic grasping and contact: a
review*, ICRA 2000). That a wedge resists lateral load geometrically is not a
finding. That **`libero_spatial` task 0's grasps are form closure, so the
benchmark cannot host a slip experiment**, is not in the literature.

### Not found, after searching

1. **C1 — the mass argument.** "Simulators model contact poorly" is everywhere
   (RoboLab, most sim-to-real position papers). The quantified claim — objects
   ~200x too light, friction-cone ratio 0.08 through a nominal carry, therefore
   no part of the score depends on holding the object and every grasp-stability
   result on this benchmark is vacuous — was not found. Nor was the observation
   that mass is the only perturbation axis leaving the observation untouched.
2. **C3 — grasp geometry is policy-dependent.** ACT wedging on the finger shafts
   where SmolVLA pinches with the pads, same task, same object, so where a
   tactile sensor belongs depends on which policy is holding it. Not found; the
   tactile literature assumes the fingertip.
3. **The yoked timing-shuffled control (F4).** Every paper above reports a
   tactile reflex improving grasping. None found runs a blind control firing the
   same intervention the same number of times at moments drawn without
   consulting the sensor. This is the methodological contribution, and it is what
   turned a positive result here into a null.
4. **F5 — the chunk-length reversal.** Longer open-loop chunks performing better
   under a dynamics perturbation, inverting the same measurement at nominal mass
   in `Faultline` F2. Not found.
5. **F6 — rate is not the variable.** 500 Hz against 20 Hz buying nothing, while
   the same loop without its force channel reproduces the baseline exactly. The
   literature argues *for* latency; no negative control on latency was found.
6. **F9 — the benchmark verdict itself**, at 1,292 rollouts across two suites,
   two policies, mass, friction and impulse.

### What this project is, restated

Not "a fast reflex fixes chunk blindness". That is being pursued by several
groups with better hardware, and at least one reports it working.

What is defensible is **a benchmark critique with a negative result attached**:
LIBERO cannot test grasp stability at its own object masses (C1), cannot host a
slip experiment at any disturbance tried (F7, F9), its grasps are form closure
whose geometry depends on the policy (C3), the only intervention that helps is a
grip force the action space cannot express (F4, F6), and a blind control matches
the sensed one (F4).

Every reflex result here was measured on a grasp with no slip mode. They are
correct about what they tested and are **not** evidence about tactile slip
reflexes on friction grasps.

*Cost of running this late*: the reflex arms were built and debugged before it
was known that the task has no slip mode and that the architecture is already
published. Searching first would have moved the project to a different simulator
on day one, and framed it as a replication with controls rather than as an open
question.
