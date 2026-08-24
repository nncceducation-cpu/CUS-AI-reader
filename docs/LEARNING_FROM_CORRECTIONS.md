# Learning from corrections

The AI proposes a grade, you correct it, the next study is graded better. This
note is about what stands between that sentence and a tool that actually
improves, because the obvious implementation of it reliably makes accuracy worse
while its own numbers look better.

## How to use it

1. Load a study and run the model.
2. **Record your grade before looking at the AI output.** Do the expert read
   first, in the Expert grading tab.
3. Open **Correct and retrain**, tick that you were blind to the AI, enter your
   reader code and the infant code, and add the study.
4. When enough corrections have accumulated, press **Refit**. You see, per
   finding, what changed and why, or why nothing did.
5. Press **Adopt** to put the new settings into use. Roll back any time.

The infant code matters more than it looks. Studies from one infant are not
independent observations, and the held-out estimates are computed by holding out
whole infants. Entering the same code for two different babies quietly inflates
every number the panel shows you.

## What actually gets learned

Not network weights. Corrections arrive a handful at a time and weights need
thousands of examples, so nothing here touches them. What is fitted is the layer
between the model's probabilities and the consensus rules:

| Corrections in the smaller class | What unlocks |
|---|---|
| fewer than 8, or fewer than 3 infants | nothing, and the panel says so |
| 8 or more | the decision threshold for that finding |
| 20 or more | threshold plus Platt calibration |

The durable value is the log itself. Every correction is a labelled example, and
the **Export as training set** button is what eventually lets a model be trained
on Canadian consensus targets rather than borrowed from another population.

## The four things stopping this from going wrong

### Nothing is scored on the data it was fitted to

Refit a threshold on twelve corrections, report accuracy on those same twelve,
and agreement looks excellent every time while the tool gets worse. Every
candidate is scored by leave-one-infant-out cross-validation instead. The numbers
in the panel are all out of fold.

### Beating the incumbent is not enough

A threshold search on scores that carry no signal will find a degenerate rule
that calls every study positive, and under a cost that weights misses four times
higher than false alarms, that rule scores well. It is not learning, it is
surrender.

In testing, an earlier version of this module promoted parameters on **pure
noise every single time**. Three further conditions fixed it. A candidate must
also beat the best constant predictor; its out-of-fold ranking must clear an AUC
floor and survive a permutation test against shuffled labels; and the fitted
operating point must not give every study the same answer. Noise now promotes in
roughly 1 run in 60, which is what a 5% permutation threshold should produce.

### Sensitivity has a floor, not a price

A cost function trades sensitivity for specificity whenever the arithmetic
favours it. For a screen whose job is not to miss severe preterm brain injury,
that trade needs a limit, and the limit is a clinical decision. The slider in the
panel sets it, defaulting to 0.95. Nothing is adopted whose out-of-fold
sensitivity falls below it, and nothing is adopted on fewer than 12 held-out
positive studies.

The floor is checked as a point estimate. The 95% lower bound is computed and
displayed next to it, always, because "sensitivity 1.00" on fourteen positives is
consistent with a true sensitivity near 0.75 and should never be read as a
guarantee. Requiring the *bound* to clear 0.95 would need roughly seventy
consecutive correct held-out positives and would keep the gate shut for years, so
the tool reports the bound honestly rather than pretending to a guarantee it
cannot yet support.

### Your corrections must be independent of the AI

A reader who sees the AI grade and adjusts it is not producing a label, they are
producing a lightly edited copy of the model's own output. Fitting on those
teaches the model to agree with itself, and agreement climbs while accuracy does
not. Corrections are eligible for fitting only when the reader recorded the grade
blind. Unblinded ones are kept in the log for audit and excluded from every fit,
and the panel shows both counts.

## What it does in practice

Driven end to end against a deliberately miscalibrated detector, scored on a
fixed set of 80 studies never used for fitting:

| | Agreement with the expert |
|---|---|
| Before any correction | 41% |
| After 60 corrections | 96% |

The adopted fit had out-of-fold AUC 0.998, permutation p = 0.002, sensitivity
0.966 with a lower bound of 0.828 on 29 held-out positives.

Across 40 repeated runs at varying stream lengths, the median held-out cost fell
monotonically with more corrections and no run was harmed at 100 corrections or
more. Reproduce with `pytest tests/test_learning.py`.

These are simulations of the fitting layer against a synthetic detector. They
show the loop is sound. They say nothing about how well any model reads an
ultrasound, and they are not a clinical validation.

## The audit trail

`data/corrections.jsonl` is append-only. `models/learned_parameters.json` holds
the current settings plus the last twenty versions, and the shipped manifest is
never rewritten, so what the vendor stated and what your site learned stay
visible side by side. Every adopted change records the corrections behind it, the
held-out scores, and the sensitivity floor in force at the time.

## Source

Mohammad K, Scott JN, Leijser LM, et al. Front Pediatr. 2021;9:618236.
doi:10.3389/fped.2021.618236
