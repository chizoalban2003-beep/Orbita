# Why Orbita is not a deep learning model

## The short version

Deep learning is a *learned* function approximator. You feed it data, it
adjusts millions of weights, and at the end you have a function that maps
inputs to outputs. Nobody can explain individual weights. The function works
when test data resembles training data and fails opaquely when it doesn't.

Orbita is a *mechanistic* simulator. You feed it the structure of the system
(what are the possible outcomes? what forces act on it?) and it simulates
forward. The parameters are physical: a mass, a drag coefficient, a position.
You can read them off the model and know why a prediction came out the way
it did.

Both approaches forecast probabilistic events. They are fundamentally
different in what they claim to know.

## Why we picked the mechanistic path

1. **Probabilistic events are not image classification.** The reason deep
   learning works for vision is that you have millions of labeled images.
   You do not have millions of nearly-identical Premier League matches.
   You have ~50 relevant ones per team per season. Small-data forecasting
   rewards strong priors. Physics is a strong prior.

2. **Explanations matter more than accuracy on the margin.** A broadcast
   producer who shows a 73% win probability needs to be able to say *why*.
   "Because the model thinks so" is not an answer. "Because the home team's
   morale-drag has trapped them in a low-energy orbit" is.

3. **Generalization across domains is a goal, not an accident.** The same
   gravitational structure that models a football match models an election,
   a market open, or a hurricane's landfall. A neural network trained on
   football does none of the others without retraining.

4. **The math is honest about uncertainty.** Bayesian posteriors give you
   a distribution, not a point estimate. The calibration loop tells you
   which intangibles are well-measured and which are guesses.

## When deep learning would be the right choice

- You have ≥ 10⁴ training samples that are i.i.d. with the events you'll
  forecast.
- You don't care why the model makes its decisions.
- The system you're modeling has no useful physical or causal structure
  to exploit.

None of those apply to the problems Orbita targets.

## Where we'll use learned components

Mechanistic doesn't mean stone-age. The roadmap includes:

- A small MLP to model *non-linear interactions* between intangibles
  (e.g., morale × fatigue may not be additive), trained on the residuals
  the physics engine can't explain.
- Hamiltonian Neural Networks as an opt-in replacement for the analytical
  potential, useful when the well shape itself is uncertain.
- Learned features for the drag ontology (e.g., crowd-dB from a stadium
  audio embedding).

These are tools in service of the physics, not replacements for it.
The core predictor remains mechanistic.
