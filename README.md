# hilo
The high-low guessing game.

The question in my mind is, to what extent do deep learning techniques
"understand" numbers? It is a bit surprising, as I write this in 2020,
that advanced AI techniques like GPT-3 often do not correctly handle
the sort of basic arithmetic that a calculator can master.

Unfortunately, at least for me, wondering about GPT-3 or reading
research papers about neural nets handling numerical problems is like
staring into an abyss. I observe the situation, come to no conclusion,
and think of no next steps. I cannot "poke" it to see what
happens. The logic is opaque and unpokable.

So I am trying to think of a good numerical playground. The high-low
game has a very simple optimized solution - guess the number precisely
in the middle of the possible range. To calculate this requires doing
`(low + high) / 2`. Is this the simplest game where playing it
optimally requires an addition? I can't think of a simpler one
offhand, except perhaps trivial ones like simply asking what `x + y`
is.

Anyway, I am going to try some standard reinforcement learning
algorithms on this, and see if different ways of representing the
numbers makes anything work better.

# Installation

Get the `stable-baselines` dependencies described [here](sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev).

Install `conda`, I've been using `miniconda` but I don't think it matters.

Then install the `hilo` conda environment from the repo's `environment.yml`:

```
conda env create
```

Then, separately install `stable-baselines3`.

```
pip install stable-baselines3
```

If you make any updates to the environment, be sure to update `environment.yml` with:

```
conda env export --name hilo > environment.yml
```
