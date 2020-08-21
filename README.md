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

```
conda create --name hilo
conda activate hilo
pip install stable-baselines3[extra]
```

You will need to rerun `conda activate hilo` whenever you are running this.
