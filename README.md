# Compositional Sound Synthesis Networks (CSSN)

A Python reimplementation (with web UI) of:

> Björn Þór Jónsson, Amy K. Hoover, and Sebastian Risi. 2015. Interactively
> Evolving Compositional Sound Synthesis Networks. In *Proceedings of the
> Conference on Genetic and Evolutionary Computation* (GECCO 2015).
> https://doi.org/10.1145/2739480.2754796

The backend uses the
[neat-python](https://github.com/codereclaimers/neat-python) library. The
frontend is pure HTML/JS/CSS.

Note that we try to follow the paper as closely as possible, but there are a
few points of divergence, and a few places where we had to make
algorithm/parameter decisions for things not explicitly documented in the
paper. The config file and source code try to be very explicit about marking
everything as either being verbatim from a specific paper section or [NOT IN
PAPER].
