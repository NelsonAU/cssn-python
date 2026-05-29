"""
Minimal NEAT implementation for Compositional Sound Synthesis Networks.

Implements the core NEAT mechanisms described in:
    Stanley, K. O. & Miikkulainen, R. (2002). "Evolving Neural Networks through
    Augmenting Topologies." Evolutionary Computation 10(2), 99–127.

NeatConfig exposes the parameters most likely to need tuning; everything else
uses published NEAT defaults hard-coded as module-level constants.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


# Innovation tracking

class _InnovationDB:
    """
    Assigns a unique innovation number to every (in_node, out_node) pair.

    Innovation numbers allow NEAT to align genes that represent the same
    structural change across different genomes, enabling meaningful crossover
    even when those genomes have different topologies.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self._db: dict[tuple[int, int], int] = {}

    def get(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._db:
            self._counter += 1
            self._db[key] = self._counter
        return self._db[key]

    def reset(self) -> None:
        self._counter = 0
        self._db.clear()


_innovations = _InnovationDB()


# Gene types

@dataclass
class NodeGene:
    id:         int
    type:       str    # 'input' | 'output' | 'hidden'
    activation: str    # name of the activation function, e.g. 'sin'


@dataclass
class ConnectionGene:
    in_node:    int
    out_node:   int
    weight:     float
    enabled:    bool = True
    innovation: int  = 0   # assigned by _innovations.get() at creation


# Hard-coded NEAT parameters
#
# TODO?: it's possible we might want to expose some of these. For now a smaller
# subset of parameters are exposed in NeatConfig.

_WEIGHT_INIT_SIGMA    = 1.0   # σ for fresh weight samples
_WEIGHT_PERTURB_SIGMA = 0.5   # σ for perturbation (small nudge) mutations
_WEIGHT_MAX           = 30.0  # symmetric clamp applied to all weights
_WEIGHT_REPLACE_RATE  = 0.1   # fraction of weight mutations that fully replace the value
_P_ACTIVATION_MUTATE  = 0.05  # per-hidden-node probability of changing activation function
_COMPAT_C1            = 1.0   # disjoint/excess coefficient in δ  (Stanley & Mii. eq. 1)
_COMPAT_C2            = 0.5   # weight-difference coefficient in δ
_COMPAT_THRESHOLD     = 3.0   # δ below which two genomes are considered conspecific
_ELITISM              = 2     # fittest N genomes copied unchanged to next generation
_CROSSOVER_RATE       = 0.75  # probability that two parents are crossed (vs. clone)


class Genome:
    """
    A NEAT genome: a set of NodeGenes and ConnectionGenes encoding a
    feed-forward neural network topology.

    Genomes begin as minimal networks (all inputs wired directly to all outputs,
    no hidden nodes) and grow through structural mutations that add nodes and
    connections.
    """

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        self.num_inputs:  int = num_inputs
        self.num_outputs: int = num_outputs
        self.nodes:       dict[int, NodeGene]                   = {}
        self.connections: dict[tuple[int, int], ConnectionGene] = {}
        self.fitness:     float = 0.0

    @classmethod
    def create_minimal(
        cls,
        num_inputs:         int,
        num_outputs:        int,
        activation_options: list[str],
    ) -> "Genome":
        """
        Create the NEAT seed genome: all input nodes wired directly to all
        output nodes, no hidden nodes.

        This is the minimal topology described in Stanley & Miikkulainen (2002)
        §3, from which evolution grows the network through structural mutations.
        """
        g = cls(num_inputs, num_outputs)

        for i in range(num_inputs):
            g.nodes[i] = NodeGene(id=i, type="input", activation="linear")

        for j in range(num_outputs):
            nid = num_inputs + j
            g.nodes[nid] = NodeGene(
                id=nid,
                type="output",
                activation=random.choice(activation_options),
            )

        for i in range(num_inputs):
            for j in range(num_outputs):
                out_id = num_inputs + j
                inn = _innovations.get(i, out_id)
                g.connections[(i, out_id)] = ConnectionGene(
                    in_node=i,
                    out_node=out_id,
                    weight=random.gauss(0.0, _WEIGHT_INIT_SIGMA),
                    enabled=True,
                    innovation=inn,
                )

        return g

    def copy(self) -> "Genome":
        g = Genome(self.num_inputs, self.num_outputs)
        g.fitness = self.fitness
        for nid, n in self.nodes.items():
            g.nodes[nid] = NodeGene(n.id, n.type, n.activation)
        for key, c in self.connections.items():
            g.connections[key] = ConnectionGene(
                c.in_node, c.out_node, c.weight, c.enabled, c.innovation
            )
        return g

    def mutate(
        self,
        p_weight:           float,
        p_add_node:         float,
        p_add_conn:         float,
        activation_options: list[str],
    ) -> None:
        """
        Apply all NEAT mutations:
          1. Weight perturbation     (probability p_weight per connection)
          2. Add a hidden node       (probability p_add_node)
          3. Add a connection        (probability p_add_conn)
          4. Change a hidden activation (probability _P_ACTIVATION_MUTATE per node)
        """
        self._mutate_weights(p_weight)
        if random.random() < p_add_node:
            self._add_node(activation_options)
        if random.random() < p_add_conn:
            self._add_connection()
        self._mutate_activations(activation_options)

    def _mutate_weights(self, p: float) -> None:
        for c in self.connections.values():
            if random.random() < p:
                c.weight = _perturb(c.weight)

    def _add_node(self, activation_options: list[str]) -> None:
        """
        Node mutation: split one enabled connection A→B (weight=w) into
        A→new (weight=1) and new→B (weight=w), then disable A→B.

        The weights 1 and w keep the new node initially transparent — it passes
        its input unchanged — so the network's behaviour is preserved while
        subsequent weight mutations refine the new node's contribution.
        """
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return

        old = random.choice(enabled)
        old.enabled = False

        new_id = max(self.nodes) + 1
        self.nodes[new_id] = NodeGene(
            id=new_id,
            type="hidden",
            activation=random.choice(activation_options),
        )

        inn1 = _innovations.get(old.in_node, new_id)
        self.connections[(old.in_node, new_id)] = ConnectionGene(
            old.in_node, new_id, weight=1.0, enabled=True, innovation=inn1
        )
        inn2 = _innovations.get(new_id, old.out_node)
        self.connections[(new_id, old.out_node)] = ConnectionGene(
            new_id, old.out_node, weight=old.weight, enabled=True, innovation=inn2
        )

    def _add_connection(self) -> None:
        """
        Connection mutation: add a new feed-forward connection between two
        nodes that are not already connected.
        """
        input_ids = {n.id for n in self.nodes.values() if n.type == "input"}
        all_ids   = list(self.nodes)

        candidates = [
            (src, dst)
            for src in all_ids
            for dst in all_ids
            if src != dst
            and dst not in input_ids
            and (src, dst) not in self.connections
            and not self._would_cycle(src, dst)
        ]

        if not candidates:
            return

        src, dst = random.choice(candidates)
        inn = _innovations.get(src, dst)
        self.connections[(src, dst)] = ConnectionGene(
            src,
            dst,
            weight=random.gauss(0.0, _WEIGHT_INIT_SIGMA),
            enabled=True,
            innovation=inn,
        )

    def _would_cycle(self, src: int, dst: int) -> bool:
        """
        Return True if adding edge src→dst would introduce a directed cycle.

        Strategy: depth-first search from dst following existing enabled edges.
        If src is reachable from dst, the new edge would close a loop.
        """
        adj: dict[int, list[int]] = {nid: [] for nid in self.nodes}
        for c in self.connections.values():
            if c.enabled:
                adj[c.in_node].append(c.out_node)

        visited: set[int] = set()
        stack = [dst]
        while stack:
            node = stack.pop()
            if node == src:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj[node])
        return False

    def _mutate_activations(self, activation_options: list[str]) -> None:
        for n in self.nodes.values():
            if n.type == "hidden" and random.random() < _P_ACTIVATION_MUTATE:
                n.activation = random.choice(activation_options)

    def distance(self, other: "Genome") -> float:
        """
        NEAT compatibility distance δ (Stanley & Miikkulainen 2002, eq. 1):

            δ = c1·(E + D)/N + c2·W̄

        E = excess genes, D = disjoint genes,
        N = size of the larger genome (normalises for network size),
        W̄ = mean |Δweight| of matching genes.
        """
        inn_self  = {c.innovation: c for c in self.connections.values()}
        inn_other = {c.innovation: c for c in other.connections.values()}

        if not inn_self and not inn_other:
            return 0.0

        max_self  = max(inn_self,  default=0)
        max_other = max(inn_other, default=0)
        boundary  = min(max_self, max_other)  # innovations beyond this are excess

        excess = disjoint = 0
        weight_diffs: list[float] = []

        for inn in set(inn_self) | set(inn_other):
            in_s = inn in inn_self
            in_o = inn in inn_other
            if in_s and in_o:
                weight_diffs.append(abs(inn_self[inn].weight - inn_other[inn].weight))
            elif inn > boundary:
                excess += 1
            else:
                disjoint += 1

        N = max(len(inn_self), len(inn_other), 1)
        W = sum(weight_diffs) / len(weight_diffs) if weight_diffs else 0.0
        return _COMPAT_C1 * (excess + disjoint) / N + _COMPAT_C2 * W

def _perturb(v: float) -> float:
    """Perturb a weight or bias: 10% full replacement, 90% Gaussian nudge."""
    if random.random() < _WEIGHT_REPLACE_RATE:
        return random.gauss(0.0, _WEIGHT_INIT_SIGMA)
    return max(-_WEIGHT_MAX, min(_WEIGHT_MAX, v + random.gauss(0.0, _WEIGHT_PERTURB_SIGMA)))


def crossover(parent1: Genome, parent2: Genome) -> Genome:
    """
    NEAT crossover (Stanley & Miikkulainen 2002, §3.2).

    Connection genes are aligned by innovation number:
      * Matching genes (present in both): randomly inherit from either parent.
        If disabled in either parent, the gene stays disabled 75% of the time
        (avoids permanently losing connections).
      * Disjoint/excess genes: inherit from the more fit parent only
        (or from parent1 as an arbitrary tie-break when fitness is equal).

    Node genes are derived from the inherited connections; input/output nodes
    are always taken from the fitter parent.
    """
    fit_p  = parent1 if parent1.fitness >= parent2.fitness else parent2
    weak_p = parent2 if parent1.fitness >= parent2.fitness else parent1

    inn_fit  = {c.innovation: c for c in fit_p.connections.values()}
    inn_weak = {c.innovation: c for c in weak_p.connections.values()}

    child = Genome(parent1.num_inputs, parent1.num_outputs)

    for inn in set(inn_fit) | set(inn_weak):
        if inn in inn_fit and inn in inn_weak:
            src     = random.choice([inn_fit[inn], inn_weak[inn]])
            enabled = src.enabled
            if not inn_fit[inn].enabled or not inn_weak[inn].enabled:
                enabled = random.random() >= 0.75
        elif inn in inn_fit:
            src     = inn_fit[inn]
            enabled = src.enabled
        else:
            continue  # disjoint/excess from weaker parent → discard

        child.connections[(src.in_node, src.out_node)] = ConnectionGene(
            src.in_node, src.out_node, src.weight, enabled, src.innovation
        )

    required_ids = (
        {n.id for n in fit_p.nodes.values() if n.type in ("input", "output")}
        | {c.in_node  for c in child.connections.values()}
        | {c.out_node for c in child.connections.values()}
    )
    for nid in required_ids:
        node = fit_p.nodes.get(nid) or weak_p.nodes.get(nid)
        if node:
            child.nodes[nid] = NodeGene(node.id, node.type, node.activation)

    return child


class FeedForwardNetwork:
    """
    Evaluates a feed-forward NEAT genome in a single forward pass.

    Each non-input node computes:
        value = activation_fn(Σ weight_i · input_i)

    Nodes are processed in topological order (Kahn's algorithm) so that every
    input to a node is available before the node itself is evaluated.
    """

    def __init__(
        self,
        input_ids:  list[int],
        output_ids: list[int],
        order:      list[int],
        act:        dict[int, Callable[[float], float]],
        incoming:   dict[int, list[tuple[int, float]]],
    ) -> None:
        self._input_ids  = input_ids
        self._output_ids = output_ids
        self._order      = order
        self._act        = act
        self._incoming   = incoming

    @classmethod
    def create(
        cls,
        genome:           Genome,
        activation_funcs: dict[str, Callable[[float], float]],
    ) -> "FeedForwardNetwork":
        """Build a FeedForwardNetwork from a Genome and an activation-function map."""
        input_ids  = sorted(n.id for n in genome.nodes.values() if n.type == "input")
        output_ids = sorted(n.id for n in genome.nodes.values() if n.type == "output")

        incoming: dict[int, list[tuple[int, float]]] = {nid: [] for nid in genome.nodes}
        for c in genome.connections.values():
            if c.enabled:
                incoming[c.out_node].append((c.in_node, c.weight))

        # Kahn's topological sort — input nodes are pre-resolved.
        in_deps: dict[int, set[int]] = {nid: set() for nid in genome.nodes}
        for c in genome.connections.values():
            if c.enabled:
                in_deps[c.out_node].add(c.in_node)

        resolved  = set(input_ids)
        remaining = {nid for nid in genome.nodes if nid not in resolved}
        order: list[int] = []

        changed = True
        while changed and remaining:
            changed = False
            for nid in sorted(remaining):   # sorted → deterministic evaluation order
                if in_deps[nid].issubset(resolved):
                    order.append(nid)
                    resolved.add(nid)
                    remaining.discard(nid)
                    changed = True
        # Any nodes still in `remaining` form a cycle and are silently skipped.

        act = {nid: activation_funcs[genome.nodes[nid].activation] for nid in order}

        return cls(input_ids, output_ids, order, act, incoming)

    def activate(self, inputs: list[float]) -> list[float]:
        """Run a forward pass and return output-node values."""
        values: dict[int, float] = dict(zip(self._input_ids, inputs))
        for nid in self._order:
            total = sum(w * values.get(src, 0.0) for src, w in self._incoming[nid])
            values[nid] = self._act[nid](total)
        return [values.get(nid, 0.0) for nid in self._output_ids]


class Species:
    def __init__(self, species_id: int, representative: Genome) -> None:
        self.id:             int          = species_id
        self.representative: Genome       = representative
        self.members:        list[Genome] = []

    def adjusted_fitness(self) -> float:
        """Fitness sharing: total member fitness divided by species size."""
        n = len(self.members)
        return sum(g.fitness for g in self.members) / n if n else 0.0


@dataclass
class NeatConfig:
    """
    Configurable NEAT parameters.

    num_inputs, num_outputs, activation_options, and activation_funcs are
    problem-specific and always required. The remaining fields have defaults.
    All other NEAT behaviour is governed by the module-level _constants above.
    """

    # Network I/O (problem-specific; always required)
    num_inputs:  int
    num_outputs: int

    # Activation functions available for hidden and output nodes
    activation_options: list[str]
    activation_funcs:   dict[str, Callable[[float], float]]

    # Population size
    pop_size: int = 10

    # Mutation rates
    p_add_conn:      float = 0.13
    p_add_node:      float = 0.13
    p_weight_mutate: float = 0.70


# Population

class Population:
    """
    Manages a population of Genomes with NEAT speciation and reproduction.

    Each generation:
      1. fitness_fn(genomes) assigns genome.fitness for every genome.
      2. Speciation groups genomes by compatibility distance.
      3. Reproduction fills the next generation:
           * The _ELITISM fittest genomes are copied unchanged.
           * Remaining slots are allocated to species proportionally to their
             adjusted (fitness-shared) fitness, then filled with offspring
             produced by crossover + mutation or mutation only.
    """

    def __init__(self, config: NeatConfig) -> None:
        self.config  = config
        self.species: list[Species] = []
        self._next_species_id = 0

        _innovations.reset()   # fresh innovation history for each run

        self.genomes: list[Genome] = [
            Genome.create_minimal(
                config.num_inputs,
                config.num_outputs,
                config.activation_options,
            )
            for _ in range(config.pop_size)
        ]

    def run(self, fitness_fn: Callable[[list[Genome]], None], n: int) -> None:
        """
        Evolve for n generations.

        fitness_fn must assign genome.fitness on every genome in the list it
        receives before returning.
        """
        for _ in range(n):
            fitness_fn(self.genomes)
            self._speciate()
            self.genomes = self._reproduce()

    def _speciate(self) -> None:
        """
        Assign each genome to the first species whose representative is within
        _COMPAT_THRESHOLD. Unmatched genomes found a new species.
        """
        for s in self.species:
            s.members = []

        for genome in self.genomes:
            for s in self.species:
                if genome.distance(s.representative) < _COMPAT_THRESHOLD:
                    s.members.append(genome)
                    break
            else:
                self._next_species_id += 1
                new_s = Species(self._next_species_id, genome)
                new_s.members.append(genome)
                self.species.append(new_s)

        self.species = [s for s in self.species if s.members]

        for s in self.species:
            s.representative = random.choice(s.members)

    def _reproduce(self) -> list[Genome]:
        """Build the next generation from the current one."""
        cfg = self.config

        # Elitism: carry the fittest genomes forward unchanged.
        by_fitness = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)
        next_gen   = [g.copy() for g in by_fitness[:_ELITISM]]

        # Allocate remaining offspring slots to species proportionally.
        remaining = cfg.pop_size - len(next_gen)
        total_adj = sum(s.adjusted_fitness() for s in self.species) or 1.0
        ranked    = sorted(
            self.species, key=lambda s: s.adjusted_fitness(), reverse=True
        )

        allocated = 0
        slots: list[tuple[Species, int]] = []
        for i, s in enumerate(ranked):
            if i < len(ranked) - 1:
                n = round(remaining * s.adjusted_fitness() / total_adj)
            else:
                n = remaining - allocated   # rounding remainder → best species
            slots.append((s, max(n, 0)))
            allocated += max(n, 0)

        # Generate offspring for each species.
        for s, n_offspring in slots:
            members = sorted(s.members, key=lambda g: g.fitness, reverse=True)
            weights = [max(g.fitness, 1e-6) for g in members]

            for _ in range(n_offspring):
                if len(members) > 1 and random.random() < _CROSSOVER_RATE:
                    p1, p2 = random.choices(members, weights=weights, k=2)
                    child  = crossover(p1, p2)
                else:
                    child = random.choices(members, weights=weights)[0].copy()

                child.mutate(
                    cfg.p_weight_mutate,
                    cfg.p_add_node,
                    cfg.p_add_conn,
                    cfg.activation_options,
                )
                next_gen.append(child)

        return next_gen[:cfg.pop_size]
