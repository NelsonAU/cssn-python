#!/usr/bin/env python3
"""
Compositional Sound Synthesis Networks (CSSN)

Reimplementation of:
    Jónsson, B. Þ., Hoover, A. K., & Risi, S. (2015).
    "Interactively Evolving Compositional Sound Synthesis Networks."
    GECCO '15, pp. 321–328. https://doi.org/10.1145/2739480.2754796

Each timbre is encoded by a variant of a CPPN (Compositional Pattern-Producing
Network) called a CSSN (Compositional Sound Synthesis Network). The CSSN is
evolved by Neuro-Evolution of Augmenting Topologies (NEAT). At query time it
maps a 1-D waveform coordinate to two audio outputs: a carrier wave and an FM
modulator.
"""

from __future__ import annotations

import math
import os
import wave

import neat
import numpy as np
from neat.activations import ActivationFunctionSet


# Activation functions
#
# §4.1: "The activation functions for each hidden neuron in the CSSNs are
#  chosen from the canonical set of Gaussian, Bipolar sigmoid, sine and linear,
#  each with a 0.25 probability of being added."
#
# Note: §3.1 somewhat inconsistently states that the implementation uses "sine,
# cosine, and inverse tangent activation functions", but we go with §4.1 here.

def gaussian_activation(x: float) -> float:
    """Gaussian: exp(−x²).  Promotes bilateral symmetry in patterns."""
    return math.exp(-x * x)


def bipolar_sigmoid_activation(x: float) -> float:
    """
    Bipolar sigmoid: 2 / (1 + exp(−5x)) − 1.
    Maps ℝ → (−1, 1).  Distinct from tanh and standard sigmoid.
    """
    return (2.0 / (1.0 + math.exp(-5.0 * x))) - 1.0


def sin_activation(x: float) -> float:
    """Sine: sin(x)."""
    return math.sin(x)


def linear_activation(x: float) -> float:
    """Linear / identity, hard-clamped to [−1, 1]."""
    return max(-1.0, min(1.0, x))


# Register our functions with neat-python's ActivationFunctionSet *before*
# neat.Config is loaded, because Config.__init__ validates activation_options
# against the registered set.  We patch the class __init__ so every new
# ActivationFunctionSet instance (created inside DefaultGenomeConfig.__init__)
# automatically includes our functions.
_orig_afs_init = ActivationFunctionSet.__init__


def _patched_afs_init(self: ActivationFunctionSet) -> None:
    _orig_afs_init(self)
    self.add("gaussian",        gaussian_activation)
    self.add("bipolar_sigmoid", bipolar_sigmoid_activation)
    self.add("sin",             sin_activation)    # shadows identical built-in
    self.add("linear",          linear_activation) # built-in 'identity' is unclamped


ActivationFunctionSet.__init__ = _patched_afs_init  # type: ignore[method-assign]


# Constants

# [PAPER] values:
WAVETABLE_SIZE   = 1024     # §3.1: "size is adjustable … with a default of 1024"
NOTE_FREQ_HZ     = 130.813  # §3.2: "one short note when clicked (C3 at 130.813 Hz)"
NOTE_DURATION_MS = 500.0    # §3.2: "one quarter note at 120 BPM, 500 ms"
POP_SIZE         = 10       # §4.1: "Population size is 10 per generation"
P_ADD_CONN       = 0.13     # §4.1: "probability of adding a new connection … 0.13"
P_ADD_NODE       = 0.13     # §4.1: "probability of adding a new node … 0.13"
P_WEIGHT_MUTATE  = 0.7      # §4.1: "probability of weight mutation … 0.7"

# §3.1: "an integer value n that determines the number of repeating
#  patterns in the waveform" — paper describes n as adjustable but gives no
#  fixed default.  The 2014 draft's experimental notes (commented out) suggest
#  n=10; the note "5 for gen 1, reduced to 1 thereafter" in the paper refers to
#  the Breedesizer UI's *mutation-count* slider (§3.2), NOT this parameter.
#  [NOT IN PAPER]: value 1 chosen as a conservative default.
N_PERIODIC       = 1

# [NOT IN PAPER] — reasonable standard defaults:
SAMPLE_RATE      = 44100    # standard CD-quality sample rate
ADSR_ATTACK_MS   = 20       # attack ramp (ms)
ADSR_DECAY_MS    = 50       # decay from peak to sustain (ms)
ADSR_SUSTAIN     = 0.8      # sustain amplitude fraction
ADSR_RELEASE_MS  = 80       # release ramp at note end (ms)
FM_MOD_AMP       = 0.3      # modulation index: peak freq deviation as fraction of carrier freq [NOT IN PAPER]


# Waveform generation

def generate_waveform(
    net: neat.nn.FeedForwardNetwork,
    n_periodic: int,
    size: int = WAVETABLE_SIZE,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Query the CSSN once per sample position to produce a single waveform cycle.

    Network inputs  [§3.1 / Figure 3 — "three inputs"]:
      [0]  abs(y)           – §3.1: "a coordinate y of the waveform ranging
                               from -1 to 1" (abs applied for symmetry)
      [1]  sin(n · |y|)     – §3.1: "a sine function of the absolute value of
                               y, allowing for an adjustable integer value n
                               that determines the number of repeating patterns"
    The paper's third input (bias 1.0) is omitted: neat-python adds a trainable
    per-node bias internally, making a constant input redundant.

    symmetric=False lifts the abs() from the periodic input, reproducing
    §3.1 Figure 4b (potentially discontinuous waveforms).

    Returns
    ───────
    carrier   : np.ndarray  shape (size,)
    modulator : np.ndarray  shape (size,)
    """
    y_seq     = np.linspace(-1.0, 1.0, size)
    carrier   = np.empty(size)
    modulator = np.empty(size)

    for i, y in enumerate(y_seq):
        abs_y    = abs(y)
        periodic = math.sin(n_periodic * (abs_y if symmetric else y))
        out = net.activate([abs_y, periodic])
        carrier[i]   = out[0]
        modulator[i] = out[1]

    return carrier, modulator


def fourier_wavetable(samples: np.ndarray) -> np.ndarray:
    """
    Band-limit raw CSSN output via an FFT → IFFT round-trip, then normalise.

    §3.1: "signal data is decomposed to its constituent frequencies by
    a Fourier transform.  The results of this process is a table of
    coefficients in a Fourier series … Using the Web Audio API, those Fourier
    coefficients are transformed into a periodic wave."

    NumPy rfft / irfft replicates the Web Audio PeriodicWave construction.
    Normalisation to [−1, 1] keeps all individuals at consistent volume.

    DIVERGENCE FROM PAPER: the rfft → irfft round-trip is mathematically
    lossless (identity up to floating-point rounding), so the only real effect
    here is the peak normalisation.  The Web Audio PeriodicWave, by contrast,
    bandlimits at playback time — it discards harmonics above the Nyquist
    frequency for the current playback pitch, preventing aliasing when the same
    wavetable is played at higher octaves.  This implementation does not
    anti-alias; audible aliasing is unlikely at the single note used (C3), but
    would appear at higher pitches.
    """
    spectrum  = np.fft.rfft(samples)
    wavetable = np.fft.irfft(spectrum, n=len(samples))
    peak = np.max(np.abs(wavetable))
    if peak > 1e-9:
        wavetable /= peak
    return wavetable


# Audio synthesis

def apply_adsr(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Apply an ADSR amplitude envelope.

    §3.2 & §4.1: "an ADSR envelope (Attack-Decay-Sustain-Release) can
    optionally be enabled … The ADSR envelope is enabled for the study
    presented here, resulting in timbres that start and end smoothly."
    Specific ADSR parameter values are [NOT IN PAPER].
    """
    n       = len(audio)
    env     = np.ones(n, dtype=float)
    a       = int(sample_rate * ADSR_ATTACK_MS  / 1000)
    d       = int(sample_rate * ADSR_DECAY_MS   / 1000)
    r       = int(sample_rate * ADSR_RELEASE_MS / 1000)
    s_start = min(a + d, n)
    s_end   = max(n - r, s_start)

    if a > 0:
        env[:min(a, n)] = np.linspace(0.0, 1.0, min(a, n))
    if a < s_start:
        env[a:s_start] = np.linspace(1.0, ADSR_SUSTAIN, s_start - a)
    env[s_start:s_end] = ADSR_SUSTAIN
    if s_end < n:
        env[s_end:] = np.linspace(ADSR_SUSTAIN, 0.0, n - s_end)

    return audio * env


def synthesize_note(
    carrier_wt:   np.ndarray,
    modulator_wt: np.ndarray,
    freq:         float = NOTE_FREQ_HZ,
    duration_ms:  float = NOTE_DURATION_MS,
    sample_rate:  int   = SAMPLE_RATE,
    fm_enabled:   bool  = True,
    mod_amp:      float = FM_MOD_AMP,
    mod_freq_ratio: float = 1.0,
) -> np.ndarray:
    """
    Render one note using wavetable lookup with optional FM modulation.

    §3.1: "one output modulates the oscillation frequency of the
    carrier oscillator produced by the second output."

    mod_freq_ratio — §3.1: "The third slider changes the modulator
    frequencies by multiples of the carrier frequency"; default = 1 (same
    frequency as carrier).

    §3.1 / Web Audio API: the modulator is connected to the carrier's frequency
    AudioParam, which is true FM — the modulator shifts instantaneous frequency,
    and phase is its time integral.  mod_amp is the modulation index (peak
    frequency deviation as a fraction of carrier frequency) — [NOT IN PAPER].
    """
    n       = int(sample_rate * duration_ms / 1000)
    wt_size = len(carrier_wt)
    t       = np.arange(n)

    if fm_enabled:
        mod_phase = (t * freq * mod_freq_ratio / sample_rate) % 1.0
        mod_idx   = (mod_phase * wt_size).astype(int) % wt_size
        instantaneous_freq = freq * (1.0 + modulator_wt[mod_idx] * mod_amp)
        car_phase = np.cumsum(instantaneous_freq / sample_rate) % 1.0
    else:
        car_phase = (t * freq / sample_rate) % 1.0

    car_idx = (car_phase * wt_size).astype(int) % wt_size

    return apply_adsr(carrier_wt[car_idx].copy(), sample_rate)


def write_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a normalised mono 16-bit PCM WAV file."""
    peak = np.max(np.abs(audio))
    normed = audio / peak if peak > 1e-9 else audio
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((normed * 32767).astype("<i2").tobytes())


# Per-genome pipeline

def render_genome(
    genome:     neat.DefaultGenome,
    config:     neat.Config,
    gen_idx:    int,
    ind_idx:    int,
    output_dir: str,
    fm_enabled: bool = True,
    n_periodic: int  = N_PERIODIC,
) -> str:
    """
    Build a CSSN from a genome, synthesise a note, write WAV. Returns path.

    DIVERGENCE FROM PAPER: §3.2 describes a mutation-count slider set to 5 for
    generation 0 and 1 thereafter, producing extra diversity in the seed
    population.  neat-python applies exactly one mutation pass per offspring
    inside DefaultReproduction.reproduce(), with no public hook to apply
    additional passes, so this schedule is not replicated here.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    car_raw, mod_raw = generate_waveform(net, n_periodic)
    carrier_wt   = fourier_wavetable(car_raw)
    modulator_wt = fourier_wavetable(mod_raw)
    audio        = synthesize_note(carrier_wt, modulator_wt, fm_enabled=fm_enabled)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"gen{gen_idx:03d}_ind{ind_idx:02d}.wav")
    write_wav(path, audio)
    return path
