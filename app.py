#!/usr/bin/env python3
"""
Flask web front-end.

Intended to match the spirit of the Breedesizer UI described in §3.2 of the
paper.

Usage:
    python app.py [--output DIR] [--no-fm] [--generations N]

Then open  http://localhost:5001  in a browser.
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import wave
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

import cssn
import neat_impl
import numpy as np

# Flask app

app = Flask(__name__)

# Shared state

_lock       = threading.Lock()
_gen_ready  = threading.Event()   # evolution → web: generation rendered
_sel_given  = threading.Event()   # web → evolution: selection submitted

# Mutable shared state (always accessed under _lock, except _sel_given payload).
_state: dict = {
    "gen":         0,
    "status":      "starting",   # starting | evolving | waiting | done | error
    "individuals": [],           # list of {id, filename, waveform}
    "error":       "",
}
_selected_ids: list[str] = []    # set just before _sel_given is set

_current_genomes: list[neat_impl.Genome] = []

_active_settings = {
    "gain": cssn.FM_MOD_GAIN,
    "detune": 0.0,
    "offsets": 0,
    "n_periodic": 10,
    "variation": False,
    "adsr": True,
    "fm": True
}


# Evolution thread

def _run_evolution(output_dir: str, n_generations: int) -> None:
    """Background thread: runs NEAT and pauses each generation for user input."""
    try:
        config = cssn.make_neat_config()
        pop    = neat_impl.Population(config)

        gen_counter: list[int] = [0]

        def fitness_fn(genomes: list[neat_impl.Genome]) -> None:
            global _current_genomes
            gen_counter[0] += 1
            g = gen_counter[0]

            with _lock:
                _state["status"] = "evolving"
                _state["gen"]    = g
                _current_genomes = genomes

            # Get current UI settings to render this generation
            with _lock:
                gain = _active_settings["gain"]
                detune = _active_settings["detune"]
                offsets = _active_settings["offsets"]
                n_periodic = _active_settings["n_periodic"]
                variation = _active_settings["variation"]
                adsr = _active_settings["adsr"]
                fm = _active_settings["fm"]

            mod_amp = gain / cssn.NOTE_FREQ_HZ
            mod_freq_ratio = 1.0 + offsets * 0.25
            symmetric = not variation

            # Render all genomes
            individuals = []
            for idx, genome in enumerate(genomes, start=1):
                path = cssn.render_genome(
                    genome, g, idx, output_dir,
                    fm_enabled=fm,
                    n_periodic=n_periodic,
                    symmetric=symmetric,
                    mod_amp=mod_amp,
                    mod_freq_ratio=mod_freq_ratio,
                    detune=detune,
                    adsr_enabled=adsr
                )
                # Compute waveform thumbnails (carrier and modulator, 120 points each).
                net          = neat_impl.FeedForwardNetwork.create(genome, cssn.ACTIVATION_FUNCS)
                car, mod     = cssn.generate_waveform(net, n_periodic, symmetric=symmetric)
                wt_car       = cssn.fourier_wavetable(car)
                wt_mod       = cssn.fourier_wavetable(mod)
                step         = max(1, len(wt_car) // 120)
                waveform_car = [round(float(v), 4) for v in wt_car[::step]]
                waveform_mod = [round(float(v), 4) for v in wt_mod[::step]]

                individuals.append({
                    "id":           str(idx),
                    "filename":     os.path.basename(path),
                    "waveform_car": waveform_car,
                    "waveform_mod": waveform_mod,
                })

            # Publish to web layer and wait for selection
            with _lock:
                _state["individuals"] = individuals
                _state["status"]      = "waiting"

            _sel_given.clear()
            _gen_ready.set()
            _sel_given.wait()   # blocks until POST /api/select

            # Assign NEAT fitness
            # [NOT IN PAPER]: selected → 1.0, unselected → 0.01
            # (non-zero fitness prevents degenerate speciation)
            with _lock:
                selected = set(_selected_ids)
            for idx, genome in enumerate(genomes, start=1):
                genome.fitness = 1.0 if str(idx) in selected else 0.01

            with _lock:
                _state["status"] = "evolving"

        pop.run(fitness_fn, n=n_generations)

        with _lock:
            _state["status"] = "done"

    except Exception as exc:  # noqa: BLE001
        with _lock:
            _state["status"] = "error"
            _state["error"]  = str(exc)
        raise


# HTTP routes

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    with _lock:
        return jsonify(dict(_state))


@app.post("/api/select")
def api_select():
    global _selected_ids, _active_settings
    data = request.get_json(force=True)
    ids  = [str(i) for i in data.get("selected", [])]

    with _lock:
        if _state["status"] != "waiting":
            return jsonify({"ok": False, "error": "Not waiting for selection"}), 400
        _selected_ids = ids

        # Save active settings from the UI
        if "gain" in data:
            _active_settings["gain"] = float(data["gain"])
        if "detune" in data:
            _active_settings["detune"] = float(data["detune"])
        if "offsets" in data:
            _active_settings["offsets"] = int(data["offsets"])
        if "n_periodic" in data:
            _active_settings["n_periodic"] = int(data["n_periodic"])
        if "variation" in data:
            _active_settings["variation"] = bool(data["variation"])
        if "adsr" in data:
            _active_settings["adsr"] = bool(data["adsr"])
        if "fm" in data:
            _active_settings["fm"] = bool(data["fm"])

    _sel_given.set()
    return jsonify({"ok": True})


@app.get("/audio/<path:filename>")
def serve_audio(filename: str):
    """Serve a WAV file from the output directory."""
    audio_path = Path(app.config["OUTPUT_DIR"]) / filename
    if not audio_path.exists():
        return "Not found", 404
    return send_file(str(audio_path.resolve()), mimetype="audio/wav")


def write_wav_to_buffer(audio: np.ndarray, sample_rate: int = 44100) -> io.BytesIO:
    peak = np.max(np.abs(audio))
    normed = audio / peak if peak > 1e-9 else audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((normed * 32767).astype("<i2").tobytes())
    buf.seek(0)
    return buf


@app.get("/api/audio/<int:ind_idx>")
def api_audio(ind_idx: int):
    with _lock:
        if not _current_genomes or ind_idx < 1 or ind_idx > len(_current_genomes):
            return "Invalid index or no genomes loaded", 400
        genome = _current_genomes[ind_idx - 1]

    # Parse query parameters, falling back to current active settings
    with _lock:
        default_gain = _active_settings["gain"]
        default_detune = _active_settings["detune"]
        default_offsets = _active_settings["offsets"]
        default_n_periodic = _active_settings["n_periodic"]
        default_variation = _active_settings["variation"]
        default_adsr = _active_settings["adsr"]
        default_fm = _active_settings["fm"]

    gain = float(request.args.get("gain", default_gain))
    detune = float(request.args.get("detune", default_detune))
    offsets = int(request.args.get("offsets", default_offsets))
    n_periodic = int(request.args.get("n_periodic", default_n_periodic))
    variation = request.args.get("variation", str(default_variation).lower()) == "true"
    adsr = request.args.get("adsr", str(default_adsr).lower()) == "true"
    fm = request.args.get("fm", str(default_fm).lower()) == "true"

    net = neat_impl.FeedForwardNetwork.create(genome, cssn.ACTIVATION_FUNCS)
    car, mod = cssn.generate_waveform(net, n_periodic, symmetric=not variation)
    wt_car = cssn.fourier_wavetable(car)
    wt_mod = cssn.fourier_wavetable(mod)

    mod_amp = gain / cssn.NOTE_FREQ_HZ
    mod_freq_ratio = 1.0 + offsets * 0.25

    audio = cssn.synthesize_note(
        wt_car,
        wt_mod,
        fm_enabled=fm,
        mod_amp=mod_amp,
        mod_freq_ratio=mod_freq_ratio,
        detune=detune,
        adsr_enabled=adsr
    )

    buf = write_wav_to_buffer(audio)
    return send_file(buf, mimetype="audio/wav")


@app.get("/api/waveform/<int:ind_idx>")
def api_waveform(ind_idx: int):
    with _lock:
        if not _current_genomes or ind_idx < 1 or ind_idx > len(_current_genomes):
            return "Invalid index or no genomes loaded", 400
        genome = _current_genomes[ind_idx - 1]

    with _lock:
        default_n_periodic = _active_settings["n_periodic"]
        default_variation = _active_settings["variation"]

    n_periodic = int(request.args.get("n_periodic", default_n_periodic))
    variation = request.args.get("variation", str(default_variation).lower()) == "true"

    net = neat_impl.FeedForwardNetwork.create(genome, cssn.ACTIVATION_FUNCS)
    car, mod = cssn.generate_waveform(net, n_periodic, symmetric=not variation)
    wt_car = cssn.fourier_wavetable(car)
    wt_mod = cssn.fourier_wavetable(mod)

    step = max(1, len(wt_car) // 120)
    waveform_car = [round(float(v), 4) for v in wt_car[::step]]
    waveform_mod = [round(float(v), 4) for v in wt_mod[::step]]

    return jsonify({
        "waveform_car": waveform_car,
        "waveform_mod": waveform_mod
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSSN web IEC — Jónsson, Hoover, Risi (GECCO 2015)",
    )
    parser.add_argument("--generations", "-g", type=int, default=20)
    parser.add_argument("--output",      "-o", default="output_web")
    parser.add_argument("--port",        "-p", type=int, default=5001)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    app.config["OUTPUT_DIR"] = os.path.abspath(args.output)

    # Start the evolution in a daemon thread so Ctrl-C stops everything.
    t = threading.Thread(
        target=_run_evolution,
        args=(args.output, args.generations),
        daemon=True,
    )
    t.start()

    print(f"\n  CSSN Breedesizer  →  http://localhost:{args.port}/\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
