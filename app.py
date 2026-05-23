#!/usr/bin/env python3
"""
Flask web front-end.

Intended to match the spirit of the Breedesizer UI described in §3.2 of the
paper.

Usage:
    python app.py [--output DIR] [--config FILE] [--no-fm] [--generations N]

Then open  http://localhost:5001  in a browser.
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

import neat
from flask import Flask, jsonify, render_template, request, send_file

# Import the CSSN synthesis pipeline from the sibling module.
# The activation-function monkey-patch in cssn.py runs at import time,
# before any neat.Config is created — that ordering is required.
import cssn

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


# Evolution thread

def _run_evolution(config_path: str, output_dir: str,
                   fm_enabled: bool, n_generations: int) -> None:
    """Background thread: runs NEAT and pauses each generation for user input."""
    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        pop = neat.Population(config)

        gen_counter: list[int] = [0]

        def fitness_fn(genomes: list, cfg: neat.Config) -> None:
            gen_counter[0] += 1
            g = gen_counter[0]

            with _lock:
                _state["status"] = "evolving"
                _state["gen"]    = g

            # Render all genomes
            individuals = []
            for idx, (_, genome) in enumerate(genomes, start=1):
                path = cssn.render_genome(
                    genome, cfg, g, idx, output_dir, fm_enabled
                )
                # Compute waveform thumbnails (carrier and modulator, 120 points each).
                net          = neat.nn.FeedForwardNetwork.create(genome, cfg)
                car, mod     = cssn.generate_waveform(net, cssn.N_PERIODIC)
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
            # (same as the console version; non-zero fitness prevents
            #  degenerate speciation in neat-python)
            with _lock:
                selected = set(_selected_ids)
            for idx, (_, genome) in enumerate(genomes, start=1):
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
    global _selected_ids
    data = request.get_json(force=True)
    ids  = [str(i) for i in data.get("selected", [])]

    with _lock:
        if _state["status"] != "waiting":
            return jsonify({"ok": False, "error": "Not waiting for selection"}), 400
        _selected_ids = ids
        # status will flip to 'evolving' inside the evolution thread

    _sel_given.set()
    return jsonify({"ok": True})


@app.get("/audio/<path:filename>")
def serve_audio(filename: str):
    """Serve a WAV file from the output directory."""
    audio_path = Path(app.config["OUTPUT_DIR"]) / filename
    if not audio_path.exists():
        return "Not found", 404
    return send_file(str(audio_path.resolve()), mimetype="audio/wav")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSSN web IEC — Jónsson, Hoover, Risi (GECCO 2015)",
    )
    parser.add_argument("--generations", "-g", type=int, default=20)
    parser.add_argument("--output",      "-o", default="output_web")
    parser.add_argument("--config",      "-c", default="neat_config.txt")
    parser.add_argument("--no-fm",       action="store_true")
    parser.add_argument("--port",        "-p", type=int, default=5001)
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.config
    )
    os.makedirs(args.output, exist_ok=True)

    app.config["OUTPUT_DIR"] = os.path.abspath(args.output)

    # Start the evolution in a daemon thread so Ctrl-C stops everything.
    t = threading.Thread(
        target=_run_evolution,
        args=(config_path, args.output, not args.no_fm, args.generations),
        daemon=True,
    )
    t.start()

    print(f"\n  CSSN Breedesizer  →  http://localhost:{args.port}/\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
