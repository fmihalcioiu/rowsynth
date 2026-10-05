"""FM synthesis with four operators: sines that bend one another's phase, each under an envelope in decibels."""
from __future__ import annotations

import math

import numpy as np

from .errors import RowSynthError

_TAU = 2.0 * math.pi

# The eight algorithms, with the operators numbered from 0:
#   0  a chain: 0 bends 1, 1 bends 2, 2 bends 3, and only 3 is heard
#   1  0 and 1 together bend 2, 2 bends 3, and only 3 is heard
#   2  1 bends 2, 0 and 2 together bend 3, and only 3 is heard
#   3  0 bends 1, 1 and 2 together bend 3, and only 3 is heard
#   4  two pairs: 0 bends 1 and 2 bends 3; 1 and 3 are heard
#   5  0 bends each of 1, 2 and 3, and those three are heard
#   6  0 bends 1; 1, 2 and 3 are heard
#   7  nothing bends anything: four sines side by side
# INTO[alg, i, j] is 1 when operator j bends operator i. CARRIER[alg, i] is 1 when operator i is heard.
INTO = np.array([
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]],
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]],
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
    [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
], dtype=np.int64)
CARRIER = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
], dtype=np.int64)

# An operator's settings are one row of params; what moves while it plays is one row of state.
_PARAMS, _STATE = 8, 6
# The oscillator in params: its multiple of the note's frequency, its detune in Hz, and its level,
# which is an amplitude when the operator is heard and radians of bend when it is not.
_MUL, _DET, _LEVEL = 0, 1, 2
# The oscillator in state: its phase, its last sample, and that sample at its level, for the
# operators it bends.
_PHASE, _LAST, _OUT = 0, 1, 2
# The envelope in params: its rise a sample, its sustain level in dB, and its fall in dB a sample
# while decaying, while held and once let go.
_ATTACK, _SUSTAIN_DB, _DECAY, _SUSTAIN, _RELEASE = 3, 4, 5, 6, 7
# The envelope in state: its stage (0 rising, 1 decaying, 2 held, 3 let go, 4 done), its level on
# the way up, and its level in dB from there on.
_STAGE, _AMP, _DB = 3, 4, 5

# An operator the instrument leaves out: silent, its envelope down within a millisecond.
_SILENT = {"level": 0, "env": {"s": 0, "d": 0.001}}


def _number(table: dict, key: str, fallback: float, where: str) -> float:
    """table[key] as a float, or the fallback when the key is absent."""
    value = table.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        name = f"{where}.{key}" if where else key
        raise RowSynthError(f"{name}: not a number: {value!r}")
    return float(value)


def _set_envelope(row: np.ndarray, env: dict, rate: int, where: str) -> None:
    """Turns an envelope's times and levels into steps a sample, in one row of params."""
    if not isinstance(env, dict):
        raise RowSynthError(f"{where}: not an envelope: {env!r}")
    a = _number(env, "a", 0.0, where)
    d = _number(env, "d", 0.3, where)
    s = _number(env, "s", 1.0, where)
    sd = _number(env, "sd", 0.0, where)
    r = _number(env, "r", 0.2, where)
    sustain_db = 20.0 * math.log10(max(s, 1e-4))
    row[_ATTACK] = 1.0 / (a * rate) if a > 0 else 1.0
    row[_SUSTAIN_DB] = sustain_db
    # No decay time is an endless step: the first sample of the decay lands on the sustain level.
    row[_DECAY] = -sustain_db / (d * rate) if d > 0 else math.inf
    row[_SUSTAIN] = sd / rate
    row[_RELEASE] = 60.0 / (max(r, 0.001) * rate)


def _algorithm(inst: dict) -> int:
    """The instrument's algorithm, 4 when it names none."""
    alg = _number(inst, "alg", 4.0, "")
    if not 0 <= alg <= 7 or alg != int(alg):
        raise RowSynthError(f"alg: not an algorithm, they run from 0 to 7: {inst['alg']!r}")
    return int(alg)


def fm_voice(inst: dict, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """The four operators of an FM instrument at a sample rate: their params, and a state to start from.

    The instrument is {"alg": 0 to 7, "fb": radians, "ops": up to four of {"mul", "det", "level", "env"}}.
    Operators it leaves out are silent. The algorithm and the feedback are only checked here.
    """
    _algorithm(inst)
    _number(inst, "fb", 0.0, "")
    ops = inst.get("ops", [])
    if not isinstance(ops, (list, tuple)):
        raise RowSynthError(f"ops: not a list: {ops!r}")
    if len(ops) > 4:
        raise RowSynthError(f"ops: {len(ops)} operators, the voice has 4")
    params = np.zeros((4, _PARAMS))
    state = np.zeros((4, _STATE))
    for i in range(4):
        op = ops[i] if i < len(ops) else _SILENT
        where = f"ops[{i}]"
        if not isinstance(op, dict):
            raise RowSynthError(f"{where}: not an operator: {op!r}")
        params[i, _MUL] = _number(op, "mul", 1.0, where)
        params[i, _DET] = _number(op, "det", 0.0, where)
        params[i, _LEVEL] = _number(op, "level", 1.0, where)
        _set_envelope(params[i], op.get("env", {}), rate, f"{where}.env")
    return params, state


def _k_env(params, state, op):
    """The next sample of operator `op`'s envelope, a level from 0 to 1."""
    stage = state[op, _STAGE]
    if stage == 0.0:
        amp = state[op, _AMP] + params[op, _ATTACK]
        if amp >= 1.0:
            amp = 1.0
            state[op, _STAGE] = 1.0
        state[op, _AMP] = amp
        return amp
    db = state[op, _DB]
    if stage == 1.0:
        db -= params[op, _DECAY]
        if db <= params[op, _SUSTAIN_DB]:
            db = params[op, _SUSTAIN_DB]
            state[op, _STAGE] = 2.0
    elif stage == 2.0:
        db -= params[op, _SUSTAIN]
    elif stage == 3.0:
        db -= params[op, _RELEASE]
    else:
        return 0.0
    state[op, _DB] = db
    if db <= -96.0:
        state[op, _STAGE] = 4.0
        return 0.0
    return 10.0 ** (db / 20.0)


def _k_release(state, op):
    """Lets operator `op`'s envelope go. Let go on the way up, it falls from the level it had reached."""
    if state[op, _STAGE] == 0.0:
        state[op, _DB] = 20.0 * math.log10(max(state[op, _AMP], 1e-4))
    if state[op, _STAGE] < 3.0:
        state[op, _STAGE] = 3.0


def _k_gate_off(params, state):
    """Lets every operator's envelope go: the note has ended."""
    for op in range(params.shape[0]):
        _k_release(state, op)


def _k_fm(into, carrier, fb, params, state, freq, rate):
    """The next sample of the voice at `freq` Hz. `into` and `carrier` are one algorithm's rows."""
    mix = 0.0
    for i in range(4):
        # Operators run in order, so one that bends another has already made this sample.
        mod = 0.0
        for j in range(4):
            if into[i, j] != 0:
                mod += state[j, _OUT]
        if i == 0 and fb != 0.0:
            mod += fb * state[i, _LAST]
        s = math.sin(state[i, _PHASE] + mod) * _k_env(params, state, i)
        phase = state[i, _PHASE] + (_TAU * (freq * params[i, _MUL] + params[i, _DET])) / rate
        if phase > _TAU:
            phase -= _TAU
        state[i, _PHASE] = phase
        state[i, _LAST] = s
        if carrier[i] != 0:
            mix += s * params[i, _LEVEL]
        else:
            state[i, _OUT] = s * params[i, _LEVEL]
    return mix


def _k_envelope(out, params, state, gate):
    """Fills `out` with the first operator's envelope, let go at sample `gate`."""
    for i in range(out.shape[0]):
        if i == gate:
            _k_release(state, 0)
        out[i] = _k_env(params, state, 0)


def _k_tone(out, into, carrier, fb, params, state, freq, rate, gate):
    """Fills `out` with the voice at `freq` Hz, let go at sample `gate`."""
    for i in range(out.shape[0]):
        if i == gate:
            _k_gate_off(params, state)
        out[i] = _k_fm(into, carrier, fb, params, state, freq, rate)


def envelope(env: dict, held: float, seconds: float, rate: int = 44100) -> np.ndarray:
    """An envelope on its own: `seconds` of it, let go after `held` seconds.

    `env` is {"a", "d", "s", "sd", "r"}: seconds of attack, seconds of decay down to the sustain
    level `s` (0 to 1), decibels a second lost while held, and seconds of release to 60 dB down.
    """
    params = np.zeros((1, _PARAMS))
    state = np.zeros((1, _STATE))
    _set_envelope(params[0], env, rate, "env")
    out = np.zeros(math.floor(seconds * rate + 0.5), dtype=np.float32)
    _k_envelope(out, params, state, math.floor(held * rate + 0.5))
    return out


def fm_tone(inst: dict, freq: float, held: float, seconds: float, rate: int = 44100) -> np.ndarray:
    """An FM instrument's voice on its own at `freq` Hz: `seconds` of it, let go after `held` seconds."""
    params, state = fm_voice(inst, rate)
    alg = _algorithm(inst)
    fb = _number(inst, "fb", 0.0, "")
    out = np.zeros(math.floor(seconds * rate + 0.5), dtype=np.float32)
    _k_tone(out, INTO[alg], CARRIER[alg], fb, params, state, float(freq), float(rate), math.floor(held * rate + 0.5))
    return out
