"""The envelope of an FM operator: a straight rise to full level, then straight lines in decibels."""
from __future__ import annotations

import math

import numpy as np

from .errors import RowSynthError

# An operator's settings are one row of params; what moves while it plays is one row of state.
_PARAMS, _STATE = 8, 6
# The envelope in params: its rise a sample, its sustain level in dB, and its fall in dB a sample
# while decaying, while held and once let go.
_ATTACK, _SUSTAIN_DB, _DECAY, _SUSTAIN, _RELEASE = 3, 4, 5, 6, 7
# The envelope in state: its stage (0 rising, 1 decaying, 2 held, 3 let go, 4 done), its level on
# the way up, and its level in dB from there on.
_STAGE, _AMP, _DB = 3, 4, 5


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


def _k_envelope(out, params, state, gate):
    """Fills `out` with the first operator's envelope, let go at sample `gate`."""
    for i in range(out.shape[0]):
        if i == gate:
            _k_release(state, 0)
        out[i] = _k_env(params, state, 0)


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
