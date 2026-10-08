"""Tests for rowsynth._jit."""
from __future__ import annotations

import ast
import contextlib
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth import _jit
from rowsynth.osc import _k_blep

PACKAGE = Path(_jit.__file__).resolve().parent

# Says whether the kernels are compiled, and what kind of thing one of them is.
STATE = """
import sys
from rowsynth import _jit
from rowsynth.osc import _k_blep
print(_jit.COMPILED, type(_k_blep).__name__)
"""

# Says whether Numba was brought in.
LOADED = """
print("numba" in sys.modules)
"""

# Stands in the way of Numba, as if it were not installed.
NO_NUMBA = """
import sys
sys.modules["numba"] = None
"""

# A note of every wave into one buffer, both low-passes among them, then the output stage. It says whether the
# kernels are compiled, and on a line of its own gives the samples in hex.
PLAY = """
import numpy as np
from rowsynth import _jit
from rowsynth.note import render_note
from rowsynth.sound import finish
out = np.zeros(900, dtype=np.float32)
for inst in ({"duty": [0, 3]}, {"wave": "saw", "lp": 900, "res": 0.5}, {"wave": "tri", "lp": 1200},
             {"wave": "noise"}, {"wave": "fm", "fb": 0.4, "ops": [{"mul": 2, "env": {"d": 0.02, "s": 0.5}}, {}]}):
    render_note(out, 20, inst, 57, 4, rate=8000)
print(_jit.COMPILED)
print(finish(out, rate=8000).tobytes().hex())
"""


def here(code: str) -> list[str]:
    """What the code prints when it runs in this process, line by line."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(code, {})
    return printed.getvalue().splitlines()


def child(code: str, switch: str | None = None) -> list[str]:
    """What the code prints when a fresh Python runs it, line by line. It has the switch only if one is given."""
    env = {key: value for key, value in os.environ.items() if key != "ROWSYNTH_NO_JIT"}
    if switch is not None:
        env["ROWSYNTH_NO_JIT"] = switch
    done = subprocess.run([sys.executable, "-W", "error", "-c", code], cwd=PACKAGE.parent, env=env,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise AssertionError(done.stderr)
    return done.stdout.splitlines()


class SwitchTest(unittest.TestCase):
    def test_compiling_stays_on_with_the_switch_unset_empty_or_0(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("ROWSYNTH_NO_JIT", None)
            self.assertFalse(_jit._switched_off())
            for value in ("", "0"):
                os.environ["ROWSYNTH_NO_JIT"] = value
                self.assertFalse(_jit._switched_off(), value)

    def test_any_other_value_of_the_switch_turns_compiling_off(self):
        for value in ("1", "yes"):
            with mock.patch.dict(os.environ, {"ROWSYNTH_NO_JIT": value}):
                self.assertTrue(_jit._switched_off(), value)

    def test_switched_off_a_kernel_is_the_plain_function_and_numba_is_left_out(self):
        self.assertEqual(child(STATE + LOADED, switch="1"), ["False function", "False"])

    def test_without_numba_a_kernel_is_the_plain_function(self):
        self.assertEqual(child(NO_NUMBA + STATE), ["False function"])


class KernelTest(unittest.TestCase):
    def test_every_kernel_in_the_package_carries_the_decorator(self):
        bare = []
        for path in sorted(PACKAGE.rglob("*.py")):
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("_k_"):
                    if [ast.unparse(decorator) for decorator in node.decorator_list] != ["kernel"]:
                        bare.append(f"{path.name}: {node.name}")
        self.assertEqual(bare, [])

    @unittest.skipUnless(_jit.COMPILED, "the kernels are not compiled in this run")
    def test_with_numba_a_kernel_is_compiled_and_kept_on_disk(self):
        self.assertEqual(_k_blep(0.05, 0.1), -0.25)
        self.assertEqual(_k_blep.py_func(0.05, 0.1), -0.25)
        # Every form it was compiled in runs without the interpreter.
        self.assertGreaterEqual(len(_k_blep.signatures), 1)
        self.assertEqual(len(_k_blep.nopython_signatures), len(_k_blep.signatures))
        self.assertTrue(os.path.isdir(_k_blep.stats.cache_path))

    @unittest.skipUnless(_jit.COMPILED, "the kernels are not compiled in this run")
    def test_notes_have_the_same_samples_compiled_and_plain(self):
        compiled, fast = here(PLAY)
        plain, slow = child(PLAY, switch="1")
        self.assertEqual((compiled, plain), ("True", "False"))
        self.assertEqual(fast, slow)
        samples = np.frombuffer(bytes.fromhex(fast), dtype=np.float32)
        self.assertEqual(samples.size, 900)
        self.assertGreater(float(np.abs(samples).max()), 0.5)


if __name__ == "__main__":
    unittest.main()
