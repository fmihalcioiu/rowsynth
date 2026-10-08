"""Tests for rowsynth.checks."""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

from rowsynth import checks
from rowsynth.checks import Finding, run, sounds
from rowsynth.errors import RowSynthError
from rowsynth.project import Project

RATE = 44100
# At 100 Hz the fade that ends the output stage is shorter than one sample, and the last sample is left as it is.
SLOW = 100

INSTRUMENTS = {
    "beep": {"vol": [15, 12, 9, 6]},
    "hiss": {"wave": "noise", "vol": [15, 10, 6, 3], "release": 1},
}


def sound(inst: dict | str, note: str | float, ticks: float) -> dict:
    """A def of one voice playing one note from the start."""
    return {"voices": [{"inst": inst, "notes": [[0, note, ticks]]}]}


# Keeps every rule. It is 5880 samples long at 44100 Hz.
BEEP = sound("beep", "a4", 4)
# Eight times the level: over the knee at the bank's gain.
LOUD = sound({"gain": 8}, "a4", 4)
# As loud, and a narrow pulse. The knee bends its tall side and not its shallow one, which leaves an offset.
LOPSIDED = sound({"duty": 0, "gain": 8}, "a3", 4)
# A triangle starts at its top, and one this low stays near it. The output stage takes most of that away, and
# what is left is not much larger than the first sample.
RUMBLE = sound({"wave": "tri"}, 0, 3)
# A pulse far below hearing is a level that does not move. Held for a hundredth of a tick and cut dead, it
# leaves a buffer too short for the output stage to settle in.
THUD = sound({"release": 0}, 0, 0.01)
# Noise clocked far above 4 kHz.
HISS = sound("hiss", "a6", 4)
QUIET = {"voices": []}

# A set of rules made for a test. Its one finding says what the module was called and what it was handed.
EXTRA = '''"""A set of rules made for a test."""
from rowsynth.checks import Finding


def check(project, rate):
    return [Finding(__name__, str(rate), ", ".join(project.sounds))]
'''


def project_of(**entries: dict) -> Project:
    """A project with the instruments above and these sounds."""
    return Project.from_dict({"instruments": INSTRUMENTS, "sounds": entries})


def found_in(entry: dict, rate: int = RATE) -> list[Finding]:
    """The findings on one sound, named zap."""
    return sounds.check(project_of(zap=entry), rate)


def zap(rule: str, message: str) -> Finding:
    """A finding on the sound named zap."""
    return Finding(rule, "sounds.zap", message)


def rules(found: list[Finding]) -> list[str]:
    """The rule of each finding."""
    return [finding.rule for finding in found]


class FindingTest(unittest.TestCase):
    def test_to_dict_gives_the_rule_the_target_and_the_message_as_plain_data(self):
        data = Finding("dc", "sounds.zap[2]", "off centre").to_dict()
        self.assertEqual(data, {"rule": "dc", "target": "sounds.zap[2]", "message": "off centre"})
        self.assertEqual(list(data), ["rule", "target", "message"])
        self.assertEqual(json.loads(json.dumps(data)), data)

    def test_a_finding_cannot_be_changed_and_equals_one_with_the_same_fields(self):
        finding = Finding("peak", "sounds.zap", "too loud")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            finding.rule = "dc"
        self.assertEqual(finding, Finding("peak", "sounds.zap", "too loud"))
        self.assertNotEqual(finding, Finding("peak", "sounds.zap[0]", "too loud"))


class RunTest(unittest.TestCase):
    def with_modules(self, *names: str) -> None:
        """Puts a module of EXTRA under each name into the package, until the test is over."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for name in names:
            Path(folder.name, f"{name}.py").write_text(EXTRA, encoding="utf-8", newline="\n")
            self.addCleanup(sys.modules.pop, f"rowsynth.checks.{name}", None)
            self.addCleanup(checks.__dict__.pop, name, None)
        checks.__path__.append(folder.name)
        self.addCleanup(sys.path_importer_cache.pop, folder.name, None)
        self.addCleanup(checks.__path__.remove, folder.name)

    def test_run_gives_what_every_module_gives_in_the_order_of_their_names(self):
        project = project_of(zap={"def": LOUD}, pop={"def": BEEP})
        own = sounds.check(project, RATE)
        self.assertEqual(rules(own), ["peak"])
        self.with_modules("zebra", "apple")
        self.assertEqual(run(project), [Finding("rowsynth.checks.apple", "44100", "zap, pop"), *own,
                                        Finding("rowsynth.checks.zebra", "44100", "zap, pop")])

    def test_run_hands_its_rate_to_the_modules(self):
        self.with_modules("apple")
        self.assertEqual(run(project_of(), 8000), [Finding("rowsynth.checks.apple", "8000", "")])

    def test_a_project_that_keeps_the_rules_gives_no_findings(self):
        kept = project_of(pop={"def": BEEP}, pips={"defs": [BEEP, BEEP]}, rest={"def": QUIET})
        for project in (kept, project_of(), Project()):
            self.assertEqual(run(project), [])


class SoundsTest(unittest.TestCase):
    def test_peak_fires_on_a_sound_that_reaches_the_knee_at_the_gain_of_the_bank(self):
        self.assertEqual(found_in({"def": LOUD}), [
            zap("peak", "reaches 0.944 at the bank's gain of 0.12, and from 0.7 up the output stage bends it")])

    def test_ends_fires_on_a_first_sample_of_a_hundredth_or_more(self):
        self.assertEqual(found_in({"def": RUMBLE}), [
            zap("ends", "starts with a jump of 0.015 out of silence, and 0.01 or more clicks")])

    def test_ends_fires_twice_when_both_ends_are_off_silence(self):
        self.assertEqual(found_in({"def": RUMBLE}, SLOW), [
            zap("ends", "starts with a jump of 0.617 out of silence, and 0.01 or more clicks"),
            zap("ends", "ends with a jump of 0.016 into silence, and 0.01 or more clicks")])

    def test_dc_fires_on_a_mean_sample_off_zero(self):
        self.assertEqual(found_in({"def": THUD}), [
            zap("dc", "its mean sample is 0.0077, which is 0.002 or more off zero")])

    def test_bright_fires_on_a_sound_with_most_of_its_energy_above_4_khz(self):
        self.assertEqual(found_in({"def": HISS}), [
            zap("bright", "0.81 of its energy is above 4 kHz, which is more than 0.3")])

    def test_length_fires_on_a_sound_longer_than_its_max_seconds(self):
        self.assertEqual(found_in({"def": BEEP, "max_seconds": 0.13}), [
            zap("length", "lasts 0.13333 seconds, which is over its limit of 0.13")])

    def test_a_sound_with_no_max_seconds_can_be_any_length(self):
        # Nearly three seconds, at a rate that leaves less of it to render.
        self.assertEqual(found_in({"def": sound("beep", "a4", 170)}, 8000), [])

    def test_max_seconds_moves_the_limit_and_a_sound_of_just_that_length_keeps_it(self):
        self.assertEqual(rules(found_in({"def": BEEP, "max_seconds": 5879 / RATE})), ["length"])
        self.assertEqual(found_in({"def": BEEP, "max_seconds": 5880 / RATE}), [])
        self.assertEqual(found_in({"def": BEEP, "max_seconds": 2}), [])

    def test_max_dc_moves_the_limit_and_a_mean_at_the_limit_breaks_it(self):
        self.assertEqual(found_in({"def": THUD, "max_dc": 0.01}), [])
        self.assertEqual(rules(found_in({"def": THUD, "max_dc": 0.0077})), ["dc"])
        self.assertEqual(rules(found_in({"def": LOPSIDED, "max_dc": 0.05})), ["peak"])

    def test_max_bright_moves_the_limit_and_a_share_at_the_limit_keeps_it(self):
        self.assertEqual(found_in({"def": HISS, "max_bright": 0.81}), [])
        self.assertEqual(rules(found_in({"def": HISS, "max_bright": 0.8})), ["bright"])
        self.assertEqual([finding.message for finding in found_in({"def": BEEP, "max_bright": 0.01})],
                         ["0.02 of its energy is above 4 kHz, which is more than 0.01"])

    def test_the_target_of_a_variant_carries_its_index(self):
        self.assertEqual([(finding.rule, finding.target) for finding in found_in({"defs": [BEEP, LOUD, HISS]})],
                         [("peak", "sounds.zap[1]"), ("bright", "sounds.zap[2]")])
        self.assertEqual([finding.target for finding in found_in({"defs": [LOUD]})], ["sounds.zap[0]"])

    def test_the_limits_of_an_entry_hold_every_variant_of_it(self):
        found = found_in({"defs": [BEEP, HISS], "max_bright": 0.9, "max_seconds": 0.12})
        self.assertEqual([(finding.rule, finding.target) for finding in found],
                         [("length", "sounds.zap[0]")])

    def test_findings_come_by_sound_then_by_variant_then_by_rule(self):
        project = project_of(zap={"defs": [LOPSIDED, HISS], "max_seconds": 0.05},
                             alarm={"def": RUMBLE, "max_seconds": 0.05})
        self.assertEqual([(finding.target, finding.rule) for finding in sounds.check(project, RATE)], [
            ("sounds.zap[0]", "peak"), ("sounds.zap[0]", "dc"), ("sounds.zap[0]", "length"),
            ("sounds.zap[1]", "bright"), ("sounds.zap[1]", "length"),
            ("sounds.alarm", "ends"), ("sounds.alarm", "length")])

    def test_an_entry_with_parts_or_held_gives_nothing_yet(self):
        held = {"note": LOUD, "start": 2, "loop": 4, "join": 1, "end": LOUD, "let_go": 2}
        project = project_of(pile={"parts": [[0, {"def": LOUD}], [0.1, {"def": HISS}]], "max_seconds": 0.01},
                             drone={"held": held, "max_seconds": 0.01})
        self.assertEqual(sounds.check(project, RATE), [])

    def test_a_def_that_cannot_be_rendered_is_refused_with_its_target_in_front(self):
        # Loading a project refuses a wave it does not know, so this one is put together by hand.
        project = Project()
        project.instruments = INSTRUMENTS
        project.sounds = {"zap": {"defs": [BEEP, sound({"wave": "sine"}, "a4", 4)]}}
        with self.assertRaises(RowSynthError) as caught:
            sounds.check(project, RATE)
        self.assertEqual(str(caught.exception), "sounds.zap[1]: voices[0].inst.wave: unknown wave 'sine'")


if __name__ == "__main__":
    unittest.main()
