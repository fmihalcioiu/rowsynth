"""Tests for rowsynth.checks."""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

from rowsynth import checks
from rowsynth.bank import held_parts, normalise, render_entry
from rowsynth.checks import Finding, run, sounds
from rowsynth.errors import RowSynthError
from rowsynth.measure import measure
from rowsynth.project import Project
from rowsynth.sound import KNEE

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


# Keeps every rule. It is 5880 samples long at 44100 Hz, and a hundredth of a second in it reads as 440.1 Hz.
BEEP = sound("beep", "a4", 4)
# The same a semitone up, a semitone down, and after a tenth of a second of silence.
SHARP = sound("beep", "a#4", 4)
FLAT = sound("beep", "g#4", 4)
LATE = {"voices": [{"inst": "beep", "notes": [[6, "a4", 4]]}]}
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
# Notes long enough to cut a start and a loop from. From its fourth tick on the beep stays at one level, and
# six ticks of a4 are 44 periods to the sample.
TONE = sound("beep", "a4", 12)
DEEP = sound("beep", "c2", 12)
DEEPER = sound("beep", "c1", 12)
GROWL = sound({"wave": "tri"}, 0, 12)

# A set of rules made for a test. Its one finding says what the module was called and what it was handed.
EXTRA = '''"""A set of rules made for a test."""
from rowsynth.checks import Finding


def check(project, rate):
    return [Finding(__name__, str(rate), ", ".join(project.sounds))]
'''


def held(note: dict, start: float, loop: float, join: float, end: dict = BEEP, **rest: object) -> dict:
    """A held entry: a start and a loop of so many ticks cut from a note, the ticks of fade at the join, an end."""
    return {"held": {"note": note, "start": start, "loop": loop, "join": join, "end": end, "let_go": 2}, **rest}


def project_of(**entries: dict) -> Project:
    """A project with the instruments above and these sounds."""
    return Project.from_dict({"instruments": INSTRUMENTS, "sounds": entries})


def found_in(entry: dict, rate: int = RATE) -> list[Finding]:
    """The findings on one sound, named zap."""
    return sounds.check(project_of(zap=entry), rate)


def zap(rule: str, message: str, part: int | str | None = None) -> Finding:
    """A finding on the sound named zap, on one of its parts by number, or on a part of a held sound by name."""
    if part is None:
        return Finding(rule, "sounds.zap", message)
    return Finding(rule, f"sounds.zap.{part}" if isinstance(part, str) else f"sounds.zap.parts[{part}]", message)


def rules(found: list[Finding]) -> list[str]:
    """The rule of each finding."""
    return [finding.rule for finding in found]


def places(found: list[Finding]) -> list[tuple[str, str]]:
    """The target and the rule of each finding."""
    return [(finding.target, finding.rule) for finding in found]


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
        kept = project_of(pop={"def": BEEP}, pips={"defs": [BEEP, BEEP]}, rest={"def": QUIET},
                          hum=held(TONE, 4, 6, 1))
        for project in (kept, project_of(), Project()):
            self.assertEqual(run(project), [])


class SoundsTest(unittest.TestCase):
    def refusal_of(self, entry: dict) -> str:
        """What the check refuses a sound named zap with. The sound is put in by hand, past what loading refuses."""
        project = Project()
        project.instruments = INSTRUMENTS
        project.sounds = {"zap": entry}
        with self.assertRaises(RowSynthError) as caught:
            sounds.check(project, RATE)
        return str(caught.exception)

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

    def test_a_note_read_back_in_tune_gives_nothing(self):
        # By name, by number, and written 30 cents above what is read.
        marks = [{"at": 0.01, "note": "a4"}, {"at": 0.01, "note": 69}, {"at": 0.01, "note": 69.3}]
        self.assertEqual(found_in({"def": BEEP, "check": marks}), [])

    def test_pitch_fires_on_a_note_more_than_35_cents_off_and_gives_the_cents(self):
        self.assertEqual(found_in({"def": BEEP, "check": [{"at": 0.01, "note": "a#4"}]}), [
            zap("pitch", "is 100 cents below the note a#4 written at 0.01 seconds, and more than 35 is out of tune")])
        self.assertEqual(found_in({"def": BEEP, "check": [{"at": 0.01, "note": 68.6}]}), [
            zap("pitch", "is 40 cents above the note 68.6 written at 0.01 seconds, and more than 35 is out of tune")])

    def test_pitch_fires_where_the_sound_is_silent_or_over(self):
        # The note of LATE is in tune a tenth of a second in. Before it there is silence, and at 0.5 the end is past.
        marks = [{"at": 0, "note": "a4"}, {"at": 0.11, "note": "a4"}, {"at": 0.5, "note": "a4"}]
        self.assertEqual(found_in({"def": LATE, "check": marks}), [
            zap("pitch", "has no note to be read at 0 seconds, where the note a4 is written"),
            zap("pitch", "has no note to be read at 0.5 seconds, where the note a4 is written")])

    def test_pitch_comes_after_the_other_rules_and_note_by_note(self):
        marks = [{"at": 0.5, "note": "a4"}, {"at": 0.01, "note": "c5"}]
        self.assertEqual(found_in({"def": BEEP, "max_seconds": 0.1, "check": marks}), [
            zap("length", "lasts 0.13333 seconds, which is over its limit of 0.1"),
            zap("pitch", "has no note to be read at 0.5 seconds, where the note a4 is written"),
            zap("pitch", "is 300 cents below the note c5 written at 0.01 seconds, and more than 35 is out of tune")])

    def test_the_target_of_a_variant_carries_its_index(self):
        self.assertEqual([(finding.rule, finding.target) for finding in found_in({"defs": [BEEP, LOUD, HISS]})],
                         [("peak", "sounds.zap[1]"), ("bright", "sounds.zap[2]")])
        self.assertEqual([finding.target for finding in found_in({"defs": [LOUD]})], ["sounds.zap[0]"])

    def test_the_limits_of_an_entry_hold_every_variant_of_it(self):
        found = found_in({"defs": [BEEP, HISS], "max_bright": 0.9, "max_seconds": 0.12})
        self.assertEqual([(finding.rule, finding.target) for finding in found],
                         [("length", "sounds.zap[0]")])

    def test_the_check_of_an_entry_is_read_on_every_variant(self):
        found = found_in({"defs": [FLAT, BEEP, SHARP], "check": [{"at": 0.01, "note": "a4"}]})
        self.assertEqual(found, [
            Finding("pitch", "sounds.zap[0]",
                    "is 100 cents below the note a4 written at 0.01 seconds, and more than 35 is out of tune"),
            Finding("pitch", "sounds.zap[2]",
                    "is 100 cents above the note a4 written at 0.01 seconds, and more than 35 is out of tune")])

    def test_findings_come_by_sound_then_by_variant_then_by_rule(self):
        project = project_of(zap={"defs": [LOPSIDED, HISS], "max_seconds": 0.05},
                             alarm={"def": RUMBLE, "max_seconds": 0.05})
        self.assertEqual(places(sounds.check(project, RATE)), [
            ("sounds.zap[0]", "peak"), ("sounds.zap[0]", "dc"), ("sounds.zap[0]", "length"),
            ("sounds.zap[1]", "bright"), ("sounds.zap[1]", "length"),
            ("sounds.alarm", "ends"), ("sounds.alarm", "length")])

    def test_parts_that_keep_the_rules_give_nothing(self):
        self.assertEqual(found_in({"parts": [[0, {"def": BEEP}], [0.1, {"def": SHARP, "db": -6}]]}), [])

    def test_a_part_that_breaks_a_rule_is_found_under_its_place_in_the_parts(self):
        self.assertEqual(found_in({"parts": [[0, {"def": BEEP}], [0.05, {"def": LOUD}]]}), [
            zap("peak", "reaches 0.944 at the bank's gain of 0.12, and from 0.7 up the output stage bends it",
                part=1)])

    def test_the_whole_of_parts_that_breaks_bright_or_length_is_found_under_the_sound(self):
        # The hiss is let through as a part, and is nearly all there is to hear of the two together.
        harsh = {"parts": [[0, {"def": BEEP, "db": -20}], [0.02, {"def": HISS, "max_bright": 0.9}]]}
        self.assertEqual(found_in(harsh), [zap("bright", "0.78 of its energy is above 4 kHz, which is more than 0.3")])
        # Each part is longer than the limit of the entry, and neither is held to it.
        long = {"parts": [[0, {"def": BEEP}], [0.1, {"def": BEEP}]], "max_seconds": 0.1}
        self.assertEqual(found_in(long), [zap("length", "lasts 0.23333 seconds, which is over its limit of 0.1")])

    def test_the_whole_of_parts_is_not_held_to_peak(self):
        # One sound twice from the start adds up to twice the bank's peak, far over the knee.
        entry = {"parts": [[0, {"def": BEEP}], [0, {"def": BEEP}]]}
        whole = render_entry(project_of(zap=entry).sound("zap"), RATE)[0]
        self.assertGreater(float(abs(whole).max()), 2 * KNEE)
        self.assertEqual(found_in(entry), [])

    def test_each_part_is_held_to_its_own_limits_and_the_whole_to_those_of_the_entry(self):
        self.assertEqual(places(found_in({"parts": [[0, {"def": THUD}], [0, {"def": HISS}]]})), [
            ("sounds.zap.parts[0]", "dc"), ("sounds.zap.parts[1]", "bright"), ("sounds.zap", "bright")])
        # The offset of the thud breaks the limit of the entry, which the thud as a part is not held to.
        parts = [[0, {"def": THUD, "max_dc": 0.01}], [0, {"def": HISS, "max_bright": 0.9}]]
        self.assertEqual(found_in({"parts": parts, "max_bright": 0.9, "max_dc": 0.001}), [
            zap("dc", "its mean sample is 0.0012, which is 0.001 or more off zero")])

    def test_the_check_of_a_part_is_read_on_the_part_and_that_of_the_entry_on_the_whole(self):
        # Both parts are told to play a#4, and the first plays a4. The whole is told to play a4 where each sounds.
        mark = {"at": 0.01, "note": "a#4"}
        entry = {"parts": [[0, {"def": BEEP, "check": [mark]}], [0.1, {"def": SHARP, "check": [mark]}]],
                 "check": [{"at": 0.01, "note": "a4"}, {"at": 0.11, "note": "a4"}]}
        self.assertEqual(found_in(entry), [
            zap("pitch", "is 100 cents below the note a#4 written at 0.01 seconds, and more than 35 is out of tune",
                part=0),
            zap("pitch", "is 100 cents above the note a4 written at 0.11 seconds, and more than 35 is out of tune")])

    def test_findings_on_parts_come_part_by_part_and_then_on_the_whole(self):
        project = project_of(zap={"parts": [[0, {"def": LOUD}], [0.1, {"def": HISS}]], "max_seconds": 0.01},
                             pop={"def": LOUD})
        self.assertEqual(places(sounds.check(project, RATE)), [
            ("sounds.zap.parts[0]", "peak"), ("sounds.zap.parts[1]", "bright"), ("sounds.zap", "length"),
            ("sounds.pop", "peak")])

    def test_a_held_sound_that_keeps_the_rules_gives_nothing(self):
        self.assertEqual(found_in(held(TONE, 4, 6, 1)), [])
        # With no fade at the join: the loop is a whole number of periods, and comes round by itself.
        self.assertEqual(found_in(held(TONE, 4, 6, 0)), [])

    def test_each_part_of_a_held_sound_is_found_under_its_name_and_held_to_the_limits_of_the_entry(self):
        loud = sound({"gain": 8}, "a4", 12)
        self.assertEqual(places(found_in(held(loud, 2, 4, 1, LOUD, max_seconds=0.01))), [
            ("sounds.zap.start", "peak"), ("sounds.zap.start", "length"),
            ("sounds.zap.loop", "peak"), ("sounds.zap.loop", "length"),
            ("sounds.zap.end", "peak"), ("sounds.zap.end", "length")])

    def test_the_inner_ends_of_a_held_sound_are_not_held_to_silence_nor_its_start_or_loop_to_dc_alone(self):
        entry = held(DEEP, 4, 6, 1)
        start, loop, _ = held_parts(project_of(zap=entry).sound("zap"), RATE)
        start, loop = (measure(normalise(part), RATE) for part in (start, loop))
        # Each of these alone is past its limit: a hundredth for a first or last sample, 0.002 for an offset.
        self.assertEqual((start["tail_step"], loop["head_step"], loop["tail_step"]), (0.099, 0.242, 0.243))
        self.assertEqual((start["dc"], loop["dc"]), (0.0061, -0.0031))
        self.assertEqual(found_in(entry), [])

    def test_the_start_of_a_held_sound_is_held_to_its_first_sample(self):
        self.assertEqual(found_in(held(GROWL, 2, 4, 1, max_dc=0.1)), [
            zap("ends", "starts with a jump of 0.015 out of silence, and 0.01 or more clicks", part="start")])

    def test_the_end_of_a_held_sound_is_held_to_every_rule(self):
        self.assertEqual(found_in(held(DEEP, 4, 6, 1, RUMBLE)), [
            zap("ends", "starts with a jump of 0.015 out of silence, and 0.01 or more clicks", part="end")])
        self.assertEqual(found_in(held(DEEP, 4, 6, 1, THUD)), [
            zap("dc", "its mean sample is 0.0077, which is 0.002 or more off zero", part="end")])

    def test_the_check_of_a_held_entry_is_read_on_each_of_its_parts(self):
        message = "is 100 cents below the note a#4 written at 0.01 seconds, and more than 35 is out of tune"
        self.assertEqual(found_in(held(TONE, 4, 6, 1, check=[{"at": 0.01, "note": "a#4"}])),
                         [zap("pitch", message, part) for part in ("start", "loop", "end")])

    def test_join_fires_where_a_loop_with_no_fade_does_not_come_round_to_its_first_sample(self):
        # Six ticks and two fifths of a4 are not a whole number of periods.
        self.assertEqual(found_in(held(TONE, 4.1, 6.4, 0)), [
            zap("join", "steps from the end of its loop round to its beginning by 1.318 of the largest step "
                        "inside the loop, and 0.25 or more clicks")])
        self.assertEqual(found_in(held(TONE, 4.1, 6.4, 1)), [])

    def test_join_fires_where_the_start_is_cut_from_the_loop_on_a_tall_step(self):
        # Two ticks and a quarter in, the pulse is on its way from one side to the other.
        self.assertEqual(found_in(held(TONE, 2.25, 4, 0)), [
            zap("join", "steps from its start into its loop by 0.952 of the largest step inside the loop, "
                        "and 0.25 or more clicks")])
        # A fade ends the loop on the last sample of the start, and the loop comes round over the same step.
        self.assertEqual(found_in(held(TONE, 3, 6, 1)), [
            zap("join", "steps from its start into its loop by 0.560 of the largest step inside the loop, "
                        "and 0.25 or more clicks"),
            zap("join", "steps from the end of its loop round to its beginning by 0.560 of the largest step "
                        "inside the loop, and 0.25 or more clicks")])

    def test_a_flat_loop_clicks_where_a_join_steps_at_all_and_nowhere_else(self):
        # A loop of one sample has no step inside it, and comes round to itself without one.
        self.assertEqual(found_in(held(TONE, 4, 0.001, 0)), [
            zap("join", "steps from its start into its loop by 0.000197, and the loop is flat: any step into it "
                        "clicks")])
        # The first sample of the note is silence, and here it is all there is of the start and the loop.
        self.assertEqual(found_in(held(TONE, 0, 0.001, 0)), [])

    def test_with_no_start_the_loop_is_held_to_its_first_sample_and_nothing_joins_it_but_itself(self):
        self.assertEqual(found_in(held(GROWL, 0, 4, 0, max_dc=0.2)), [
            zap("ends", "starts with a jump of 0.015 out of silence, and 0.01 or more clicks", part="loop"),
            zap("join", "steps from the end of its loop round to its beginning by 0.305 of the largest step "
                        "inside the loop, and 0.25 or more clicks")])

    def test_dc_fires_on_the_offset_of_the_start_and_the_loop_together_and_max_dc_moves_its_limit(self):
        self.assertEqual(found_in(held(DEEPER, 4, 6, 1)), [
            zap("dc", "the mean sample of its start and its loop together is 0.0066, which is 0.002 or more off "
                      "zero")])
        self.assertEqual(found_in(held(DEEPER, 4, 6, 1, max_dc=0.01)), [])
        # A mean at the limit breaks it.
        self.assertEqual(found_in(held(DEEP, 4, 6, 1, max_dc=0.0017)), [
            zap("dc", "the mean sample of its start and its loop together is 0.0017, which is 0.0017 or more off "
                      "zero")])

    def test_findings_on_a_held_sound_come_part_by_part_then_join_by_join_then_the_offset(self):
        project = project_of(zap=held(TONE, 2.25, 4, 1, max_seconds=0.03), pop={"def": LOUD})
        self.assertEqual(places(sounds.check(project, RATE)), [
            ("sounds.zap.start", "length"), ("sounds.zap.loop", "length"), ("sounds.zap.end", "length"),
            ("sounds.zap", "join"), ("sounds.zap", "join"), ("sounds.zap", "dc"), ("sounds.pop", "peak")])

    def test_a_def_that_cannot_be_rendered_is_refused_with_its_target_in_front(self):
        odd = sound({"wave": "sine"}, "a4", 4)
        self.assertEqual(self.refusal_of({"defs": [BEEP, odd]}),
                         "sounds.zap[1]: voices[0].inst.wave: unknown wave 'sine'")
        self.assertEqual(self.refusal_of({"parts": [[0, {"def": BEEP}], [0, {"def": odd}]]}),
                         "sounds.zap.parts[1]: voices[0].inst.wave: unknown wave 'sine'")

    def test_parts_the_bank_cannot_add_up_are_refused_with_the_sound_in_front(self):
        self.assertEqual(self.refusal_of({"parts": [[-1, {"def": BEEP}]]}),
                         "sounds.zap: parts[0]: seconds: must be 0 or more, not -1")

    def test_a_held_sound_the_bank_cannot_cut_is_refused_with_the_sound_in_front(self):
        self.assertEqual(self.refusal_of(held(BEEP, 2, 40, 1)),
                         "sounds.zap: held: note: must be long enough for the start and the loop: it is 5880 "
                         "samples at 44100 Hz, and they take 30870")
        self.assertEqual(self.refusal_of(held(TONE, 4, 6, 1, sound({"wave": "sine"}, "a4", 4))),
                         "sounds.zap: held: end: voices[0].inst.wave: unknown wave 'sine'")


if __name__ == "__main__":
    unittest.main()
