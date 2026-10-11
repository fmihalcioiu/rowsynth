"""Tests for rowsynth.gui.history."""
from __future__ import annotations

import unittest

from rowsynth.edit import NOTHING
from rowsynth.gui.history import MAX_STEPS, History, same

LEAD = "songs.tune.patterns.a.lead"
GAIN = "songs.tune.gain"


class SameTest(unittest.TestCase):
    def test_two_values_are_the_same_when_neither_is_there_or_both_are_written_alike(self):
        for one, other, alike in ((NOTHING, NOTHING, True), (NOTHING, None, False), (0.5, NOTHING, False),
                                  ({"a": [1, 2]}, {"a": [1, 2]}, True), (1, True, False), (1, 1.0, False),
                                  ({"a": 1, "b": 2}, {"b": 2, "a": 1}, False)):
            with self.subTest(one=one, other=other):
                self.assertEqual(same(one, other), alike)


class HistoryTest(unittest.TestCase):
    def test_a_history_with_no_step_has_nothing_to_take_back_or_to_make_again(self):
        history = History()
        self.assertEqual([history.can_undo, history.can_redo, history.undo_label, history.redo_label],
                         [False, False, "", ""])
        self.assertIsNone(history.undo())
        self.assertIsNone(history.redo())
        self.assertEqual(history.limit, MAX_STEPS)

    def test_undo_gives_the_opposite_of_the_last_step_and_moves_it_to_those_that_can_be_made_again(self):
        history = History()
        history.add([(LEAD, "c e", "c g")], "lead of pattern a")
        history.add([(GAIN, 0.5, 0.4)], "gain of tune")
        self.assertEqual([history.can_undo, history.can_redo, history.undo_label, history.redo_label],
                         [True, False, "gain of tune", ""])
        self.assertEqual(history.undo(), ((GAIN, 0.4, 0.5),))
        self.assertEqual([history.can_undo, history.can_redo, history.undo_label, history.redo_label],
                         [True, True, "lead of pattern a", "gain of tune"])
        self.assertEqual(history.undo(), ((LEAD, "c g", "c e"),))
        self.assertEqual([history.can_undo, history.undo_label, history.redo_label],
                         [False, "", "lead of pattern a"])
        self.assertIsNone(history.undo())

    def test_redo_gives_the_step_taken_back_last_as_it_was_made_and_moves_it_back(self):
        history = History()
        history.add([(LEAD, "c e", "c g")], "lead of pattern a")
        history.add([(GAIN, 0.5, 0.4)], "gain of tune")
        history.undo()
        history.undo()
        self.assertEqual(history.redo(), ((LEAD, "c e", "c g"),))
        self.assertEqual([history.undo_label, history.redo_label], ["lead of pattern a", "gain of tune"])
        self.assertEqual(history.redo(), ((GAIN, 0.5, 0.4),))
        self.assertEqual([history.can_undo, history.can_redo], [True, False])
        self.assertIsNone(history.redo())
        self.assertEqual(history.undo(), ((GAIN, 0.4, 0.5),))

    def test_the_changes_of_a_step_are_taken_back_from_the_last_to_the_first(self):
        history = History()
        made = (("instruments.bell", NOTHING, {"wave": "saw"}), ("sounds.zap", {"old": 1}, {"new": 1}),
                ("sounds.pop", {"gone": 1}, NOTHING))
        history.add(iter(made), "changed outside: bell, pop, zap")
        self.assertEqual(history.undo(), (("sounds.pop", NOTHING, {"gone": 1}), ("sounds.zap", {"new": 1}, {"old": 1}),
                                          ("instruments.bell", {"wave": "saw"}, NOTHING)))
        self.assertEqual(history.redo(), made)

    def test_a_new_step_empties_what_could_be_made_again(self):
        history = History()
        history.add([(GAIN, 0.5, 0.4)], "gain of tune")
        history.add([(GAIN, 0.4, 0.3)], "gain of tune")
        history.undo()
        history.add([(LEAD, "c e", "c g")], "lead of pattern a")
        self.assertEqual([history.can_redo, history.redo_label], [False, ""])
        self.assertIsNone(history.redo())
        self.assertEqual(history.undo(), ((LEAD, "c g", "c e"),))
        self.assertEqual(history.undo(), ((GAIN, 0.4, 0.5),))

    def test_steps_with_the_same_key_one_after_the_other_are_one_step_from_the_first_was_to_the_last_now(self):
        history = History()
        history.add([(LEAD, "c e", "c g")], "lead of pattern a")
        for was, now in ((0.5, 0.45), (0.45, 0.4), (0.4, 0.35)):
            history.add([(GAIN, was, now)], f"gain of tune to {now}", key="gain")
        self.assertEqual(history.undo_label, "gain of tune to 0.35")
        self.assertEqual(history.undo(), ((GAIN, 0.35, 0.5),))
        self.assertEqual(history.undo_label, "lead of pattern a")
        self.assertEqual(history.redo(), ((GAIN, 0.5, 0.35),))

    def test_steps_with_no_key_or_with_another_key_stay_apart(self):
        for keys in ((None, None), ("gain", "level"), ("gain", None)):
            with self.subTest(keys=keys):
                history = History()
                history.add([(GAIN, 0.5, 0.45)], "gain of tune", key=keys[0])
                history.add([(GAIN, 0.45, 0.4)], "gain of tune", key=keys[1])
                self.assertEqual(history.undo(), ((GAIN, 0.4, 0.45),))
                self.assertEqual(history.undo(), ((GAIN, 0.45, 0.5),))

    def test_a_step_with_the_same_key_that_does_not_go_on_where_the_last_ended_is_a_step_of_its_own(self):
        history = History()
        history.add([(GAIN, 0.5, 0.45)], "gain of tune", key="drag")
        # The place was not left as this step finds it: something else changed it in between.
        history.add([(GAIN, 0.2, 0.4)], "gain of tune", key="drag")
        history.add([(GAIN, 0.4, 0.35)], "gain of tune", key="drag")
        history.add([(LEAD, "c e", "c g")], "lead of pattern a", key="drag")
        self.assertEqual(history.undo(), ((LEAD, "c g", "c e"),))
        self.assertEqual(history.undo(), ((GAIN, 0.35, 0.2),))
        self.assertEqual(history.undo(), ((GAIN, 0.45, 0.5),))
        self.assertFalse(history.can_undo)

    def test_a_step_is_not_joined_to_one_that_was_taken_back_and_made_again(self):
        history = History()
        history.add([(GAIN, 0.5, 0.45)], "gain of tune", key="gain")
        history.undo()
        history.redo()
        history.add([(GAIN, 0.45, 0.4)], "gain of tune", key="gain")
        self.assertEqual(history.undo(), ((GAIN, 0.4, 0.45),))
        self.assertTrue(history.can_undo)

    def test_a_step_that_ends_where_it_began_is_not_kept(self):
        history = History()
        history.add([(GAIN, 0.5, 0.4)], "gain of tune")
        history.undo()
        for changes in ([], [(GAIN, 0.5, 0.5)], [(LEAD, "c e", "c e"), ("sounds.zap", NOTHING, NOTHING)]):
            history.add(changes, "nothing")
        # Nothing was made, so what could be made again still can.
        self.assertEqual([history.can_undo, history.redo_label], [False, "gain of tune"])
        # One change of a step that ends elsewhere keeps the whole step.
        history.add([(GAIN, 0.5, 0.5), (LEAD, "c e", "c g")], "both")
        self.assertEqual(history.undo(), ((LEAD, "c g", "c e"), (GAIN, 0.5, 0.5)))

    def test_steps_joined_back_to_where_the_first_began_are_not_kept(self):
        history = History()
        history.add([(LEAD, "c e", "c g")], "lead of pattern a")
        history.add([(GAIN, 0.5, 0.4)], "gain of tune", key="gain")
        history.add([(GAIN, 0.4, 0.5)], "gain of tune", key="gain")
        self.assertEqual(history.undo_label, "lead of pattern a")
        # The drag goes on: what follows is a step of its own, and is not joined to the one before the drag.
        history.add([(GAIN, 0.5, 0.6)], "gain of tune", key="gain")
        self.assertEqual(history.undo(), ((GAIN, 0.6, 0.5),))
        self.assertEqual(history.undo(), ((LEAD, "c g", "c e"),))

    def test_no_more_steps_than_the_limit_are_kept_and_the_oldest_go_first(self):
        history = History(limit=3)
        for at in range(5):
            history.add([(GAIN, at, at + 1)], f"step {at}")
        taken = []
        while history.can_undo:
            taken.append((history.undo_label, history.undo()))
        self.assertEqual(taken, [("step 4", ((GAIN, 5, 4),)), ("step 3", ((GAIN, 4, 3),)),
                                 ("step 2", ((GAIN, 3, 2),))])
        self.assertEqual(history.redo_label, "step 2")


if __name__ == "__main__":
    unittest.main()
