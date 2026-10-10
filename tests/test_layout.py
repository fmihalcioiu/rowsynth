"""Tests for rowsynth.layout."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from rowsynth.layout import WIDTH, dumps

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# A song with every table a song has, each of them short enough for one line.
SONG = {
    "row_ticks": 6,
    "instruments": {"pad": {"wave": "saw"}, "tap": {"wave": "noise"}},
    "drums": {"k": {"inst": "tap", "note": "c3"}},
    "channels": {"top": {"inst": "pad"}, "kit": {"inst": "tap", "drums": True}},
    "patterns": {"a": {"top": "o4 c1", "kit": "k1"}, "b": {"top": "o4 e1"}},
    "order": ["a", "b"],
    "chords": {"a": ["C"]},
    "layers": {"groups": {"all": ["top", "kit"]}, "versions": {"full": ["all"]}, "keys": [0]},
    "intro": {"order": ["b"], "tail": 1},
}
SONG_TEXT = """{
  "songs": {
    "air": {
      "row_ticks": 6,
      "instruments": {
        "pad": {"wave": "saw"},
        "tap": {"wave": "noise"}
      },
      "drums": {
        "k": {"inst": "tap", "note": "c3"}
      },
      "channels": {
        "top": {"inst": "pad"},
        "kit": {"inst": "tap", "drums": true}
      },
      "patterns": {
        "a": {
          "top": "o4 c1",
          "kit": "k1"
        },
        "b": {
          "top": "o4 e1"
        }
      },
      "order": ["a", "b"],
      "chords": {"a": ["C"]},
      "layers": {
        "groups": {"all": ["top", "kit"]},
        "versions": {"full": ["all"]},
        "keys": [0]
      },
      "intro": {"order": ["b"], "tail": 1}
    }
  }
}
"""


def lines(data: dict) -> list[str]:
    """The lines of the text `data` is written as."""
    return dumps(data).splitlines()


class Fitting(unittest.TestCase):
    def test_a_value_that_fits_is_on_one_line(self) -> None:
        data = {"instruments": {"deep": {"wave": "tri", "vol": [15, 9], "vib": {"depth": 0.2}}}}
        self.assertEqual(
            dumps(data),
            '{\n  "instruments": {\n    "deep": {"wave": "tri", "vol": [15, 9], "vib": {"depth": 0.2}}\n  }\n}\n')

    def test_a_step_table_is_opened_out_only_when_it_ends_past_the_width(self) -> None:
        fits = [10] + [5] * 31
        over = [10, 10] + [5] * 30
        text = lines({"instruments": {"hiss": {"vol": fits, "arp": over}}})
        # The table that fits ends at the last column. The comma after it is not counted.
        self.assertEqual(text[3], '      "vol": ' + json.dumps(fits) + ",")
        self.assertEqual(len(text[3]), WIDTH + 1)
        self.assertEqual(text[4:8], ['      "arp": [', "        10,", "        10,", "        5,"])
        self.assertEqual(text[36:], ["        5", "      ]", "    }", "  }", "}"])
        self.assertEqual(len(text), 41)

    def test_a_value_is_measured_from_where_it_starts_on_its_line(self) -> None:
        table = [10, 10] + [5] * 28
        text = lines({"instruments": {"a": {"vol": table}, "ab": {"vol": table}}})
        row = '{"vol": ' + json.dumps(table) + "}"
        self.assertEqual(text[2], '    "a": ' + row + ",")
        self.assertEqual(len(text[2]), WIDTH + 1)
        # With one more letter in its name the same instrument ends past the width. Its table still fits a line.
        self.assertEqual(text[3:6], ['    "ab": {', '      "vol": ' + json.dumps(table), "    }"])

    def test_a_list_that_does_not_fit_is_opened_out_one_item_a_line(self) -> None:
        defs = [{"voices": [{"inst": "tick", "notes": [[0, note, 3], [3, note, 2, 0.5]]}]} for note in ("a5", "c6")]
        self.assertEqual(lines({"sounds": {"run": {"defs": defs}}})[2:8], [
            '    "run": {',
            '      "defs": [',
            '        {"voices": [{"inst": "tick", "notes": [[0, "a5", 3], [3, "a5", 2, 0.5]]}]},',
            '        {"voices": [{"inst": "tick", "notes": [[0, "c6", 3], [3, "c6", 2, 0.5]]}]}',
            "      ]",
            "    }",
        ])

    def test_a_text_is_never_broken(self) -> None:
        played = "o4 l16 " + "c e g e " * 40
        song = {**SONG, "patterns": {"a": {"top": played}}}
        row = '          "top": ' + json.dumps(played)
        self.assertGreater(len(row), 3 * WIDTH)
        self.assertIn(row, lines({"songs": {"air": song}}))

    def test_an_empty_dict_and_an_empty_list_stay_closed(self) -> None:
        self.assertEqual(dumps({}), "{}\n")
        data = {"instruments": {}, "sounds": {"ping": {"def": {"voices": []}, "meta": {}}}}
        self.assertEqual(
            dumps(data),
            '{\n  "instruments": {},\n  "sounds": {\n    "ping": {"def": {"voices": []}, "meta": {}}\n  }\n}\n')


class Opened(unittest.TestCase):
    def test_the_top_tables_are_opened_out_however_little_they_hold(self) -> None:
        chords = {"C": {"root": 0, "tones": [0, 4, 7]}}
        data = {"instruments": {"a": {}}, "chords": chords, "more": {"b": 1, "c": [2]}}
        self.assertEqual(lines(data), [
            "{",
            '  "instruments": {',
            '    "a": {}',
            "  },",
            '  "chords": {',
            '    "C": {"root": 0, "tones": [0, 4, 7]}',
            "  },",
            '  "more": {',
            '    "b": 1,',
            '    "c": [2]',
            "  }",
            "}",
        ])

    def test_a_song_is_opened_out_with_one_channel_of_a_pattern_a_line(self) -> None:
        self.assertEqual(dumps({"songs": {"air": SONG}}), SONG_TEXT)

    def test_a_song_with_empty_tables_is_still_opened_out(self) -> None:
        song = {"row_ticks": 6, "channels": {}, "patterns": {"a": {}}, "order": []}
        self.assertEqual(lines({"songs": {"air": song}})[2:9], [
            '    "air": {',
            '      "row_ticks": 6,',
            '      "channels": {},',
            '      "patterns": {',
            '        "a": {}',
            "      },",
            '      "order": []',
        ])


class Written(unittest.TestCase):
    def test_keys_keep_the_order_they_have_in_the_data(self) -> None:
        data = {"songs": {}, "instruments": {"zed": {"vol": 1, "wave": "saw", "arp": 0}, "ace": {}}}
        self.assertEqual(lines(data), [
            "{",
            '  "songs": {},',
            '  "instruments": {',
            '    "zed": {"vol": 1, "wave": "saw", "arp": 0},',
            '    "ace": {}',
            "  }",
            "}",
        ])

    def test_text_that_is_not_ascii_is_written_as_it_is(self) -> None:
        data = {"instruments": {"fl\xfbte": {}}, "sounds": {"se\xf1al": {"meta": {"says": "d\xeda"}}}}
        text = lines(data)
        self.assertEqual(text[2], '    "fl\xfbte": {}')
        self.assertEqual(text[5], '    "se\xf1al": {"meta": {"says": "d\xeda"}}')

    def test_numbers_true_false_and_null_are_written_as_json_writes_them(self) -> None:
        values = [12, 0.35, -4.5, 1e-07, 1e22, 2.0, True, False, None]
        said = ["12", "0.35", "-4.5", "1e-07", "1e+22", "2.0", "true", "false", "null"]
        on_one_line = lines({"sounds": {"ping": {"meta": {"all": values}}}})
        self.assertEqual(on_one_line[2], '    "ping": {"meta": {"all": [' + ", ".join(said) + "]}}")
        opened_out = lines({"sounds": {"ping": {"all": values * 3}}})
        self.assertEqual(opened_out[4:13], [f"        {one}," for one in said])

    def test_a_key_that_is_not_a_text_is_written_as_json_writes_it(self) -> None:
        self.assertEqual(
            dumps({7: {True: {}, None: {}, 0.5: {}}}),
            '{\n  "7": {\n    "true": {},\n    "null": {},\n    "0.5": {}\n  }\n}\n')

    def test_each_example_reads_back_as_the_data_it_was_written_from(self) -> None:
        for path in sorted(EXAMPLES.glob("*.json")):
            with self.subTest(example=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                read = json.loads(dumps(data))
                self.assertEqual(read, data)
                self.assertEqual(json.dumps(read), json.dumps(data))
