# Sounds: the four forms, the echo and FM

An entry of `"sounds"` has one of `"def"`, `"defs"`, `"parts"` and `"held"`, and only one. `references/recipes.md` has a working file of each.

## The forms

| Form | Holds | Level |
|---|---|---|
| `"def"` | One sound: `{"voices": [...], "echo": {...}}` | `"db"`, one number |
| `"defs"` | A list of defs, the variants of one sound | `"db"`, one number for all or a list with one for each |
| `"parts"` | `[[seconds, entry], ...]`: each part is an entry of one `"def"` and starts at its seconds. The parts are added into one buffer | Each part has its `"db"`. The entry has none |
| `"held"` | A start, a loop and an end, cut as the next table says | `"db"`, one number for the three |

`"check"` marks and the three limits hold every buffer of the entry: each variant, and each of the three of a held sound. So keep marks off variants that differ in pitch, and put a held sound's mark at a time that lies inside its start, its loop and its end.

## A held sound

All six keys are needed.

| Key | Is |
|---|---|
| `note` | A def: one long note, rendered once |
| `start` | Ticks. That much of the note from its beginning, played once |
| `loop` | Ticks. What follows the start, played round and round while the sound is held |
| `join` | Ticks. That much of the loop's end is faded into what came just before the loop, so that it comes round to its first sample without a click |
| `end` | A def, played when the sound is let go |
| `let_go` | Ticks a player takes to let the loop go. It is carried to the player and changes no sample |

- The note must last for the start and the loop together. A length need not be a whole number of ticks.
- `"join"` can be no longer than the start, and no longer than the loop. With a join of 0 the loop clicks unless it is a whole number of the note's periods.
- Let `"vol"` reach its last value before the start is over. A loop whose level is still rising swells again every time it comes round.
- A wave with an edge, a saw or a pulse under an `"lp"`, passes the `join` rule. A pure sine from FM breaks it at most places, since the rule sets the step across the join against the largest step inside the loop and a sine has no large one. Move `"start"` by a tenth of a tick at a time until it passes: 12.1 in place of 12.
- A loop of a low note that is short and no whole number of periods gets a `dc` finding. Make it longer.

## The echo

```json
{
  "sounds": {
    "ping": {
      "def": {
        "voices": [
          {
            "inst": {"wave": "saw", "vol": [15, 12, 9, 6, 3, 1], "lp": [4000, 2500, 1500, 900], "release": 3},
            "notes": [[0, "a4", 6], [3, "e5", 4, 0.5]]
          }
        ],
        "echo": {"time": [0.08, 0.13], "feedback": 0.3, "mix": 0.3, "lp": 3000}
      }
    }
  }
}
```

| Key | Left out | Does |
|---|---|---|
| `time` | needed | Seconds until the sound comes back. A list gives one delay line for each time, and they share the mix |
| `feedback` | 0.3 | The share of each return that comes back again. Held between 0 and 0.95 |
| `mix` | 0.3 | The level of the first return against the sound |
| `lp` | 3000 | The cutoff in Hz that every return goes through once more, so each is darker than the last |

The sound grows longer by its returns, down to the one that is 60 dB under the first. An echo is on one def. A channel of a song has an echo of another kind, in `references/songs.md`.

## FM

An instrument with `"wave": "fm"` is four operators, each a sine under an envelope. An operator is either heard or bends the phase of another, which adds overtones: the more bend, the brighter. `"vol"`, `"lp"`, `"release"` and the other keys work on the voice as on any wave.

```json
{
  "instruments": {
    "bell": {
      "wave": "fm",
      "alg": 4,
      "ops": [
        {"mul": 3.5, "level": 1.5, "env": {"a": 0, "d": 0.4, "s": 0, "r": 0.3}},
        {"mul": 1, "level": 1, "env": {"a": 0, "d": 1.2, "s": 0, "r": 0.4}}
      ],
      "release": 30
    }
  },
  "sounds": {
    "bell": {
      "def": {"voices": [{"inst": "bell", "notes": [[0, "a4", 20]]}]},
      "check": [{"at": 0.3, "note": "a4"}]
    }
  }
}
```

| Key | Left out | Is |
|---|---|---|
| `alg` | 4 | Which operator bends which, 0 to 7, as the table below has it |
| `fb` | 0 | How far operator 0 bends itself, in radians. At 1 it adds overtones, and at 2 it has turned to noise |
| `ops` | all silent | Up to four operators, numbered from 0. One left out is silent |
| `mul` | 1 | The operator's frequency as a multiple of the note's. A whole number keeps the sound in tune with itself, and 3.5 gives a bell |
| `det` | 0 | Hz added to that, for a slow beat between two operators |
| `level` | 1 | Its level when it is heard, and radians of bend when it bends another |
| `env` | | `"a"` seconds to rise (0), `"d"` seconds to fall to the sustain (0.3), `"s"` the sustain from 0 to 1 (1), `"sd"` dB a second lost while held (0), `"r"` seconds to fall 60 dB once let go (0.2) |

| `alg` | Bends | Heard |
|---|---|---|
| 0 | 0 bends 1, 1 bends 2, 2 bends 3 | 3 |
| 1 | 0 and 1 bend 2, 2 bends 3 | 3 |
| 2 | 1 bends 2, 0 and 2 bend 3 | 3 |
| 3 | 0 bends 1, 1 and 2 bend 3 | 3 |
| 4 | 0 bends 1, 2 bends 3 | 1 and 3 |
| 5 | 0 bends 1, 2 and 3 | 1, 2 and 3 |
| 6 | 0 bends 1 | 1, 2 and 3 |
| 7 | nothing | all four |

The envelopes are let go when the note is, and the instrument's `"release"` still fades the whole voice from there. Give it as many ticks as the longest `"r"` needs, or the ring is cut: 30 ticks is half a second. Early in a note a strong bend can read an octave low, so put a `"check"` mark where the bend has died down.
