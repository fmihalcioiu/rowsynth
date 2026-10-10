# Recipes

Each block is a whole project file, and each passes `check`. Take the nearest one, change its notes first and its tables after, and run the loop on what you made.

## A coin

```json
{
  "instruments": {
    "chime": {
      "wave": "pulse",
      "duty": 2,
      "vol": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
      "release": 3
    }
  },
  "sounds": {
    "coin": {"def": {"voices": [{"inst": "chime", "notes": [[0, "b5", 4], [4, "e6", 14]]}]}}
  }
}
```

Two notes on a square wave, the second a fourth above the first and held longer. The `"vol"` table loses one step a tick, and that slow fall is the ring of it.

## A jump

```json
{
  "instruments": {
    "hop": {
      "wave": "pulse",
      "duty": 1,
      "vol": [11, 13, 12, 11, 10, 8, 6, 4, 2],
      "pitch": [0, 3, 3, 2, 2, 2, 1, 1, 1],
      "release": 2
    }
  },
  "sounds": {
    "jump": {"def": {"voices": [{"inst": "hop", "notes": [[0, "c4", 9]]}]}}
  }
}
```

One note whose `"pitch"` table climbs 15 semitones in nine ticks, fast at first and slower after. Turn the signs and it falls.

## A laser

```json
{
  "instruments": {
    "zap": {
      "wave": "pulse",
      "duty": [0, 1, 2, 1],
      "vol": [15, 14, 12, 10, 8, 6, 4, 2],
      "pitch": [0, -4, -4, -4, -3, -3, -3, -3],
      "release": 1,
      "lp": 5000
    }
  },
  "sounds": {
    "laser": {"def": {"voices": [{"inst": "zap", "notes": [[0, "c6", 8]]}]}}
  }
}
```

A high note that drops two octaves in eight ticks while the pulse changes width under it. The `"lp"` cuts the top of the narrow pulse.

## An explosion

```json
{
  "instruments": {
    "rumble": {
      "wave": "noise",
      "vol": [15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1],
      "pitch": {"v": [0, -1], "loop": 1},
      "lp": [5000, 4000, 3200, 2600, 2100, 1700, 1400, 1200, 1000, 850, 700, 600],
      "release": 6
    },
    "thump": {
      "wave": "tri",
      "vol": [15, 14, 12, 10, 8, 6, 4, 2],
      "pitch": {"v": [0, -2], "loop": 1},
      "release": 2
    }
  },
  "sounds": {
    "explosion": {
      "parts": [
        [0, {"def": {"voices": [{"inst": "thump", "notes": [[0, "c3", 10]]}]}, "db": -5}],
        [0.02, {"def": {"voices": [{"inst": "rumble", "notes": [[0, "c5", 30]]}]}, "db": -6}]
      ]
    }
  }
}
```

Two parts on a timeline. A triangle thump falls two semitones a tick, and 0.02 seconds after it starts a noise whose low-pass closes from 5000 Hz to 600 while it sinks a semitone a tick. Each part has a `"db"` of its own.

## A blip in variants

```json
{
  "instruments": {
    "tick": {"wave": "pulse", "duty": 2, "vol": [10, 7, 3], "release": 1}
  },
  "sounds": {
    "blip": {
      "defs": [
        {"voices": [{"inst": "tick", "notes": [[0, "a5", 3]]}]},
        {"voices": [{"inst": "tick", "notes": [[0, "c6", 3]]}]},
        {"voices": [{"inst": "tick", "notes": [[0, "e6", 3]]}]}
      ]
    }
  }
}
```

One short tick on three notes of a chord. A player picks one each time, so a sound heard often does not repeat exactly. `--variant 1` renders the second.

## A power-up with an echo

```json
{
  "instruments": {
    "chime": {
      "wave": "pulse",
      "duty": 2,
      "vol": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
      "release": 3
    }
  },
  "sounds": {
    "powerup": {
      "def": {
        "voices": [
          {
            "inst": "chime",
            "notes": [[0, "c5", 4], [4, "e5", 4], [8, "g5", 4], [12, "c6", 4], [16, "e6", 4], [20, "g6", 12]]
          }
        ],
        "echo": {"time": 0.09, "feedback": 0.35, "mix": 0.3, "lp": 4000}
      }
    }
  }
}
```

Six notes up a chord, four ticks each and the last held, with an echo that comes back every 0.09 seconds. The echo is what makes the sound 1.2 seconds long.

## A held hum

```json
{
  "instruments": {
    "drone": {"wave": "saw", "vol": [2, 5, 8, 11, 13, 14], "lp": 900, "release": 2},
    "drone_end": {"wave": "saw", "vol": [14, 12, 10, 8, 6, 5, 4, 3, 2, 1], "lp": 900, "release": 4}
  },
  "sounds": {
    "hum": {
      "held": {
        "note": {"voices": [{"inst": "drone", "notes": [[0, "a2", 90]]}]},
        "start": 12,
        "loop": 60,
        "join": 6,
        "end": {"voices": [{"inst": "drone_end", "notes": [[0, "a2", 10]]}]},
        "let_go": 4
      },
      "db": -9,
      "check": [{"at": 0.1, "note": "a2"}]
    }
  }
}
```

One long note of a saw under a low-pass, cut in two: its first 12 ticks are the start, where the level rises, and the 60 after them are the loop, which goes round for as long as the sound is held. a2 is 110 Hz, so a loop of one second is 110 whole periods. The end is the same note on an instrument that falls away. `--part loop` renders or measures the loop, and `--part end` the end.

## What they measure

`measure` gives these for the recipes, at 44100 samples a second. Read them as what sounds that pass look like in numbers. None of them is a target.

| Sound | `seconds` | `rms_db` | `centroid_hz` | `above_4k` |
|---|---|---|---|---|
| coin | 0.383 | -11.3 | 1846 | 0.06 |
| jump | 0.217 | -13.0 | 971 | 0.04 |
| laser | 0.183 | -14.1 | 980 | 0.02 |
| explosion | 0.653 | -20.2 | 441 | 0.0 |
| blip, variant 0 | 0.1 | -9.5 | 1454 | 0.07 |
| powerup | 1.247 | -13.6 | 1589 | 0.06 |
| hum, its loop | 1.0 | -20.2 | 178 | 0.0 |
