# Songs: the text language, chords, layers and the build

A whole project file with one song of two bars. It passes `check`.

```json
{
  "instruments": {
    "lead": {"wave": "pulse", "duty": 1, "vol": [12, 14, 12, 10, 9, 8], "release": 3},
    "bass": {"wave": "tri", "vol": [15, 14, 13, 12], "release": 2, "gate": 0.8},
    "kick": {"wave": "tri", "vol": [15, 14, 12, 9, 6, 3], "pitch": [0, -6, -5, -4, -3, -2], "release": 1},
    "snare": {"wave": "noise", "vol": [14, 11, 8, 6, 4, 2, 1], "release": 2, "lp": 7000},
    "hat": {"wave": "noise", "vol": [8, 4, 1], "release": 1, "lp": 6000}
  },
  "chords": {
    "Am": {"root": 9, "tones": [9, 0, 4]},
    "F": {"root": 5, "tones": [5, 9, 0]}
  },
  "songs": {
    "tune": {
      "row_ticks": 6,
      "drums": {
        "k": {"inst": "kick", "note": "c3", "rows": 2},
        "s": {"inst": "snare", "note": "c5", "rows": 2},
        "h": {"inst": "hat", "note": "c6", "rows": 1}
      },
      "channels": {
        "lead": {"inst": "lead", "gain": 0.8, "pan": 0.2},
        "bass": {"inst": "bass", "gain": 0.9},
        "kit": {"inst": "kick", "gain": 0.8, "drums": true}
      },
      "patterns": {
        "a": {
          "lead": "o5 l8 e g a g e4 c4 | c f a f c2",
          "bass": "o2 l4 a a a a | f f f f",
          "kit": "l8 k h s h k k s h | k h s h k k s h"
        }
      },
      "order": ["a"],
      "chords": {"a": ["Am", "F"]},
      "layers": {
        "groups": {"rhythm": ["kit"], "low": ["bass"], "top": ["lead"]},
        "versions": {"beat": ["rhythm"], "thin": ["rhythm", "low"], "full": ["rhythm", "low", "top"]},
        "keys": [0, 3]
      }
    }
  }
}
```

## The keys of a song

| Key | Left out | Is |
|---|---|---|
| `row_ticks` | needed | The ticks in a row, sixty ticks to a second. It need not be whole |
| `rows_per_beat` | 4 | The rows in a beat. A bar is four beats |
| `gain` | 0.35 | The level of the whole song |
| `channels` | needed | `{name: {"inst", "gain", "pan", "drums", "echo"}}`. Only `"inst"` is needed. The gain is 1 and the pan 0 without one, and the pan runs from -1 on the left to 1 on the right |
| `drums` | none | What each drum letter plays: `{letter: {"inst", "note", "rows", "vel"}}`, the last two optional |
| `patterns` | needed | `{name: {channel: text}}` |
| `order` | needed | The patterns to play, end to end. A name can come more than once |
| `instruments` | none | Instruments of the song's own, laid over the file's: a name with an instrument, or with the name of one of the file's |
| `chords` | none | `{pattern: [one string for each bar]}` |
| `layers` | none | Groups, versions and keys |

A channel's `"echo"` is `{"rows": 3, "gain": 0.3}`, both needed: every note is played again that many rows later at that share of its velocity.

A note is held for the share of its rows that its instrument's `"gate"` gives, 0.9 without one and one tick at the least, and then fades for the instrument's `"release"`.

## The text of a channel

| Text | Means |
|---|---|
| `c d e f g a b` | A note. `+` or `#` after the letter makes it sharp and `-` flat: `f+`, `b-` |
| `r` | A rest |
| a number after either | The length: `1` a whole note, `2` a half, `4` a quarter, `8` an eighth, `16` a sixteenth. Without one the length is what `l` last set |
| `.` after that | Half as long again: `c4.`. A second dot adds a quarter more |
| `l8` | The length of what is written without one, from here on |
| `o4` | The octave from here on, one digit. In octave 4, `c` is middle C |
| `>` and `<` | An octave up, and an octave down |
| `&` | A tie. The note after it, when it is the same note, makes the one before it longer: `c1 & c4` |
| `v12` | The volume from here on, 0 to 15 |
| `@name` | The instrument from here on, in place of the channel's own |
| `k s h H t` | A drum hit, timed as a note is: `k4`, `h8.`. Only in a channel with `"drums": true` |

Spaces and the bar line `|` change nothing. They are there for you to read by.

Every text starts afresh: at octave 4, a length of 16, a volume of 15 and the channel's own instrument. What a pattern set does not carry into the next one.

A drum letter plays the `"inst"` of its entry in `"drums"` at its `"note"`. `"rows"` caps how long the hit is held, whatever length it is written with, and `"vel"` scales it. The five letters are only names: which one is the kick is up to `"drums"`. A note letter in a drum channel plays the channel's own instrument.

## Rows, and channels of one length

At four rows to a beat:

| Length | 1 | 2 | 2. | 4 | 4. | 8 | 8. | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| Rows | 16 | 8 | 12 | 4 | 6 | 2 | 3 | 1 | 0.5 |

A row lasts `row_ticks / 60` seconds, so the tempo is `3600 / (row_ticks * rows_per_beat)` beats a minute.

A pattern is as long as its longest channel, and `check` wants every channel in it to be that long. To keep them so:

- Write a bar line after every bar, and count each bar to 16 rows.
- Set `l` once at the start of a text and write lengths only on the notes that differ.
- A channel that is silent for a pattern is left out of the pattern, or written as rests: `r1 | r1`.
- When `check` reports `channel 'bass' has 62 rows, and the longest channel of the pattern has 64`, the bass is two rows short, an eighth note, somewhere in that pattern.
- Keep to the lengths of the table. A length of 12 is a triplet eighth, three to a beat, but thirds of a row do not always add up exactly: a bar of six notes of length 6 comes to 15.999999999999998 rows, and `check` reports it as shorter than a bar of 16.

## Chords

The file's `"chords"` table names each chord by its pitch classes, where c is 0, c# 1, d 2 and so on up to b at 11: A minor is `{"root": 9, "tones": [9, 0, 4]}`. A song's `"chords"` gives a pattern one string for each bar. A string names one chord, or several with spaces between them, which share the bar evenly.

`check` then holds every note that starts on a beat to the chord under it, on every channel but the drum channels. A note between beats can be anything. Chords are optional, and a pattern without them is not held to any.

Write the bass on the root of each bar's chord and the lead's notes on the beat from its tones, and the rule passes on its own.

## Layers

A song with `"layers"` is built as files that a player adds up while they loop in step.

| Key | Is |
|---|---|
| `groups` | `{group: [channels]}`. Each group becomes a file. Every channel is in one group and no more |
| `versions` | `{version: [groups]}`. A version is the groups heard together |
| `keys` | The keys the song is built in, each a number of semitones up from how it is written. `[0]` when left out |
| `premix` | Optional: `[{"version": "full", "key": 0}]`, versions to mix ahead into one file each |

A drum channel is not moved with the key, so a group of drum channels alone is the same in every key and gets one file.

`check` holds every version in every key to the `level` rule and every group to the `join` rule. A song without layers has the mix of all its channels held to `level`.

## What `build` writes

`python -m rowsynth build audio.json --out out/music` writes into the folder, at 48000 samples a second unless `--rate` says otherwise:

| File | Is |
|---|---|
| `music.json` | The manifest: `{"rate": ..., "songs": {name: entry}}` |
| `NAME.wav` | A song without layers: one stereo loop, its loudest sample at -1 dBFS |
| `NAME-GROUP-KEY.wav` | A layered song: one stereo loop for each group in each key |
| `NAME-GROUP.wav` | A group of drum channels alone |
| `NAME.VERSION.KEY.wav` | A version of `"premix"` |

The files of the groups of one song all share one scale, with the loudest sample among them at -1 dBFS, so that adding the files of a version gives that version.

The entry of a song without layers has `"loop_samples"`, `"gain"`, `"source"` and `"file"`. That of a layered song has `"loop_samples"`, `"bar_samples"`, `"gain"`, `"source"`, `"keys"`, `"groups"` and `"versions"`, and `"premixed"` when the song has a premix. `"gain"` is what a player multiplies the samples by to have the song at its own level again. `"source"` is 16 hex digits that change when the song, the rate or the package's version does. In `"groups"` a group has `{"files": {"0": file, "3": file}}`, a file for each key, or `{"file": file}` when it is drums alone.

`build` refuses a layered song with a channel in no group or in two. It leaves alone whatever else the folder holds.
