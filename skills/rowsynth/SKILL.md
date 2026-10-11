---
name: rowsynth
description: Write sound effects and songs as data with the rowsynth Python package, a chip synth driven by text, then render them to WAV, measure them, check them and build a page to hear them on. Use this skill whenever the user asks for a sound effect, a jingle, chip music, a music loop for a game, music in layers or stems, and whenever they ask to change, fix or review a rowsynth project file, even if they do not name rowsynth.
---

# Making sounds and songs with rowsynth

rowsynth keeps sound as text. A sound effect is a few notes on an instrument, a song is channels of notes in a small language, and both sit in one JSON project file. You write the data; the package renders it to WAV, measures it and checks it.

You cannot hear what you wrote. Data that reads well comes out an octave off, harsh, clicking or too loud. So the work is a loop: write, render, read the numbers, check, fix. The numbers are what you have in place of ears, and at the end a person listens. Never hand over a sound you have not measured and checked.

## Before you start

See that the package answers:

```
python -m rowsynth -h
```

If it does not, it is not installed. In a clone of the rowsynth repo, run `pip install .`, or work from the repo's root folder, where `python -m rowsynth` runs without an install as long as NumPy is there. A script of your own that imports `rowsynth` without an install needs the repo's root in the `PYTHONPATH` environment variable, since Python looks beside the script and not in the working folder.

The repo's `examples/arcade.json` holds six sound effects and a song in layers, and `examples/piece.json` a longer song on FM instruments. Both pass `check`.

## The loop

1. Decide what the sound is for and how long it should last. For a song, decide the tempo, the key and the length in bars. If the user did not give them, choose and say what you chose in one line.
2. Write the project file (next sections). Start from a recipe in `references/recipes.md` when one is close. For a song, read `references/songs.md` first.
3. Render it and read what comes back: the length, the loudness, the offset, the step at each end.
4. Ask for the pitch at a time when one note sounds alone, and hold it against the note you wrote. Read the brightness.
5. Run `check`. Fix every finding, or decide it is meant. Go back to step 3. Two or three rounds is normal.
6. Build the listen page and give the person the path of its `index.html`. They can hear, so what they say about a sound outranks its numbers.
7. Hand over the project file. Say what `check` still reports and why you left it.

```
python -m rowsynth render audio.json coin --out out/coin.wav
python -m rowsynth measure audio.json coin --at 0.15
python -m rowsynth check audio.json
python -m rowsynth page audio.json --out out/page
```

`page` writes `out/page/index.html`, one page with a player for every sound and every song, and their audio files in `out/page/audio`. The files are WAV unless `--format mp3` or `--format opus` asks for smaller ones, which takes ffmpeg on the path. The page is for the person. The numbers are for you.

## Reading the numbers

`render` writes the WAV and prints its figures under `"measure"`. `measure` writes nothing, and prints the same figures with `"brightness"` and, when `--at` is given, `"pitch"`.

| Figure | Says |
|---|---|
| `seconds` | The length. A sound runs two ticks past the end of its last release, and longer with an echo |
| `peak_db` | The loudest sample, in decibels against full scale. Every sound is brought to -3 and from there to its `"db"`, so only a song's peak is news |
| `rms_db` | The loudness. Two sounds with one peak can be 10 dB apart here: set them against each other with `"db"` |
| `dc` | The mean sample, which is the offset. A sound that is played once reads 0.0 |
| `clipped` | How many samples are at full scale. It should read 0 |
| `head_step`, `tail_step` | The size of the first sample and of the last. In a sound that is played once, 0.01 or more is a jump out of silence or into it, and clicks |
| `centroid_hz` | Where the middle of the energy sits. Higher is brighter |
| `above_4k` | The share of the energy above 4 kHz, from 0 to 1. `check` reports more than 0.3 as too bright |
| `hz` | The note sounding at `--at` seconds, or 0.0 when none can be read |

The start and the loop of a held sound and the stems of a song are cut out of something longer, so their offset and their ends read otherwise. `check` knows which rule holds which of them.

Times are in seconds and notes in ticks, sixty to a second: a note at tick 4 held for 14 ticks sounds from 0.067 to 0.3 seconds, so ask at 0.15. The reading is of the 0.04 seconds after the time you give, and only notes from 50 to 4000 Hz are read.

a4 is 440 Hz and each octave doubles it. Octave 4 runs c 261.6, d 293.7, e 329.6, f 349.2, g 392.0, a 440.0, b 493.9, and a sharp is 6% above its letter. Within 2% of the note you wrote is in tune. A reading of double or half means the octave is wrong.

## The project file

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
    "coin": {
      "def": {"voices": [{"inst": "chime", "notes": [[0, "b5", 4], [4, "e6", 14]]}]},
      "check": [{"at": 0.02, "note": "b5"}, {"at": 0.15, "note": "e6"}]
    }
  }
}
```

- The file has four tables, all optional: `"instruments"`, `"sounds"`, `"songs"` and `"chords"`. A key that is not known is refused at any depth, with the place it sits at: `error: instruments.chime: unknown key 'volume'`.
- Time inside a sound is in ticks. A tick is a sixtieth of a second.
- A note is `[tick, note, ticks]`: when it starts, what it is, how long it is held. A fourth number is its velocity, 1 when left out.
- The note is a name, `"c4"` for middle C, `"f#5"`, `"eb3"`, or a number, where 60 is middle C and 69 the a at 440 Hz.
- A voice is one instrument and its notes. `"inst"` names an instrument of the file, or is an instrument written out in place. Voices can overlap, and so can the notes of one voice.
- A sound has one of four forms: `"def"` is one sound, `"defs"` a list of variants of it, `"parts"` several sounds on a timeline, and `"held"` a sound that goes on for as long as something holds it. `references/sounds.md` has each.
- `"db"` sets the level of a sound against the others, 0 when left out. Every sound is rendered at one fixed gain, has its loudest sample brought to -3 dB, and is then moved by its `"db"`. The `"gain"` a def can carry is set aside.
- `"check"` lists notes to read back. `check` reads the pitch at each time and reports a note that is more than 35 cents from the one written. Give marks to every `"def"` that has a tune.
- `"max_seconds"`, `"max_bright"` and `"max_dc"` give the entry limits of its own. `"meta"` is a dict of yours, carried through to whoever plays the sounds.

A setting that moves in time is a step table, read once a tick from the start of the note:

- a number is the same at every tick
- a list gives one value a tick, and its last value holds from there on
- `{"v": [0, -1], "loop": 1}` goes round again from index 1 each time it reaches its end

## Instruments

| Key | Value | Left out | Does |
|---|---|---|---|
| `wave` | `"pulse"`, `"saw"`, `"tri"`, `"noise"`, `"fm"` | `"pulse"` | The oscillator. `tri` is the darkest, `saw` and a narrow pulse the brightest. Noise has no note to read: its note sets how high it hisses |
| `vol` | step table, 0 to 15 | 15 | The level at each tick, which is the shape of the sound in time |
| `duty` | step table, 0 to 3 | 2 | The width of a pulse: 12.5%, 25%, 50%, 75%. Narrower is thinner and brighter, and 3 measures as 1 does |
| `arp` | step table, semitones | 0 | Added to the note at each tick: `[0, 4, 7]` steps through a chord |
| `pitch` | step table, semitones | 0 | Added up tick after tick, so the note keeps moving: a sweep. The last value goes on being added, so end the list on 0 to stop |
| `slide` | semitones a second | 0 | A steady glide for the whole note |
| `vib` | `{"depth": semitones, "rate": Hz, "delay": ticks}` | none | Vibrato. Only the depth is needed: the rate is 6 and the delay 0 without one |
| `lp` | step table, Hz | none | A low-pass at that cutoff, 12 dB an octave. The first thing to reach for when a sound is too bright |
| `res` | 0 to 0.98 | none | A resonant peak at the cutoff. Does nothing without `lp` |
| `release` | ticks | 2 | How long a note fades for once it is let go |
| `gain` | number | 1 | The level of every note of the instrument |
| `key_scale` | dB an octave | 0 | Added to the level for each octave above c4 and taken off for each below. A negative one holds high notes back |
| `short` | true or false | false | Noise only: a short register, a buzz with a pitch to it |
| `gate` | 0 to 1 | 0.9 | Songs only: the share of its rows a note is held for |
| `alg`, `fb`, `ops` | | | The FM voice: `references/sounds.md` |

## Writing a sound that works

- The `"vol"` table is the shape. One that opens high and falls is a hit or a pluck, and one that rises first has a soft start. Let it fall to 1 or 2 before the note ends, and give it a `"release"` of a few ticks.
- Keep it short. The recipes run 0.1 seconds for a blip, about 0.2 for a jump or a laser, 0.4 for a coin and 0.65 for an explosion.
- Move the pitch. A rise reads as a gain or a jump, a fall as a shot or a loss. `"pitch"` moves it inside one note, and a run of notes moves it in steps.
- Bring a harsh sound down with `"lp"`. Lowering its level leaves `above_4k` where it was.
- Noise makes the drum hits and the blasts. On c4 it is a rumble and on c7 a hiss, and from c6 up it needs an `"lp"` to pass `check`.
- A sound heard many times a minute wants a lower `"db"` than one heard once, and variants, so that a player need not repeat it exactly.

## Reading `check`

`check` prints `{"findings": [...]}` and exits with 1 when there are any. A finding has a `"rule"`, a `"target"` and a `"message"` with the number that broke the rule. The target is the place in the file: `sounds.coin`, `sounds.blip[1]` for a second variant, `sounds.explosion.parts[0]`, `sounds.hum.loop`, `songs.tune`, `songs.tune.patterns.a`. A file that cannot be loaded or rendered gives no findings: the command prints `error:` with the place and exits with 2.

Sounds:

| Rule | It found | Usual fix |
|---|---|---|
| `peak` | Before it is levelled the sound reaches 0.7 of full scale, the knee, from where the output stage bends it | Lower the instrument's `"gain"` or the velocity of the notes, or have fewer notes sound at once. The def's `"gain"` does not count |
| `ends` | The first sample or the last is 0.01 or more: a click | Open the `"vol"` table low, `[5, 15, 12]`, so the note rises over a tick. It comes from several `tri` or `noise` voices starting on one tick |
| `dc` | The mean sample is 0.002 or more off zero | A sound played once has its offset taken off as it is rendered, so this is a held sound, for its start and loop together. Make the loop longer, and a whole number of the note's periods |
| `bright` | More than 0.3 of the energy is above 4 kHz | Add an `"lp"` or lower it, play noise on a lower note, or widen a narrow pulse |
| `length` | The sound is longer than its `"max_seconds"` | Shorten the notes, the release or the echo |
| `pitch` | A mark of `"check"` reads more than 35 cents from its note, or reads no note | A wrong octave, most often. Or the mark sits where `arp` or `pitch` has moved the note, where two notes sound, or past the end: move the mark |
| `join` | A held sound steps where its start runs into its loop, or where the loop comes round | Give `"join"` a few ticks, and `"start"` enough of them for `"vol"` to settle. `references/sounds.md` has the rest |

Songs:

| Rule | It found | Usual fix |
|---|---|---|
| `rows` | A channel of a pattern fills fewer rows than the longest one, so its notes fall off the beat. The song is not played until this is gone | Count the channel bar by bar. At four rows to a beat every bar comes to 16 |
| `join` | A stem, or a group of a layered song, steps by 0.02 or more from its last sample round to its first | A note stops dead on the last sample of the loop. End that channel's last note early, with a rest after it: `a8` becomes `a16 r16` |
| `level` | A mix peaks at -1 dBFS or over, or has samples at full scale | Lower the song's `"gain"`, or the `"gain"` of its loudest channels |
| `layers` | Groups and versions that do not fit: a channel in no group or in two, a group with no channels or in no version, a version that names a group twice, two groups that are in exactly the same versions | Put every channel in one group, and every group in a set of versions no other group has |
| `chords` | A note that starts on a beat is no tone of the chord written for its bar, or a pattern has more or fewer chords than bars | Change the note, the chord or the list. A note off the beat is not held to the chord |

A finding can be meant: a hi-hat is bright. Give that entry a `"max_bright"` or a `"max_dc"` of its own, and say so when you hand the work over. The other rules have no limit to move, so fix what they find.

## Songs

A song is channels, patterns and an order. A channel plays one instrument, a pattern holds a line of text for each channel, and the order plays the patterns end to end as one loop. `references/songs.md` has a whole song to start from, the text language, chords, layers, the intro and what `build` writes. While you write one:

- A row is `"row_ticks"` ticks. At four rows to a beat the tempo is `900 / row_ticks` beats a minute: 6 gives 150 and 7.5 gives 120. A bar is 16 rows.
- Every channel of a pattern must fill the same number of rows. A length of 4 is 4 rows, 8 is 2, 16 is 1, and a dot adds half. Write `|` between bars and count each bar to 16.
- Song text spells a sharp `+` or `#` and a flat `-`: `f+`, `b-`. A sound's notes spell them `"f#4"` and `"bb4"`.
- To read the notes of one channel, render the stems and measure that channel's file. In the mix several notes sound at once and the reading is of none of them. Row `r` of the song starts `r * row_ticks / 60` seconds in.

```
python -m rowsynth render audio.json tune --stems --out out/stems
python -m rowsynth measure out/stems/bass.wav --at 0.05
python -m rowsynth build audio.json --out out/music
```

`--stems` writes each channel as a loop of its own, `CHANNEL.wav`. Without it `render` writes the song played once, with half a second after its last row. `build` writes the loops a game loads, as WAV or with `--format opus` as Opus, and `music.json`, which names them.

## Changing a project that exists

To change one value in a project file that is there already, use `set`. Do not write the file out again by hand, and never from a copy you read earlier.

```
python -m rowsynth set audio.json songs.tune.channels.bass.gain 0.7
python -m rowsynth set audio.json songs.tune.patterns.b.lead "o5 l4 a >c f c< | b >d g d< | >e2 c4< a4 | a2 r2"
python -m rowsynth set audio.json sounds.coin.db -3
python -m rowsynth set audio.json songs.tune.channels.lead.echo --delete
```

The place is spelt as the target of a finding is: names joined by dots, and `[n]` for an item of a list, counted from 0. The value is read as JSON when it is JSON and as text when it is not. `--text` makes it text whatever it looks like, and `--from PATH` reads it from a file, or from standard input with `-`, which keeps a long text or a whole sound out of the shell's quoting. A key that is not there yet is added, and so is the next item of a list. The command prints `{"file", "where", "was", "now"}`: `was` is what your change replaced.

Why `set`, and not the file written again:

- It reads the file, makes the change and writes it in one run, so what the person or another program wrote a moment before is kept. A copy you read a minute ago may be stale.
- The project is checked before anything is written, as loading checks it. A change that fails is refused with `error:` and exit code 2, and the file is as it was.
- The file is written whole or not at all. The text goes to a temporary file that takes the old one's place in one move, so nothing that reads the file gets half of it.
- It writes one tidy layout, with each channel of each pattern on a line of its own. One value changed is one line of a diff, which a person can read.

`set` checks what loading checks: the keys, the kinds of the values, and that every name points at something. It does not read song text and it runs none of the rules, so a channel that is a bar short is written without a word. Run `check` after every `set` that touches notes.

Write a whole file when it is new. For anything else, make the changes one `set` at a time.

The person may have the file open in the window, `python -m rowsynth gui audio.json`, where they place notes with the mouse and hear them. Then `set` is the way to change the file. The window sees your change within a quarter of a second, shows it, names it in its status line, and keeps it as a step: the person takes it back with Ctrl+Z if they do not want it. What they change is written to the file as they change it, so the file moves under you. Never edit a file the window has open in a way that leaves it half written, even for a moment, and never write it whole from a copy of your own: a file that does not load stops the window from saving the person's changes until it is whole again, and a whole file written from a copy puts back everything they changed since you read it. `references/window.md` has what the person sees, what to tell them to look at and listen to after a change, and how to get a picture of the window without opening one.

## Commands

Every command prints JSON. The usage lines are each command's own: type `python -m rowsynth` where they say `rowsynth`. After the command, its flags and its other words can come in any order.

```
rowsynth info [-h] FILE
rowsynth check [-h] [--rate N] FILE
rowsynth render [-h] --out PATH [--rate N] [--variant N] [--part {start,loop,end}] [--stems] FILE NAME
rowsynth measure [-h] WAV [--at SECONDS]
rowsynth measure [-h] FILE NAME [--rate N] [--variant N] [--part {start,loop,end}] [--at SECONDS]
rowsynth page [-h] --out DIR [--rate N] [--format {wav,mp3,opus}] FILE
rowsynth build [-h] --out DIR [--rate N] [--format {wav,opus}] FILE [NAME ...]
rowsynth set [-h] [--text] [--from PATH] [--delete] FILE WHERE [VALUE]
rowsynth gui [-h] [--scale N] [--frames N] [--shot PATH] [FILE]
```

| Command | Use |
|---|---|
| `info` | What the file holds by name, with the rows and the seconds of each song |
| `check` | The findings; exit code 1 when there are any |
| `render` | A sound as a mono WAV or a song as a stereo one, with its figures. `--variant` picks a variant counted from 0, and `--part` a part of a held sound |
| `measure` | The figures of a WAV file, or of a sound or a song of a project file, with the brightness and the pitch. It reads no MP3 and no Opus |
| `page` | The listen page, `index.html`, with its audio files in a folder `audio` beside it |
| `build` | The songs named, or every song, as loops in a folder with `music.json` |
| `set` | One value of a project file changed and the rest kept: checked as loading checks it, then written whole or not at all |
| `gui` | The window on a project file, which is for the person. Run it yourself only with `--frames` and `--shot`, for a picture: `references/window.md` |

`check`, `render` and `measure` work at 44100 samples a second unless `--rate` says otherwise, and `page` and `build` at 48000.

WAV needs nothing but the package. MP3 and Opus are made by ffmpeg, and without it on the path the command stops with `error: ffmpeg is needed to write mp3, and none is found on the path` before a file is written. Opus takes a rate of 8000, 12000, 16000, 24000 or 48000 and no other.
