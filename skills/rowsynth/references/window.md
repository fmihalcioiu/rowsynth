# The window: a file the person has open

`python -m rowsynth gui audio.json` opens a window on a project file. It is for the person: they place notes with the mouse, turn knobs, and hear what they do. You do not work in it. You change the file with `set`, and the window shows what you changed.

## What it is

| Part | Holds |
|---|---|
| Title row | The name of the file, and buttons to minimise, maximise and close. The window has it in place of the system's frame |
| Top bar | `FILE`, undo and redo, play, stop and loop, `PATTERN` to play only the pattern that is shown, `DRY` to play without echoes, the time, a meter, `LIST` and `INSPECTOR` to show or hide the two side panels, and at the right end the count of findings: a green tick with 0, or a red sign with how many |
| The list on the left | `SONGS`, `SOUNDS` and `INSTRUMENTS` by name, with a `+` after the last of each. A click picks one, and Space plays what is picked |
| The middle | What is picked. The three editors are laid out alike: a heading, one bar with the knobs, the main area, and a foot. A song has a strip of blocks, one for each pattern in its order, and `NOTES`, `MIXER` or `LAYERS` under it. A sound is its notes beside a keyboard on a clock of seconds, over its wave. An instrument is its tables as bars beside their names, over a keyboard to try it on |
| The inspector on the right | `MEASURED`: the figures of what is picked, as `LENGTH`, `PEAK`, `LOUDNESS`, `OFFSET`, `CLIPPED`, `START`, `END`, `HIGHS`, `BRIGHTNESS` and for a song or a held sound `LOOP JOIN`, with one that breaks a rule in red. `FOUND`: the findings of `check` in words |
| Status line | What the window has to say, and for whatever the mouse is on what a press, a right press and the wheel do there. At its right end is what is picked, as in `SONG tune` |

A window too narrow for all three columns has no inspector, and a narrower one has no list either. If the person cannot find one of them, `LIST` and `INSPECTOR` in the top bar bring it back.

Every change the person makes is written to the file as they make it, in the layout `set` writes. There is no Save. Each change is a step, and Ctrl+Z takes the last one back.

## How the person starts it

The window needs pygame, which the package does not install by itself: `pip install pygame`, or `pip install ".[gui]"` in a clone of the repo. Without it the command prints `error: the GUI needs pygame, which is not installed: pip install rowsynth[gui]`.

```
python -m rowsynth gui audio.json
```

With no file it opens on a first screen, where a project is made or opened. The command does not come back until the window is closed, and it adds the file to the person's list of files opened last. So it is the person who runs it, in a terminal of their own. Do not run it for them without `--frames`.

## What the person sees when you change the file

The window looks at its file four times a second.

- What you changed is shown as the file has it now: the notes, the knobs, the list on the left.
- The status line says `changed outside:` and the names of the things that differ, three at the most, as in `changed outside: tune`. A thing is a whole song, sound, instrument or chord.
- A song that is playing goes on from where it is, with your change in it. Nothing starts to play by itself.
- What is picked is rendered again and the checks run again, so `MEASURED`, `FOUND` and the count in the top bar are those of the file as you left it.
- Your change is a step. Ctrl+Z in the window takes it back, in the file too, and with the mouse on the undo arrow the status line reads `undo: changed outside: tune`. What one look finds is one step, so two changes that land between two looks are taken back together.
- If what was picked is gone from the file, the first thing in the list is picked.

When the person changes the same place in the moment your change lands, theirs is the one that stands. Their status line says so, and their Ctrl+Z puts yours back. Changes to different places are both kept.

A file that does not load, half written or no JSON, changes nothing in the window. The status line says what is wrong with it, and what the person changes meanwhile is refused. `set` never leaves the file so.

Their changes reach the file as they make them. Read a place again before you change it, or go by the `was` that `set` prints.

## What to tell the person after a change

Say what you changed and where it is, in the words the window uses:

- A song: its name in the list on the left, then the block of the pattern in the strip along the top, then the channel in the list under `NOTES`. Space plays the song and stops it. With `FOLLOW` on, as it is when the window opens, the pattern that is shown goes along with the playing.
- One channel among the others: `MIXER`, and `S` on that channel to hear it alone, or `M` to hear the song without it.
- A sound or an instrument: a click on its name in the list shows it, and Space plays it. Nothing plays by itself. A song or a sound that plays an instrument you changed has changed with it.
- The volume, the pan or the echo of a channel: `VOLUME`, `PAN` and `ECHO` in the bar over its notes, and its fader in `MIXER`.
- A knob of an instrument or of a sound: the window names them `VOLUME`, `RELEASE`, `GATE`, `RESONANCE`, `KEY TRACKING`, and the tables `VOLUME SHAPE`, `PULSE WIDTH`, `ARPEGGIO`, `PITCH` and `CUTOFF`, for `gain`, `release`, `gate`, `res`, `key_scale`, `vol`, `duty`, `arp`, `pitch` and `lp` of the file.
- The findings: the count at the right end of the top bar, and `FOUND` in the inspector, where a click on a finding goes to the thing it is in.
- If they do not want the change: Ctrl+Z, with the window in front.

What they say about the sound outranks its numbers, here as on the listen page.

## A picture of the window without opening one

`--frames N` closes the window by itself after N frames, and `--shot PATH` writes the last frame as a PNG. With the two SDL drivers set to `dummy` in the environment, no window opens and no sound plays. Set them every time: without them a window opens on the person's screen.

In a POSIX shell:

```
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m rowsynth gui audio.json --frames 300 --shot out/window.png
```

In PowerShell:

```
$env:SDL_VIDEODRIVER = "dummy"; $env:SDL_AUDIODRIVER = "dummy"
python -m rowsynth gui audio.json --frames 300 --shot out/window.png
```

```
{
  "file": "audio.json",
  "frames": 300,
  "shot": "out/window.png"
}
```

- The picture is the window as it opens: the first song of the file, its first pattern, and the first tuned channel with notes in that pattern. A file with no song shows its first sound. Nothing else can be picked from the command line.
- Leave `--scale` out. The picture is then the whole layout, 960 by 540 pixels. With `--scale 2` on the dummy screen it came out 512 by 336, with parts of the layout missing.
- The render and the checks run beside the window. Until they are through, `MEASURED` and `FOUND` show dots. How many frames that takes goes by the length of the song and by how busy the machine is, and a count of frames is no count of seconds. Start with 300, or 600 for a song as long as `piece` in `examples/demo.json`, and if the picture has dots, run it again with more.
- A run with `--frames` reads no list of files opened last and adds nothing to it.
- Read the PNG to see what the person would see. It is a way to check what you describe to them. It is not a way to hear anything.
