# rowsynth from Python

The command line covers the loop. Write a script when the data is easier to compute than to type, such as one sound on every note of a scale, or when you want many candidates measured side by side.

A script loads a project or builds one as plain data, changes its tables, and saves the file. After that the loop is the same: render, measure, `check`, fix the script, run it again.

If the package is not installed, run the script with the repo's root folder in `PYTHONPATH`, or `import rowsynth` fails.

```python
import rowsynth
from rowsynth import checks
from rowsynth.measure import brightness, pitch_at

project = rowsynth.Project.load("audio.json")

# One tick on five notes, as the variants of a sound.
project.instruments["tick"] = {"wave": "pulse", "duty": 2, "vol": [10, 7, 3], "release": 1}
project.sounds["steps"] = {
    "defs": [{"voices": [{"inst": "tick", "notes": [[0, note, 3]]}]} for note in ("c5", "d5", "e5", "g5", "a5")]
}
project = rowsynth.Project.from_dict(project.to_dict())    # reads it all again, as loading does

for number, samples in enumerate(rowsynth.render_entry(project.sound("steps"))):
    print(number, rowsynth.measure(samples)["rms_db"], brightness(samples)["above_4k"], pitch_at(samples, 0.01))

for finding in checks.run(project):
    print(finding.rule, finding.target, finding.message)

project.save("out/steps.json")
```

## The project

The four tables are plain dicts, and what you put in them is not looked at until the project is read again. `Project.from_dict(project.to_dict())` does that, and refuses what is wrong with the place it sits at.

| Call | Does |
|---|---|
| `Project.load(path)`, `.save(path)` | Reads and writes the project file |
| `Project.from_dict(data)`, `.to_dict()` | The same, from and to plain data |
| `.instruments`, `.sounds`, `.songs`, `.chords` | The four tables, to read and to change |
| `.sound(name)` | A copy of one sound with every instrument it names written out: what `render_entry` takes |
| `.song(name)` | A copy of one song with the file's instruments under its own: what the three song renderers take |

## Rendering

Samples are float32 NumPy arrays, with full scale at 1. The renderers and `measure` work at 44100 samples a second unless you pass another rate.

| Call | Gives |
|---|---|
| `render_sound(defn, rate, gain=None)` | One def as samples, at its own `"gain"`, which is 0.5 without one. Its voices must hold instruments written out, as `.sound(name)["def"]` has them |
| `render_entry(entry, rate)` | The buffers of an entry at the level the command line gives them: one for a def or for parts, one for each variant, and the start, the loop and the end of a held sound |
| `render_bank(sounds, rate)` | `{name: {"buffers", "meta", "held"}}` for `{name: entry}`, each entry as `.sound(name)` gives it |
| `render_song(song, only=None, dry=False, rate=44100)` | `{"left", "right", "rows", "seconds"}`: the song played once. `only` lists the channels to play, and `dry=True` leaves out the echoes of the channels |
| `render_stems(song, rate)` | `{"stems": [{"name", "pan", "samples"}], "rows", "loop_samples", "seconds"}`: a mono loop for each channel |
| `render_loop(song, rate)` | `{"left", "right", "rows", "loop_samples", "seconds"}`: the stems mixed into one stereo loop |

## Measuring, checking and files

| Call | Does |
|---|---|
| `measure(samples, rate)` | The figures `render` prints, as a dict |
| `brightness(samples, rate=44100)` | `{"centroid_hz", "above_4k"}`. From `rowsynth.measure` |
| `pitch_at(samples, t, rate=44100)` | The note sounding at `t` seconds in Hz, or 0.0. From `rowsynth.measure` |
| `checks.run(project, rate)` | The findings as a list, each with `.rule`, `.target`, `.message` and `.to_dict()` |
| `save_wav(path, samples, rate)` | Writes a 16-bit WAV, and the folders above it when they are missing. One array is mono, and `[left, right]` is stereo |
| `load_wav(path)` | `(channels, rate)`: one array or two, and the rate of the file |
| `build(project, out_dir, rate=48000, songs=None)` | What the `build` command does. Gives the manifest |
| `write_page(project, out_dir, rate=48000)` | What the `page` command does. Gives `{"page": path, "files": [paths]}` |

`rowsynth.measure` is the function, so the two others in its module are reached with `from rowsynth.measure import brightness, pitch_at`.

To make the files without the command line:

```python
song = project.song("tune")
mix = rowsynth.render_song(song)
rowsynth.save_wav("out/tune.wav", [mix["left"], mix["right"]])
for stem in rowsynth.render_stems(song)["stems"]:
    rowsynth.save_wav(f"out/stems/{stem['name']}.wav", stem["samples"])
rowsynth.build(project, "out/music", songs=["tune"])
print(rowsynth.write_page(project, "out/page")["page"])
```

Bad input raises `rowsynth.RowSynthError` with a message that names the place at fault.
