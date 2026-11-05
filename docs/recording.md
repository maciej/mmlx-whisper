# Record evaluation audio

On an interactive macOS terminal, use the standalone recorder. `uv` installs its
dependencies in an isolated script environment; no model or server is required.
Keep prompt files and recordings outside the checkout. Create a JSON file such
as `/tmp/prompts.json` containing `["Your first sentence.", "Another sentence."]`,
or an object with an `utterances` list of objects containing `text` fields.

```sh
uv run tools/record.py --script /tmp/prompts.json --preview
uv run tools/record.py --list-devices
uv run tools/record.py --script /tmp/prompts.json
```

Hold **Space** while speaking the displayed prompt; release it to save and
advance. **R** returns to the previous prompt for another take, and **Q** finishes.
Allow the terminal microphone access when macOS asks. Use `--device INDEX` to
select an input from the device list.

Audio stays local as 16 kHz mono PCM WAV files. Each take is retained, and
`session.json` maps prompts to their latest takes, including audio warnings.
Output defaults to a new timestamped folder beside the prompt file under
`recordings/`; `--output-dir PATH` selects a new session directory instead.
Ctrl+C retains saved takes and marks an unfinished session as partial.
