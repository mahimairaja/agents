<!-- claude-projects-attribution: {"github_login":"mahimairaja"} -->
_Requested by **Mahimai** · [project thread](https://claude.ai/code/project/chan_01ShZjNBhSgQgUj4vRoEC8CF?thread=cmsg_01ShZjNBhSgQgUj4vRoEC8CFQbZ8KTyrm7guRt88wVRdCQ)_

Part of #5351. Stacked on #<STT PR>.

**After:** `hf.TTS(model=...)` runs two model families in-process, picked from the config:

- **Orpheus** (Llama models that speak in SNAC audio codes): `canopylabs/orpheus-3b-0.1-ft` and its fine-tunes, such as [Sunbird's multilingual model](https://huggingface.co/Sunbird/orpheus-3b-tts-multilingual) (20 African languages). Audio is streamed while the model generates.
- **Text to waveform** (`AutoModelForTextToWaveform`): VITS/MMS voices for about 1,100 languages.

```python
tts = hf.TTS(model="Sunbird/orpheus-3b-tts-multilingual", voice="salt_lug_0001")
```

### How

- Orpheus tokens come from a `BaseStreamer` on the generate thread. `_SnacStreamDecoder` groups them into 7-code frames, drops codes in the wrong slot, and decodes windows of `chunk_frames` (default 4, ~340 ms) with context frames on both sides, keeping only the middle, so chunk edges don't click.
- Cancelling the stream stops `generate()` with a `StoppingCriteria`, like `hf.LLM`.
- Output is resampled to `sample_rate` (24 kHz default) with `rtc.AudioResampler`.
- SNAC is an optional extra (`livekit-plugins-hf[orpheus]`). `download-files` fetches it with any Orpheus model, and falls back to `.bin` weights for repos without safetensors.

### Testing

- `tests/test_plugin_hf_tts.py`: 7 unit tests (every frame emitted exactly once across windows, stray tokens and misplaced codes ignored, context kept out of output, waveform resampling, family detection, SNAC download).
- On Modal (L4 and H100), real weights. Whisper large-v3-turbo transcribes the Orpheus output word for word.
  - MMS (VITS) English: real-time factor 0.034 on H100.
  - Streamed SNAC output is 24-29 dB SNR against one full decode of the same codes. A context sweep set the defaults (left 4, right 2, chunk 4): a second right-context frame gains 2-5 dB, more left context or bigger chunks under 0.5 dB.
- Performance note: real-time Orpheus needs ~83 tokens/s, and plain transformers `generate` doesn't reach it:

  | Model | GPU | First audio | Real-time factor |
  |---|---|---|---|
  | canopylabs/orpheus-3b-0.1-ft | L4 | 2.1 s | 4.8 |
  | canopylabs/orpheus-3b-0.1-ft | H100 | 0.97 s | 2.1 |
  | Sunbird/orpheus-3b-tts-multilingual (lug) | H100 | 1.14 s | 2.5 |

  So in-process Orpheus suits development; for production the README points to vLLM behind a dedicated endpoint (next PR adds `base_url=`).
