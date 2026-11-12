<!-- claude-projects-attribution: {"github_login":"mahimairaja"} -->
_Requested by **Mahimai** · [project thread](https://claude.ai/code/project/chan_01ShZjNBhSgQgUj4vRoEC8CF?thread=cmsg_01ShZjNBhSgQgUj4vRoEC8CFQbZ8KTyrm7guRt88wVRdCQ)_

Part of #5351. Stacked on #7731 (the first commit here is #7731's head); I'll rebase once it merges.

**Before:** `livekit-plugins-hf` ran chat models only, so an open STT model meant a separate server.

**After:** `hf.STT(model=...)` runs any `automatic-speech-recognition` model from the Hub in-process: Whisper and its fine-tunes, Moonshine, wav2vec2.

```python
session = AgentSession(stt=hf.STT(model="openai/whisper-large-v3-turbo", language="en"), vad=silero.VAD.load(), ...)
```

### How

- Batch STT (`streaming=False`): the session VAD cuts utterances and `StreamAdapter` hands each one to the model in one pass.
- Audio is merged, downmixed and resampled with `rtc.AudioResampler` to the feature extractor's rate. No torchaudio.
- Same loading model as `hf.LLM`: `local_files_only=True` by default, weights load in a thread on `prewarm()` or via `load()`, cached per process by (model, revision, device, dtype) and shared across instances, one pass at a time per model under a thread lock, then a warm-up pass.
- `language` goes to Whisper's `generate`. Whisper fine-tunes that add languages on their own token ids take `language_token_ids`, e.g. [SunflowerASR](https://huggingface.co/Sunbird/SunflowerASR-51-african-languages) (51 African languages): the token is added to a copy of `generation_config.lang_to_id`, never the shared one.
- A small `_cache.get_or_load` holds the per-process cache for STT (and TTS next). `hf.LLM` keeps its own for now so #7731 doesn't change.

### Testing

- `tests/test_plugin_hf_stt.py`: 8 unit tests with a fake pipeline (resampling/downmix, language and task, fine-tune token ids, empty buffer, error mapping, prewarm loads once). Ruff and strict mypy clean.
- On Modal (L4 and H100), real weights, speech made by `hf.TTS`:
  - `openai/whisper-large-v3-turbo`: English WER 0.0 (0.165 s for a 5 s clip on H100, 0.26 s on L4); Swahili WER 0.33.
  - `Sunbird/SunflowerASR-51-african-languages` with `language_token_ids`: Swahili WER 0.0, English 0.0, Luganda 0.17 (one word), 0.5-0.6 s per clip on H100.
