<!-- claude-projects-attribution: {"github_login":"mahimairaja"} -->
_Requested by **Mahimai** · [project thread](https://claude.ai/code/project/chan_01ShZjNBhSgQgUj4vRoEC8CF?thread=cmsg_01ShZjNBhSgQgUj4vRoEC8CFQbZ8KTyrm7guRt88wVRdCQ)_

Part of #5351. Stacked on #<TTS PR>.

In-process inference suits development and single GPU boxes, but in production it ties model memory to every job process. `hf.STT` and `hf.TTS` now take:

- `provider=`: any [Hugging Face Inference Provider](https://huggingface.co/docs/inference-providers) serving the task (`auto`, `hf-inference`, `fal-ai`, `deepinfra`, ...).
- `base_url=`: a dedicated endpoint, such as an HF Inference Endpoint.

```python
stt = hf.STT(model="openai/whisper-large-v3-turbo", provider="hf-inference")
tts = hf.TTS(model="hexgrad/Kokoro-82M", provider="deepinfra", voice="af_heart")
```

Nothing loads locally in that mode and torch isn't touched. Calls go through `huggingface_hub.AsyncInferenceClient` with `HF_TOKEN`. HTTP errors become `APIStatusError` (5xx and 429 retry, other 4xx don't), timeouts `APITimeoutError`. Provider audio is an encoded clip; its container is sniffed and the emitter decodes it.

### Testing

- `tests/test_plugin_hf_hosted.py`: unit tests with a fake client.
- Live on real providers (Modal CPU container, `HF_TOKEN`): `hf.TTS` with Kokoro-82M, then `hf.STT` with Whisper transcribing that audio.

  | Step | Provider | Time | Result |
  |---|---|---|---|
  | TTS, Kokoro-82M | deepinfra | 10.9 s for 4.1 s of audio | 24 kHz WAV |
  | TTS, Kokoro-82M | fal-ai | 161 s for 4.1 s of audio (queued) | 24 kHz |
  | STT, whisper-large-v3-turbo | hf-inference | 0.81-0.93 s | WER 0.0 |
  | STT, whisper-large-v3 | fal-ai | 0.89-1.06 s | WER 0.0 |

  STT works for live calls. Provider TTS is too slow to speak live, so the README calls it a fallback and points to a dedicated endpoint or local streaming.
