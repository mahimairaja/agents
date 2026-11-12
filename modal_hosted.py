"""Live check of hf.STT / hf.TTS hosted mode (HF Inference Providers). CPU only.

    modal run modal_hosted.py
"""

from __future__ import annotations

import os

import modal

SHA = os.environ.get("HF_SPEECH_SHA", "d257216a84ed01a32a3421cbc0092646b462105e")
HF_SECRET = os.environ.get("HF_SECRET", "huggingface")
REPO = "git+https://github.com/mahimairaja/agents"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"livekit-agents @ {REPO}@{SHA}#subdirectory=livekit-agents",
        f"livekit-plugins-hf @ {REPO}@{SHA}#subdirectory=livekit-plugins/livekit-plugins-hf",
        "jiwer==3.1.0",
    )
)
app = modal.App("hf-speech-hosted", image=image)
TEXT = "Hello, thanks for calling the clinic. How can I help you today?"


@app.function(timeout=900, secrets=[modal.Secret.from_name(HF_SECRET)])
def verify() -> dict:
    import asyncio
    import time
    import traceback

    import jiwer

    from livekit.plugins import hf

    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    results: dict = {"tts": [], "stt": []}

    async def main() -> None:
        clips = {}
        for provider in ("fal-ai", "deepinfra"):
            try:
                tts = hf.TTS(model="hexgrad/Kokoro-82M", provider=provider, voice="af_heart")
                tts.load()  # must not load anything
                t0 = time.perf_counter()
                frames = [a.frame async for a in tts.synthesize(TEXT)]
                took = time.perf_counter() - t0
                clips[provider] = frames
                results["tts"].append({"provider": provider, "total_s": round(took, 2),
                                       "audio_s": round(sum(f.duration for f in frames), 2),
                                       "sample_rate": frames[0].sample_rate})
                await tts.aclose()
            except Exception as e:
                results["tts"].append({"provider": provider, "error": f"{type(e).__name__}: {e}"[:400],
                                       "trace": traceback.format_exc()[-1500:]})
        from livekit import rtc

        for provider, model in (("hf-inference", "openai/whisper-large-v3-turbo"), ("fal-ai", "openai/whisper-large-v3")):
            for source, frames in clips.items():
                try:
                    stt = hf.STT(model=model, provider=provider, language="en")
                    stt.load()
                    t0 = time.perf_counter()
                    ev = await stt.recognize(rtc.combine_audio_frames(frames))
                    hyp = ev.alternatives[0].text
                    results["stt"].append({"provider": provider, "audio_from": source,
                                           "latency_s": round(time.perf_counter() - t0, 2),
                                           "wer": round(jiwer.wer(norm(TEXT), norm(hyp)), 3), "hyp": hyp})
                    await stt.aclose()
                except Exception as e:
                    results["stt"].append({"provider": provider, "audio_from": source,
                                           "error": f"{type(e).__name__}: {e}"[:400],
                                           "trace": traceback.format_exc()[-1500:]})

    asyncio.run(main())
    return results


@app.local_entrypoint()
def run() -> None:
    import json

    print(json.dumps(verify.remote(), indent=2))
