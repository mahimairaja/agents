"""End-to-end check of livekit-plugins-hf STT and TTS on a Modal GPU.

Usage (from a machine with the modal CLI logged in):
    HF_SPEECH_SHA=<commit on mahimairaja/agents> HF_SECRET=<modal secret with HF_TOKEN> \
        modal run modal_verify.py
HF_SECRET is optional; without it the gated models (canopylabs Orpheus, SunflowerASR) are skipped.
"""

from __future__ import annotations

import os

import modal

SHA = os.environ.get("HF_SPEECH_SHA", "feat/hf-tts-plugin")
HF_SECRET = os.environ.get("HF_SECRET")
REPO = "git+https://github.com/mahimairaja/agents"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"livekit-agents @ {REPO}@{SHA}#subdirectory=livekit-agents",
        f"livekit-plugins-hf[orpheus] @ {REPO}@{SHA}#subdirectory=livekit-plugins/livekit-plugins-hf",
        "jiwer==3.1.0",
        "soundfile==0.13.1",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)
volume = modal.Volume.from_name("hf-speech-cache", create_if_missing=True)
app = modal.App("hf-speech-verify", image=image)

TEXTS = {
    "eng": "Hello, thanks for calling the clinic. How can I help you today?",
    "lug": "Oli otya? Nsanyuse nnyo okukulaba leero.",
    "swa": "Habari yako? Leo hali ya hewa ni nzuri sana.",
}
SUNFLOWER_TOKENS = {"eng": 50259, "swa": 50318, "lug": 50332}


@app.function(
    gpu="L4",
    timeout=3600,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name(HF_SECRET)] if HF_SECRET else [],
)
def verify() -> dict:
    import asyncio
    import json
    import time

    import jiwer
    import numpy as np
    import soundfile as sf
    import torch

    from livekit.plugins import hf
    from livekit.plugins.hf import tts as hf_tts

    has_token = bool(os.environ.get("HF_TOKEN"))
    results: dict = {"gpu": torch.cuda.get_device_name(0), "has_token": has_token, "tts": [], "stt": []}
    os.makedirs("/cache/out", exist_ok=True)

    # record every code the streaming decoder sees, to compare against one full decode
    recorded: list[list[int]] = []

    class _Recording(hf_tts._SnacStreamDecoder):
        def push(self, token_id):  # type: ignore[no-untyped-def]
            window = super().push(token_id)
            recorded[-1][:] = self._codes
            return window

    hf_tts._SnacStreamDecoder = _Recording  # type: ignore[misc]

    def download(model: str) -> float:
        t = time.perf_counter()
        hf.download_model(model)
        return time.perf_counter() - t

    async def synth(name: str, tts: "hf.TTS", text: str) -> np.ndarray:
        t = time.perf_counter()
        tts.load()
        load_s = time.perf_counter() - t
        out = {}
        for run in range(2):  # the first run warms up kernels
            recorded.append([])
            t0 = time.perf_counter()
            ttfb = None
            chunks = []
            async for audio in tts.synthesize(text):
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                chunks.append(np.frombuffer(audio.frame.data, dtype=np.int16))
            total = time.perf_counter() - t0
            pcm = np.concatenate(chunks)
            out = {"ttfb_s": round(ttfb or 0, 3), "total_s": round(total, 3)}
        dur = len(pcm) / tts.sample_rate
        entry = {"name": name, "model": tts.model, "load_s": round(load_s, 1), **out,
                 "audio_s": round(dur, 2), "rtf": round(out["total_s"] / max(dur, 1e-6), 3),
                 "peak": int(np.abs(pcm).max())}
        if recorded[-1]:
            loaded = tts._get_model()
            codes = recorded[-1]
            n = len(codes) // 7
            full = hf_tts._decode_snac(loaded, hf_tts._DecodeWindow(codes=codes[: n * 7], start=0, end=n * 2048))
            streamed = pcm[: n * 2048].astype(np.float32) / 32767.0
            m = min(len(full), len(streamed))
            err = streamed[:m] - np.clip(full[:m], -1, 1)
            entry["stream_vs_full_snr_db"] = round(float(10 * np.log10(np.sum(full[:m] ** 2) / max(np.sum(err**2), 1e-12))), 1)
            entry["frames"] = n
        sf.write(f"/cache/out/{name}.wav", pcm, tts.sample_rate)
        results["tts"].append(entry)
        print(json.dumps(entry))
        return pcm

    async def recognize(name: str, stt: "hf.STT", pcm: np.ndarray, rate: int, ref: str, language: str | None) -> None:
        from livekit import rtc

        frame = rtc.AudioFrame(data=pcm.tobytes(), sample_rate=rate, num_channels=1, samples_per_channel=len(pcm))
        stt.load()
        await stt.recognize(frame, language=language) if language else await stt.recognize(frame)
        t0 = time.perf_counter()
        ev = await (stt.recognize(frame, language=language) if language else stt.recognize(frame))
        latency = time.perf_counter() - t0
        hyp = ev.alternatives[0].text
        norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        wer = jiwer.wer(norm(ref), norm(hyp))
        entry = {"name": name, "model": stt.model, "language": language, "latency_s": round(latency, 3),
                 "audio_s": round(len(pcm) / rate, 2), "wer": round(wer, 3), "ref": ref, "hyp": hyp}
        results["stt"].append(entry)
        print(json.dumps(entry))

    async def main() -> None:
        dl = {}
        models = ["facebook/mms-tts-eng", "Sunbird/orpheus-3b-tts-multilingual", "openai/whisper-large-v3-turbo"]
        if has_token:
            models += ["canopylabs/orpheus-3b-0.1-ft", "Sunbird/SunflowerASR-51-african-languages"]
        for m in models:
            try:
                dl[m] = round(download(m), 1)
            except Exception as e:  # gated model without access
                dl[m] = f"failed: {type(e).__name__}: {e}"[:300]
        results["download_s"] = dl
        volume.commit()
        ok = {m for m, v in dl.items() if not isinstance(v, str)}

        audio: dict[str, tuple[np.ndarray, int, str]] = {}
        mms = hf.TTS(model="facebook/mms-tts-eng")
        audio["mms_eng"] = (await synth("mms_eng", mms, TEXTS["eng"]), mms.sample_rate, TEXTS["eng"])
        sunbird = "Sunbird/orpheus-3b-tts-multilingual"
        for lang in ("lug", "swa"):
            tts = hf.TTS(model=sunbird, voice=f"salt_{lang}_0001")
            audio[f"sunbird_{lang}"] = (await synth(f"sunbird_{lang}", tts, TEXTS[lang]), tts.sample_rate, TEXTS[lang])
        if "canopylabs/orpheus-3b-0.1-ft" in ok:
            tts = hf.TTS()  # default model and voice
            audio["orpheus_eng"] = (await synth("orpheus_eng", tts, TEXTS["eng"]), tts.sample_rate, TEXTS["eng"])
            # a long multi-clause sentence, to see streaming hold up
            long_text = ("Your appointment with doctor Patel is confirmed for Thursday at three thirty in the "
                         "afternoon, and please bring your insurance card and a list of your current medications.")
            await synth("orpheus_eng_long", tts, long_text)

        whisper = hf.STT(model="openai/whisper-large-v3-turbo")
        for key in ("mms_eng", "orpheus_eng"):
            if key in audio:
                pcm, rate, ref = audio[key]
                await recognize(f"whisper_{key}", whisper, pcm, rate, ref, "en")
        pcm, rate, ref = audio["sunbird_swa"]
        await recognize("whisper_sunbird_swa", whisper, pcm, rate, ref, "sw")

        if "Sunbird/SunflowerASR-51-african-languages" in ok:
            sunflower = hf.STT(model="Sunbird/SunflowerASR-51-african-languages", language_token_ids=SUNFLOWER_TOKENS,
                               generate_kwargs={"num_beams": 1, "do_sample": False})
            for lang in ("lug", "swa"):
                pcm, rate, ref = audio[f"sunbird_{lang}"]
                await recognize(f"sunflower_sunbird_{lang}", sunflower, pcm, rate, ref, lang)
            if "orpheus_eng" in audio:
                pcm, rate, ref = audio["orpheus_eng"]
                await recognize("sunflower_orpheus_eng", sunflower, pcm, rate, ref, "eng")

        results["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)

    asyncio.run(main())
    volume.commit()
    with open("/cache/out/results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    return results


@app.local_entrypoint()
def run() -> None:
    import json

    print(json.dumps(verify.remote(), indent=2))
