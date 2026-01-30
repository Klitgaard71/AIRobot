#!/usr/bin/env python3
import os
import json
import base64
import asyncio
import argparse
import logging
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("realtime_voice")

# ----------------------------
# Audio config
# ----------------------------
SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = "int16"          # PCM16
CHUNK_FRAMES = 1024      # Lower latency; keep since your pipeline works

DEFAULT_MODEL = "gpt-realtime"
DEFAULT_WAKE_WORD = "atlas"
DEFAULT_VOICE = "echo"  # <-- change here or via --voice / env var

# ----------------------------
# Helpers
# ----------------------------
def list_audio_devices():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        in_ch = d.get("max_input_channels", 0)
        out_ch = d.get("max_output_channels", 0)
        log.info(f"[{i:2d}] {d['name']} | in:{in_ch} out:{out_ch}")


def read_text_file(path: str) -> str:
    script_dir = Path(__file__).resolve().parent
    p = Path(path)
    if not p.is_absolute():
        p = script_dir / p
    if not p.exists():
        raise SystemExit(f"System prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def contains_wake_word(transcript: str, wake_word: str) -> bool:
    # simple/robust: substring match; you can tighten later (word boundary)
    return normalize_text(wake_word) in normalize_text(transcript)


class AsyncAudioIO:
    def __init__(self, input_device=None, output_device=None):
        self.input_device = input_device
        self.output_device = output_device
        self._in_stream = None
        self._out_stream = None

    def start(self):
        if self._in_stream is None:
            self._in_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_FRAMES,
                device=self.input_device,
            )
            self._in_stream.start()

        if self._out_stream is None:
            self._out_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_FRAMES,
                device=self.output_device,
            )
            self._out_stream.start()

    def stop(self):
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._in_stream = None
        self._out_stream = None

    async def read_chunk_bytes(self) -> bytes:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, lambda: self._in_stream.read(CHUNK_FRAMES)[0])
        return data.tobytes()

    async def play_chunk_bytes(self, pcm_bytes: bytes):
        loop = asyncio.get_running_loop()
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        await loop.run_in_executor(None, self._out_stream.write, samples)


async def ws_connect(uri: str, headers: dict):
    """
    websockets changed parameter name from extra_headers -> additional_headers.
    Try new name first, fall back to old.
    """
    try:
        return await websockets.connect(uri, additional_headers=headers)  # websockets >= 13
    except TypeError:
        return await websockets.connect(uri, extra_headers=headers)       # older websockets


async def send_json(ws, obj):
    await ws.send(json.dumps(obj))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_REALTIME_MODEL", DEFAULT_MODEL))
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--input-device", type=int, default=None, help="sounddevice input device index")
    ap.add_argument("--output-device", type=int, default=None, help="sounddevice output device index")

    # System prompt as file (recommended) or inline fallback
    ap.add_argument("--system-prompt-file", default="homebot_system_prompt.txt")
    ap.add_argument("--system-prompt", default=None, help="Inline prompt override (optional)")

    # Wake word
    ap.add_argument("--wake-word", default=os.environ.get("OPENAI_WAKE_WORD", DEFAULT_WAKE_WORD))
    ap.add_argument("--wake-word-enabled", action="store_true", help="Require wake word to respond")

    # Voice (THIS is where you change it)
    ap.add_argument("--voice", default=os.environ.get("OPENAI_REALTIME_VOICE", DEFAULT_VOICE),
                    help="Voice name, e.g. alloy, verse, etc. (depends on account/model)")

    # Behavior: keep responses English unless user requests otherwise
    ap.add_argument("--english-only", action="store_true", default=True)

    args = ap.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY env var. Example: export OPENAI_API_KEY='sk-...'")

    ws_url = f"wss://api.openai.com/v1/realtime?model={args.model}"
    headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}

    # Load system prompt
    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        try:
            system_prompt = read_text_file(args.system_prompt_file)
        except FileNotFoundError as e:
            raise SystemExit(str(e))

    # Add English lock (simple + effective)
    if args.english_only:
        system_prompt = (
            system_prompt.strip()
            + "\n\n"
            + "Language rule: Speak English by default. Only switch language if the user explicitly asks."
        )

    audio = AsyncAudioIO(input_device=args.input_device, output_device=args.output_device)
    audio.start()

    play_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

    # Wake word state (no audio pipeline changes; we only gate response.create)
    wake_enabled = bool(args.wake_word_enabled)
    wake_word = normalize_text(args.wake_word)
    awake = not wake_enabled
    last_user_transcript_buf = ""

    # Prevent overlapping responses
    response_active = False

    async with await ws_connect(ws_url, headers) as ws:
        log.info("Connected to Realtime WebSocket.")
        log.info("Voice: %s", args.voice)
        if wake_enabled:
            log.info("Wake word enabled: '%s'", wake_word)
        else:
            log.info("Wake word disabled (always responding).")

        # Configure session
        await send_json(ws, {
            "type": "session.update",
            "session": {
                "instructions": system_prompt,
                "voice": args.voice,  # <-- CHANGE VOICE HERE (or via --voice / env var)
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "gpt-4o-transcribe"},
            }
        })
        log.info("Session initialized.")

        async def record_and_send():
            try:
                while True:
                    chunk = await audio.read_chunk_bytes()
                    b64 = base64.b64encode(chunk).decode("utf-8")
                    await send_json(ws, {"type": "input_audio_buffer.append", "audio": b64})
            except asyncio.CancelledError:
                return

        async def maybe_start_response():
            nonlocal response_active
            # If server_vad is set, some setups auto-respond. But your working setup seems NOT to.
            # So we explicitly create a response *only when allowed*.
            if response_active:
                return
            response_active = True
            await send_json(ws, {
                "type": "response.create",
                "response": {"modalities": ["audio", "text"]}
            })

        async def receiver():
            """
            Receive events from Realtime API, push audio deltas to play queue,
            detect wake word from USER transcription, and gate responses.
            """
            nonlocal awake, last_user_transcript_buf, response_active
            try:
                async for msg in ws:
                    event = json.loads(msg)
                    etype = event.get("type", "")

                    # --- MODEL audio
                    if etype in ("response.audio.delta", "response.output_audio.delta"):
                        delta = event.get("delta", "")
                        if delta:
                            pcm = base64.b64decode(delta)
                            if not play_q.full():
                                play_q.put_nowait(pcm)

                    # --- MODEL transcript (optional print)
                    elif etype in ("response.audio_transcript.delta", "response.output_text.delta"):
                        text = event.get("delta", "")
                        if text:
                            print(text, end="", flush=True)

                    # --- USER transcription (wake word detection)
                    elif etype in (
                        "conversation.item.input_audio_transcription.delta",
                        "conversation.item.input_audio_transcription.completed",
                    ):
                        part = event.get("delta") or event.get("transcript") or ""
                        if part:
                            last_user_transcript_buf += part
                            # print(part, end="", flush=True)  # uncomment if you want to see your own transcript
                            if wake_enabled and (not awake) and contains_wake_word(last_user_transcript_buf, wake_word):
                                awake = True
                                log.info("[wake word detected]")

                    # --- VAD signals end of user speech => decide to respond
                    elif etype == "input_audio_buffer.speech_stopped":
                        # If wake word is enabled: only respond when awake
                        if wake_enabled and (not awake):
                            last_user_transcript_buf = ""
                            continue

                        # Start response
                        await maybe_start_response()

                        # Go back to sleep after each turn (forces "Atlas ...")
                        if wake_enabled:
                            awake = False
                        last_user_transcript_buf = ""

                    # --- Response lifecycle
                    elif etype in ("response.done", "response.completed"):
                        response_active = False

                    elif etype == "error":
                        log.error("Server error: %s", event)
                        # If response died, allow next attempt
                        response_active = False

            except asyncio.CancelledError:
                return

        async def player():
            try:
                while True:
                    pcm = await play_q.get()
                    await audio.play_chunk_bytes(pcm)
            except asyncio.CancelledError:
                return

        tasks = [
            asyncio.create_task(record_and_send()),
            asyncio.create_task(receiver()),
            asyncio.create_task(player()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            audio.stop()


if __name__ == "__main__":
    asyncio.run(main())
