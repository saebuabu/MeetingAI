"""
record.py — Vergadering opnemen (microfoon + systeem audio)

Gebruik:
    python record.py
    python record.py --output mijn_vergadering.wav

Stoppen: Ctrl+C
Vereist: pyaudiowpatch
"""

import argparse
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

SAMPLE_RATE = 16000   # Whisper verwacht 16kHz
CHANNELS = 1          # Mono
CHUNK = 1024
FORMAT = pyaudio.paInt16


def get_loopback_device(pa):
    """Zoek het WASAPI loopback device (systeem audio)."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("isLoopbackDevice", False):
            return i, info
    return None, None


def get_default_mic(pa):
    """Geef de standaard microfoon terug."""
    try:
        info = pa.get_default_input_device_info()
        return info["index"], info
    except OSError:
        return None, None


def resample_chunk(data, src_rate, dst_rate):
    """Eenvoudige lineaire resampling naar dst_rate."""
    if src_rate == dst_rate:
        return data
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    ratio = dst_rate / src_rate
    new_len = int(len(samples) * ratio)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, new_len),
        np.arange(len(samples)),
        samples,
    ).astype(np.int16)
    return resampled.tobytes()


def mix_frames(mic_data, sys_data):
    """Mix microfoon en systeem audio (gemiddelde, clamp op int16 grenzen)."""
    mic = np.frombuffer(mic_data, dtype=np.int16).astype(np.int32)
    sys = np.frombuffer(sys_data, dtype=np.int16).astype(np.int32)

    # Maak even lang
    min_len = min(len(mic), len(sys))
    mic = mic[:min_len]
    sys = sys[:min_len]

    mixed = np.clip((mic + sys) // 2, -32768, 32767).astype(np.int16)
    return mixed.tobytes()


def main():
    parser = argparse.ArgumentParser(description="Vergadering opnemen")
    parser.add_argument(
        "--output", "-o",
        help="Uitvoerbestand (standaard: vergadering_YYYYMMDD_HHMM.wav)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(args.output) if args.output else Path(f"vergadering_{timestamp}.wav")

    pa = pyaudio.PyAudio()

    # Zoek devices
    loopback_idx, loopback_info = get_loopback_device(pa)
    mic_idx, mic_info = get_default_mic(pa)

    if loopback_idx is None:
        print("[!] Geen WASAPI loopback device gevonden.")
        print("    Zorg dat 'Stereo Mix' of een loopback device actief is in Windows.")
        print("    Alleen microfoon wordt opgenomen.")
    else:
        print(f"[+] Systeem audio: {loopback_info['name']}")

    if mic_idx is None:
        print("[!] Geen microfoon gevonden. Alleen systeem audio wordt opgenomen.")
    else:
        print(f"[+] Microfoon: {mic_info['name']}")

    if loopback_idx is None and mic_idx is None:
        print("[!] Geen audio devices gevonden. Afbreken.")
        pa.terminate()
        sys.exit(1)

    print(f"\n[*] Opname gestart → {output_path}")
    print("[*] Druk op Ctrl+C om te stoppen.\n")

    frames = []
    stop_event = threading.Event()
    lock = threading.Lock()

    mic_buffer = []
    sys_buffer = []

    def mic_callback(in_data, frame_count, time_info, status):
        with lock:
            mic_buffer.append(in_data)
        return (None, pyaudio.paContinue)

    def sys_callback(in_data, frame_count, time_info, status):
        with lock:
            sys_buffer.append(in_data)
        return (None, pyaudio.paContinue)

    streams = []

    if mic_idx is not None:
        mic_rate = int(mic_info["defaultSampleRate"])
        mic_stream = pa.open(
            format=FORMAT,
            channels=1,
            rate=mic_rate,
            input=True,
            input_device_index=mic_idx,
            frames_per_buffer=CHUNK,
            stream_callback=mic_callback,
        )
        streams.append(("mic", mic_stream, mic_rate))

    if loopback_idx is not None:
        sys_rate = int(loopback_info["defaultSampleRate"])
        sys_channels = int(loopback_info["maxInputChannels"])
        sys_stream = pa.open(
            format=FORMAT,
            channels=sys_channels,
            rate=sys_rate,
            input=True,
            input_device_index=loopback_idx,
            frames_per_buffer=CHUNK,
            stream_callback=sys_callback,
        )
        streams.append(("sys", sys_stream, sys_rate, sys_channels))

    for s in streams:
        s[1].start_stream()

    start_time = time.time()

    try:
        while True:
            time.sleep(0.5)
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            print(f"\r[*] Opnametijd: {mins:02d}:{secs:02d}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n[*] Opname gestopt.")

    for s in streams:
        s[1].stop_stream()
        s[1].close()

    # Verwerk buffers
    with lock:
        mic_frames = list(mic_buffer)
        sys_frames = list(sys_buffer)

    # Bepaal welke combinatie beschikbaar is
    has_mic = len(mic_frames) > 0
    has_sys = len(sys_frames) > 0

    if has_mic and has_sys:
        # Resample beide naar SAMPLE_RATE en mix
        mic_rate = int(mic_info["defaultSampleRate"])
        sys_rate = int(loopback_info["defaultSampleRate"])
        sys_channels = int(loopback_info["maxInputChannels"])

        mic_data = b"".join(mic_frames)
        sys_data = b"".join(sys_frames)

        # Systeem audio: stereo → mono
        if sys_channels > 1:
            sys_samples = np.frombuffer(sys_data, dtype=np.int16)
            sys_samples = sys_samples.reshape(-1, sys_channels).mean(axis=1).astype(np.int16)
            sys_data = sys_samples.tobytes()

        # Resample naar 16kHz
        mic_resampled = resample_chunk(mic_data, mic_rate, SAMPLE_RATE)
        sys_resampled = resample_chunk(sys_data, sys_rate, SAMPLE_RATE)

        # Mix
        final_data = mix_frames(mic_resampled, sys_resampled)

    elif has_mic:
        mic_rate = int(mic_info["defaultSampleRate"])
        raw = b"".join(mic_frames)
        final_data = resample_chunk(raw, mic_rate, SAMPLE_RATE)

    else:
        sys_rate = int(loopback_info["defaultSampleRate"])
        sys_channels = int(loopback_info["maxInputChannels"])
        raw = b"".join(sys_frames)
        if sys_channels > 1:
            samples = np.frombuffer(raw, dtype=np.int16)
            samples = samples.reshape(-1, sys_channels).mean(axis=1).astype(np.int16)
            raw = samples.tobytes()
        final_data = resample_chunk(raw, sys_rate, SAMPLE_RATE)

    # Schrijf WAV
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(final_data)

    pa.terminate()

    duration = len(np.frombuffer(final_data, dtype=np.int16)) / SAMPLE_RATE
    mins, secs = divmod(int(duration), 60)
    print(f"[+] Opgeslagen: {output_path} ({mins:02d}:{secs:02d})")
    print(f"[>] Volgende stap: python transcribe.py {output_path}")


if __name__ == "__main__":
    main()
