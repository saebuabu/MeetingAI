"""
live_transcribe.py — Realtime opname + transcriptie met Whisper

Gebruik:
    python live_transcribe.py
    python live_transcribe.py --model medium --taal nl
    python live_transcribe.py --output output/vergadering.txt
    python live_transcribe.py --save-wav
    python live_transcribe.py --wie          # Sprekerherkenning inschakelen

Stoppen: Ctrl+C
Vereist: pyaudiowpatch, openai-whisper, torch, resemblyzer (voor --wie)
"""

import argparse
import queue
import sys
import threading
import time
import wave
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHUNK_SECONDS = 20
OVERLAP_SECONDS = 2

AudioChunk = namedtuple("AudioChunk", ["audio_bytes", "wall_offset_seconds", "chunk_index"])


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
    if not data or src_rate == dst_rate:
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


def register_speakers(pa, mic_idx, mic_info):
    """Registreer deelnemers door ~5 seconden stem op te nemen en embeddings op te slaan."""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        print("[!] resemblyzer niet gevonden. Installeer met: pip install resemblyzer")
        sys.exit(1)

    encoder = VoiceEncoder()
    print("\n[*] Sprekerregistratie")
    print("[*] Elke deelnemer spreekt ~5 seconden (bijv. naam + een korte zin)\n")

    n = 0
    while n < 1:
        try:
            n = int(input("[?] Hoeveel lokale deelnemers? "))
        except ValueError:
            pass

    speaker_profiles = {}
    mic_rate = int(mic_info["defaultSampleRate"])
    record_seconds = 5

    for i in range(n):
        name = input(f"[?] Naam van deelnemer {i + 1}: ").strip() or f"Spreker{i + 1}"
        input(f"[>] Druk Enter en spreek dan ~{record_seconds} seconden ({name})...")

        frames = []
        stream = pa.open(
            format=FORMAT, channels=1, rate=mic_rate,
            input=True, input_device_index=mic_idx,
            frames_per_buffer=CHUNK,
        )
        print("[*] Opnemen... ", end="", flush=True)
        for _ in range(int(mic_rate / CHUNK * record_seconds)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        print("klaar.")

        raw = b"".join(frames)
        if mic_rate != SAMPLE_RATE:
            raw = resample_chunk(raw, mic_rate, SAMPLE_RATE)
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        wav = preprocess_wav(audio_np, source_sr=SAMPLE_RATE)
        embed = encoder.embed_utterance(wav)
        speaker_profiles[name] = embed
        print(f"[+] {name} geregistreerd.\n")

    print(f"[+] {len(speaker_profiles)} deelnemer(s) geregistreerd. Remote sprekers worden als 'Remote' gelabeld.\n")
    return encoder, speaker_profiles, preprocess_wav


def identify_speaker(embed, speaker_profiles, threshold=0.75):
    """Vergelijk embedding met geregistreerde sprekers. Geen match boven threshold → 'Remote'."""
    best_name = "Remote"
    best_sim = threshold
    for name, profile_embed in speaker_profiles.items():
        sim = float(np.dot(embed, profile_embed))
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name


def mix_frames(mic_data, sys_data):
    """Mix microfoon en systeem audio (gemiddelde, clamp op int16 grenzen)."""
    mic = np.frombuffer(mic_data, dtype=np.int16).astype(np.int32)
    sys_audio = np.frombuffer(sys_data, dtype=np.int16).astype(np.int32)
    min_len = min(len(mic), len(sys_audio))
    mixed = np.clip((mic[:min_len] + sys_audio[:min_len]) // 2, -32768, 32767).astype(np.int16)
    return mixed.tobytes()


def main():
    parser = argparse.ArgumentParser(description="Realtime opname + transcriptie met Whisper")
    parser.add_argument(
        "--model", "-m",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model (standaard: large-v3)",
    )
    parser.add_argument(
        "--taal", "-t",
        default="nl",
        help="Taalcode, bijv. 'nl' of 'en' (standaard: nl)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Uitvoerbestand .txt (standaard: output/<sessie>/<sessie>.txt)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=CHUNK_SECONDS,
        help=f"Seconden audio per Whisper-batch (standaard: {CHUNK_SECONDS}; verhoog op CPU)",
    )
    parser.add_argument(
        "--save-wav",
        action="store_true",
        help="Sla ook een WAV-bestand op na afloop",
    )
    parser.add_argument(
        "--wie",
        action="store_true",
        help="Sprekerherkenning inschakelen (vereist resemblyzer)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    basis_naam = f"vergadering_{timestamp}"

    if args.output:
        txt_pad = Path(args.output)
        session_dir = txt_pad.parent
        basis_naam = txt_pad.stem
    else:
        session_dir = Path("output") / basis_naam
        txt_pad = session_dir / f"{basis_naam}.txt"

    session_dir.mkdir(parents=True, exist_ok=True)
    tijdstempel_pad = txt_pad.with_name(txt_pad.stem + "_tijdstempels.txt")

    # Laad Whisper (vóór streams openen zodat vertraging geen audio mist)
    try:
        import whisper
    except ImportError:
        print("[!] whisper niet gevonden. Installeer met: pip install openai-whisper")
        sys.exit(1)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    if device == "cpu":
        print("[!] CUDA niet beschikbaar — transcriptie draait op CPU (langzamer).")
        print(f"    Overweeg --chunk-seconds te verhogen of een kleiner model te gebruiken.\n")
    else:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[+] GPU: {gpu_name}")
        except Exception:
            pass

    print(f"[*] Model laden: {args.model} (device: {device}) ...")
    model = whisper.load_model(args.model, device=device)
    print(f"[+] Model geladen.\n")

    # Initialiseer PyAudio en zoek devices
    pa = pyaudio.PyAudio()
    sample_width = pa.get_sample_size(FORMAT)  # ophalen vóór terminate()

    loopback_idx, loopback_info = get_loopback_device(pa)
    mic_idx, mic_info = get_default_mic(pa)

    if loopback_idx is None:
        print("[!] Geen WASAPI loopback device gevonden. Alleen microfoon wordt opgenomen.")
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

    # Sprekerregistratie (optioneel)
    encoder = None
    speaker_profiles = {}
    preprocess_wav = None
    if args.wie:
        if mic_idx is None:
            print("[!] --wie vereist een microfoon. Sprekerherkenning uitgeschakeld.")
        else:
            encoder, speaker_profiles, preprocess_wav = register_speakers(pa, mic_idx, mic_info)

    print(f"\n[*] Live transcriptie gestart — Ctrl+C om te stoppen")
    print(f"[*] Uitvoer: {txt_pad}")
    print(f"[*] Eerste transcriptie verschijnt na ~{args.chunk_seconds}s\n")

    # Gedeelde toestand
    lock = threading.Lock()
    mic_buffer = []
    sys_buffer = []
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    print_lock = threading.Lock()
    segments_list = []       # (abs_start, abs_end, speaker, text)
    all_audio_bytes = []     # voor --save-wav
    stats = {"chunks_verwerkt": 0, "in_wachtrij": 0}

    def mic_callback(in_data, frame_count, time_info, status):
        with lock:
            mic_buffer.append(in_data)
        return (None, pyaudio.paContinue)

    def sys_callback(in_data, frame_count, time_info, status):
        with lock:
            sys_buffer.append(in_data)
        return (None, pyaudio.paContinue)

    def accumulator():
        """Draint audiobuffers, mixt/resamplet en stuurt chunks naar de transcriptie-queue."""
        pending = b""
        overlap_tail = b""
        chunk_index = 0
        bytes_per_sample = 2  # int16
        chunk_bytes = args.chunk_seconds * SAMPLE_RATE * bytes_per_sample
        overlap_bytes = OVERLAP_SECONDS * SAMPLE_RATE * bytes_per_sample
        next_start_samples = 0

        while not stop_event.is_set():
            time.sleep(0.25)

            with lock:
                local_mic = list(mic_buffer)
                local_sys = list(sys_buffer)
                mic_buffer.clear()
                sys_buffer.clear()

            has_mic = len(local_mic) > 0
            has_sys = len(local_sys) > 0

            if has_mic and has_sys:
                mic_rate = int(mic_info["defaultSampleRate"])
                sys_rate = int(loopback_info["defaultSampleRate"])
                sys_channels = int(loopback_info["maxInputChannels"])
                mic_data = b"".join(local_mic)
                sys_data = b"".join(local_sys)
                if sys_channels > 1:
                    s = np.frombuffer(sys_data, dtype=np.int16)
                    s = s.reshape(-1, sys_channels).mean(axis=1).astype(np.int16)
                    sys_data = s.tobytes()
                mic_r = resample_chunk(mic_data, mic_rate, SAMPLE_RATE)
                sys_r = resample_chunk(sys_data, sys_rate, SAMPLE_RATE)
                mixed = mix_frames(mic_r, sys_r)
            elif has_mic:
                mic_rate = int(mic_info["defaultSampleRate"])
                raw = b"".join(local_mic)
                mixed = resample_chunk(raw, mic_rate, SAMPLE_RATE)
            elif has_sys:
                sys_rate = int(loopback_info["defaultSampleRate"])
                sys_channels = int(loopback_info["maxInputChannels"])
                raw = b"".join(local_sys)
                if sys_channels > 1:
                    s = np.frombuffer(raw, dtype=np.int16)
                    s = s.reshape(-1, sys_channels).mean(axis=1).astype(np.int16)
                    raw = s.tobytes()
                mixed = resample_chunk(raw, sys_rate, SAMPLE_RATE)
            else:
                continue

            if args.save_wav:
                all_audio_bytes.append(mixed)

            pending += mixed

            while len(pending) >= chunk_bytes:
                wall_offset = next_start_samples / SAMPLE_RATE
                chunk_payload = overlap_tail + pending[:chunk_bytes]
                overlap_tail = pending[chunk_bytes - overlap_bytes:chunk_bytes]
                pending = pending[chunk_bytes:]

                audio_queue.put(AudioChunk(chunk_payload, wall_offset, chunk_index))
                stats["in_wachtrij"] = audio_queue.qsize()

                next_start_samples += (chunk_bytes - overlap_bytes) // bytes_per_sample
                chunk_index += 1

        # Flush resterende audio (minimaal 1 seconde)
        min_bytes = SAMPLE_RATE * bytes_per_sample
        if len(pending) >= min_bytes:
            wall_offset = next_start_samples / SAMPLE_RATE
            chunk_payload = overlap_tail + pending
            audio_queue.put(AudioChunk(chunk_payload, wall_offset, chunk_index))
            stats["in_wachtrij"] = audio_queue.qsize()

    def transcription_worker():
        """Transcribeert audiochunks en print segmenten realtime."""
        last_committed_time = -1.0

        while not stop_event.is_set() or not audio_queue.empty():
            try:
                chunk = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            stats["in_wachtrij"] = audio_queue.qsize()

            audio_np = np.frombuffer(chunk.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                result = model.transcribe(
                    audio_np,
                    language=args.taal,
                    verbose=False,
                    fp16=(device == "cuda"),
                )
            except Exception as e:
                with print_lock:
                    print(f"\n[!] Transcriptie fout (chunk {chunk.chunk_index}): {e}", flush=True)
                audio_queue.task_done()
                continue

            for seg in result["segments"]:
                abs_start = chunk.wall_offset_seconds + seg["start"]
                abs_end = chunk.wall_offset_seconds + seg["end"]
                text = seg["text"].strip()

                if not text:
                    continue

                # Overlap-deduplicatie: sla segmenten over die al in vorige chunk zaten
                if abs_start <= last_committed_time:
                    continue

                last_committed_time = abs_start

                # Sprekerherkenning op basis van segment-audio
                speaker = ""
                if speaker_profiles and encoder is not None:
                    bytes_per_sample = 2
                    seg_start_byte = int(seg["start"] * SAMPLE_RATE) * bytes_per_sample
                    seg_end_byte = int(seg["end"] * SAMPLE_RATE) * bytes_per_sample
                    seg_audio = chunk.audio_bytes[seg_start_byte:seg_end_byte]
                    # Minimaal 0.5 seconde audio nodig voor betrouwbare embedding
                    if len(seg_audio) >= SAMPLE_RATE * bytes_per_sample // 2:
                        try:
                            audio_np = np.frombuffer(seg_audio, dtype=np.int16).astype(np.float32) / 32768.0
                            wav = preprocess_wav(audio_np, source_sr=SAMPLE_RATE)
                            embed = encoder.embed_utterance(wav)
                            speaker = identify_speaker(embed, speaker_profiles)
                        except Exception:
                            speaker = "?"

                segments_list.append((abs_start, abs_end, speaker, text))

                label = f"{speaker}: " if speaker else ""
                mins, secs = divmod(int(abs_start), 60)
                with print_lock:
                    print(f"\n[{mins:02d}:{secs:02d}] {label}{text}", flush=True)

            stats["chunks_verwerkt"] += 1
            audio_queue.task_done()

    # Open PyAudio streams
    streams = []

    if mic_idx is not None:
        mic_rate = int(mic_info["defaultSampleRate"])
        mic_stream = pa.open(
            format=FORMAT, channels=1, rate=mic_rate,
            input=True, input_device_index=mic_idx,
            frames_per_buffer=CHUNK, stream_callback=mic_callback,
        )
        streams.append(mic_stream)

    if loopback_idx is not None:
        sys_rate = int(loopback_info["defaultSampleRate"])
        sys_channels = int(loopback_info["maxInputChannels"])
        sys_stream = pa.open(
            format=FORMAT, channels=sys_channels, rate=sys_rate,
            input=True, input_device_index=loopback_idx,
            frames_per_buffer=CHUNK, stream_callback=sys_callback,
        )
        streams.append(sys_stream)

    for s in streams:
        s.start_stream()

    acc_thread = threading.Thread(target=accumulator, daemon=True, name="accumulator")
    acc_thread.start()

    trans_thread = threading.Thread(target=transcription_worker, daemon=False, name="transcriptie")
    trans_thread.start()

    start_time = time.time()

    try:
        while True:
            time.sleep(0.5)
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            with print_lock:
                print(
                    f"\r[*] Opnametijd: {mins:02d}:{secs:02d} | "
                    f"Chunks verwerkt: {stats['chunks_verwerkt']} | "
                    f"In wachtrij: {stats['in_wachtrij']}   ",
                    end="", flush=True,
                )
    except KeyboardInterrupt:
        print("\n\n[*] Opname gestopt. Resterende audio verwerken...")

    stop_event.set()

    for s in streams:
        s.stop_stream()
        s.close()

    # Wacht tot transcriptie-queue leeg is (max 5 minuten)
    trans_thread.join(timeout=300)
    if trans_thread.is_alive():
        print("[!] Transcriptie time-out — sommige audio mogelijk niet verwerkt.")

    pa.terminate()

    if not segments_list:
        print("[!] Geen transcriptie beschikbaar. Bestanden niet opgeslagen.")
        sys.exit(1)

    segments_list.sort(key=lambda x: x[0])

    with open(txt_pad, "w", encoding="utf-8") as f:
        for _, _, speaker, text in segments_list:
            label = f"{speaker}: " if speaker else ""
            f.write(f"{label}{text}\n")
    print(f"[+] Transcript opgeslagen: {txt_pad}")

    with open(tijdstempel_pad, "w", encoding="utf-8") as f:
        for abs_start, abs_end, speaker, text in segments_list:
            mins_s, secs_s = divmod(int(abs_start), 60)
            mins_e, secs_e = divmod(int(abs_end), 60)
            label = f"{speaker}: " if speaker else ""
            f.write(f"[{mins_s:02d}:{secs_s:02d} → {mins_e:02d}:{secs_e:02d}] {label}{text}\n")
    print(f"[+] Tijdstempels opgeslagen: {tijdstempel_pad}")

    if args.save_wav:
        wav_pad = txt_pad.with_suffix(".wav")
        final_data = b"".join(all_audio_bytes)
        if final_data:
            with wave.open(str(wav_pad), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(sample_width)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(final_data)
            print(f"[+] Audio opgeslagen: {wav_pad}")

    print(f"[>] Volgende stap: python notulen.py {txt_pad}")


if __name__ == "__main__":
    main()
