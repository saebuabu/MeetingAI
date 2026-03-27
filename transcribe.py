"""
transcribe.py — Spraak naar tekst met Whisper (GPU)

Gebruik:
    python transcribe.py vergadering_20260327_1400.wav
    python transcribe.py vergadering.wav --model medium
    python transcribe.py vergadering.wav --taal en

Vereist: openai-whisper, torch (CUDA)
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Transcribeer een audiobestand met Whisper")
    parser.add_argument("bestand", help="WAV-bestand om te transcriberen")
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
        help="Uitvoerbestand (standaard: <invoer>.txt)",
    )
    args = parser.parse_args()

    invoer = Path(args.bestand)
    if not invoer.exists():
        print(f"[!] Bestand niet gevonden: {invoer}")
        sys.exit(1)

    uitvoer = Path(args.output) if args.output else invoer.with_suffix(".txt")

    # Laad Whisper (importeer pas hier zodat foutmeldingen duidelijk zijn)
    try:
        import whisper
    except ImportError:
        print("[!] whisper niet gevonden. Installeer met:")
        print("    pip install openai-whisper")
        sys.exit(1)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    if device == "cpu":
        print("[!] CUDA niet beschikbaar — transcriptie draait op CPU (langzamer).")
    else:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[+] GPU: {gpu_name}")
        except Exception:
            pass

    print(f"[*] Model laden: {args.model} (device: {device})")
    model = whisper.load_model(args.model, device=device)

    print(f"[*] Transcriberen: {invoer}")
    print(f"[*] Taal: {args.taal}\n")

    result = model.transcribe(
        str(invoer),
        language=args.taal,
        verbose=True,          # Toont voortgang per segment
        fp16=(device == "cuda"),
    )

    transcript = result["text"].strip()

    # Schrijf resultaat
    with open(uitvoer, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"\n[+] Transcript opgeslagen: {uitvoer}")
    print(f"[>] Volgende stap: python notulen.py {uitvoer}")

    # Toon ook gedetailleerd transcript met tijdstempels
    tijdstempel_pad = uitvoer.with_name(uitvoer.stem + "_tijdstempels.txt")
    with open(tijdstempel_pad, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = segment["start"]
            eind = segment["end"]
            tekst = segment["text"].strip()
            mins_s, secs_s = divmod(int(start), 60)
            mins_e, secs_e = divmod(int(eind), 60)
            f.write(f"[{mins_s:02d}:{secs_s:02d} → {mins_e:02d}:{secs_e:02d}] {tekst}\n")

    print(f"[+] Transcript met tijdstempels: {tijdstempel_pad}")


if __name__ == "__main__":
    main()
