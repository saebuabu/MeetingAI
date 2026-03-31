"""
vergadering.py — Volledige vergaderpipeline in één commando

Gebruik:
    python vergadering.py                        # Opnemen + transcriberen + notulen
    python vergadering.py --input bestand.wav    # Sla opname over
    python vergadering.py --model medium         # Kies Whisper model
    python vergadering.py --skip-notulen         # Alleen opname + transcriptie
    python vergadering.py --ollama deepseek-r1:14b  # Kies Ollama model
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def stap_banner(nr: int, titel: str):
    print(f"\n{'='*50}")
    print(f"  Stap {nr}: {titel}")
    print(f"{'='*50}\n")


def run_script(script: str, args: list[str]) -> int:
    """Voer een ander script uit als subprocess."""
    cmd = [sys.executable, script] + args
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n[!] Gestopt door gebruiker.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Vergaderpipeline: opnemen → transcriberen → notulen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voorbeelden:
  python vergadering.py
  python vergadering.py --input opname.wav
  python vergadering.py --model medium --skip-notulen
        """,
    )
    parser.add_argument(
        "--input", "-i",
        help="Bestaand WAV-bestand (sla opname over)",
    )
    parser.add_argument(
        "--model", "-m",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model voor transcriptie (standaard: large-v3)",
    )
    parser.add_argument(
        "--taal", "-t",
        default="nl",
        help="Taalcode voor transcriptie (standaard: nl)",
    )
    parser.add_argument(
        "--ollama",
        default="deepseek-r1:14b",
        help="Ollama model voor notulen (standaard: deepseek-r1:14b)",
    )
    parser.add_argument(
        "--skip-notulen",
        action="store_true",
        help="Sla notulengeneratie over",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    basis_naam = f"vergadering_{timestamp}"

    print("\n" + "="*50)
    print("  MeetingAI — Lokale vergaderingsassistent")
    print("="*50)

    # --- Stap 1: Opname ---
    if args.input:
        wav_pad = Path(args.input)
        if not wav_pad.exists():
            print(f"[!] Invoerbestand niet gevonden: {wav_pad}")
            sys.exit(1)
        print(f"\n[*] Bestaand bestand gebruiken: {wav_pad}")
        # Gebruik de basis_naam van het invoerbestand
        basis_naam = wav_pad.stem
    else:
        stap_banner(1, "Vergadering opnemen")
        wav_pad = Path(f"{basis_naam}.wav")
        code = run_script("record.py", ["--output", str(wav_pad)])
        if code != 0:
            print("[!] Opname mislukt.")
            sys.exit(code)

    # --- Stap 2: Transcriptie ---
    stap_banner(2, "Transcriberen met Whisper")
    txt_pad = wav_pad.with_suffix(".txt")
    code = run_script(
        "transcribe.py",
        [str(wav_pad), "--model", args.model, "--taal", args.taal],
    )
    if code != 0:
        print("[!] Transcriptie mislukt.")
        sys.exit(code)

    if not txt_pad.exists():
        print(f"[!] Transcript niet gevonden na transcriptie: {txt_pad}")
        sys.exit(1)

    # --- Stap 3: Notulen ---
    if not args.skip_notulen:
        stap_banner(3, "Notulen genereren via Ollama")
        notulen_pad = txt_pad.with_name(txt_pad.stem + "_notulen.md")
        code = run_script(
            "notulen.py",
            [str(txt_pad), "--model", args.ollama, "--output", str(notulen_pad)],
        )
        if code != 0:
            print("[!] Notulengeneratie mislukt.")
            sys.exit(code)
    else:
        print("\n[*] Notulengeneratie overgeslagen (--skip-notulen).")
        notulen_pad = None

    # --- Overzicht ---
    print("\n" + "="*50)
    print("  Klaar! Bestanden:")
    print("="*50)
    print(f"  Audio:      {wav_pad}")
    print(f"  Transcript: {txt_pad}")
    if notulen_pad:
        print(f"  Notulen:    {notulen_pad}")
    print()


if __name__ == "__main__":
    main()
