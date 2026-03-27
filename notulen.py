"""
notulen.py — Genereer gestructureerde notulen via Ollama

Gebruik:
    python notulen.py vergadering_20260327_1400.txt
    python notulen.py vergadering.txt --model deepseek-r1:14b
    python notulen.py vergadering.txt --output mijn_notulen.md

Vereist: Ollama actief op http://localhost:11434
"""

import argparse
import json
import sys
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
STANDAARD_MODEL = "deepseek-r1:14b"

NOTULEN_PROMPT = """Je bent een professionele notulist. Hieronder staat het transcript van een vergadering.
Maak er gestructureerde notulen van in het Nederlands.

Gebruik precies deze structuur in Markdown:

## Samenvatting
(3 tot 5 zinnen die de kern van de vergadering beschrijven)

## Besluiten
(Bullet list van genomen beslissingen. Als er geen duidelijke besluiten zijn, schrijf dan: "Geen expliciete besluiten genomen.")

## Actiepunten
(Bullet list van: wie doet wat, indien vermeld. Formaat: **Naam** — actie. Als namen niet bekend zijn: "- [onbekend] — beschrijving". Als er geen actiepunten zijn: "Geen actiepunten vastgesteld.")

## Overige aandachtspunten
(Bullet list van onderwerpen die besproken zijn maar geen besluit of actie hebben gekregen, of zaken die opgevolgd moeten worden.)

---
TRANSCRIPT:
{transcript}
---

Geef alleen de notulen terug, geen uitleg of commentaar daarbuiten."""


def check_ollama():
    """Controleer of Ollama bereikbaar is."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def genereer_notulen(transcript: str, model: str) -> str:
    """Stuur transcript naar Ollama en ontvang notulen."""
    prompt = NOTULEN_PROMPT.format(transcript=transcript)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    print(f"[*] Notulen genereren met {model}...")
    print("[*] Voortgang: ", end="", flush=True)

    response_tekst = []
    in_think_block = False

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")

                # Sla <think>...</think> blokken over (deepseek-r1 redenering)
                if "<think>" in token:
                    in_think_block = True
                if in_think_block:
                    if "</think>" in token:
                        in_think_block = False
                    else:
                        print(".", end="", flush=True)
                    continue

                response_tekst.append(token)

                if data.get("done", False):
                    break

    except requests.exceptions.Timeout:
        print("\n[!] Timeout — Ollama reageert te langzaam.")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n[!] Kan geen verbinding maken met Ollama.")
        sys.exit(1)

    print(" klaar\n")
    return "".join(response_tekst).strip()


def main():
    parser = argparse.ArgumentParser(description="Genereer notulen van een transcript")
    parser.add_argument("bestand", help="Transcript tekstbestand (.txt)")
    parser.add_argument(
        "--model", "-m",
        default=STANDAARD_MODEL,
        help=f"Ollama model (standaard: {STANDAARD_MODEL})",
    )
    parser.add_argument(
        "--output", "-o",
        help="Uitvoerbestand (standaard: <invoer>_notulen.md)",
    )
    args = parser.parse_args()

    invoer = Path(args.bestand)
    if not invoer.exists():
        print(f"[!] Bestand niet gevonden: {invoer}")
        sys.exit(1)

    uitvoer = Path(args.output) if args.output else invoer.with_name(invoer.stem + "_notulen.md")

    # Controleer Ollama
    if not check_ollama():
        print("[!] Ollama is niet bereikbaar op http://localhost:11434")
        print("    Zorg dat Ollama actief is als Windows-service.")
        sys.exit(1)

    print(f"[+] Ollama bereikbaar")

    # Lees transcript
    transcript = invoer.read_text(encoding="utf-8").strip()
    if not transcript:
        print(f"[!] Transcript is leeg: {invoer}")
        sys.exit(1)

    woorden = len(transcript.split())
    print(f"[*] Transcript: {invoer.name} ({woorden} woorden)")

    # Genereer notulen
    notulen = genereer_notulen(transcript, args.model)

    # Voeg metadata toe
    from datetime import datetime
    header = f"# Notulen\n\n_Gegenereerd op {datetime.now().strftime('%d-%m-%Y %H:%M')} met {args.model}_\n_Bron: {invoer.name}_\n\n---\n\n"
    inhoud = header + notulen

    uitvoer.write_text(inhoud, encoding="utf-8")
    print(f"[+] Notulen opgeslagen: {uitvoer}")


if __name__ == "__main__":
    main()
