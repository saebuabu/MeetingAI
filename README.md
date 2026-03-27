# MeetingAI

Lokale vergaderingsassistent — volledig offline, geen cloud.

Opnemen → transcriberen → notulen genereren, alles op eigen hardware.

---

## Vereisten

- Windows 11
- NVIDIA GPU met CUDA (bijv. RTX 5080)
- Miniconda
- Ollama actief op poort 11434 met `deepseek-r1:14b`

---

## Installatie

```bash
conda create -n meetingai python=3.11
conda activate meetingai

# PyTorch met CUDA (pas cu121 aan naar jouw CUDA versie indien nodig)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Overige dependencies
pip install pyaudiowpatch openai-whisper numpy requests
```

---

## Gebruik

### Alles in één keer

```bash
python vergadering.py
```

Start opname → Druk Ctrl+C om te stoppen → Whisper transcribeert → Ollama genereert notulen.

### Stap voor stap

```bash
# 1. Opname
python record.py
# → vergadering_20260327_1400.wav

# 2. Transcriptie
python transcribe.py vergadering_20260327_1400.wav
# → vergadering_20260327_1400.txt
# → vergadering_20260327_1400_tijdstempels.txt

# 3. Notulen
python notulen.py vergadering_20260327_1400.txt
# → vergadering_20260327_1400_notulen.md
```

### Bestaand audiobestand verwerken

```bash
python vergadering.py --input mijn_opname.wav
```

---

## Opties

| Script | Optie | Beschrijving |
|---|---|---|
| `vergadering.py` | `--input bestand.wav` | Sla opname over |
| `vergadering.py` | `--model medium` | Whisper model kiezen |
| `vergadering.py` | `--skip-notulen` | Alleen transcriberen |
| `vergadering.py` | `--ollama deepseek-r1:14b` | Ollama model kiezen |
| `transcribe.py` | `--model large-v3` | Whisper model (tiny/base/small/medium/large-v3) |
| `transcribe.py` | `--taal nl` | Taal voor transcriptie |
| `notulen.py` | `--model deepseek-r1:14b` | Ollama model voor notulen |
| `record.py` | `--output bestand.wav` | Uitvoerbestandsnaam |

---

## Uitvoerbestanden

| Bestand | Inhoud |
|---|---|
| `vergadering_YYYYMMDD_HHMM.wav` | Audio-opname (16kHz mono) |
| `vergadering_YYYYMMDD_HHMM.txt` | Volledig transcript |
| `vergadering_YYYYMMDD_HHMM_tijdstempels.txt` | Transcript met tijdstempels per segment |
| `vergadering_YYYYMMDD_HHMM_notulen.md` | Gestructureerde notulen (Markdown) |

---

## Whisper modellen

| Model | VRAM | Snelheid | Kwaliteit |
|---|---|---|---|
| `tiny` | ~1GB | Zeer snel | Basis |
| `medium` | ~5GB | Snel | Goed |
| `large-v3` | ~10GB | Normaal | Beste (aanbevolen) |
