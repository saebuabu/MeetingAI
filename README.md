# MeetingAI

Lokale vergaderingsassistent — volledig offline, geen cloud.

Opnemen → transcriberen → notulen genereren, alles op eigen hardware.

---

## Vereisten

- Windows 11
- NVIDIA GPU met CUDA (bijv. RTX 5080)
- Miniconda
- Ollama actief op poort 11434 met `deepseek-r1:14b`
- ffmpeg (zie installatie hieronder)

---

## Installatie

### 1. Repo clonen

```bash
git clone https://github.com/saebuabu/MeetingAI.git
cd MeetingAI
```

### 2. Miniconda installeren

Download Miniconda (Windows 64-bit installer) via:
https://docs.conda.io/en/latest/miniconda.html

Kies de Python 3.11 installer. Na installatie is conda beschikbaar in de **Anaconda Prompt**.
pip is automatisch beschikbaar binnen elke conda environment.

### 3. ffmpeg installeren

Whisper vereist ffmpeg. Download de Windows build via:
https://github.com/BtbN/FFmpeg-Builds/releases

Kies `ffmpeg-master-latest-win64-gpl.zip`, pak uit en voeg de `bin`-map toe aan je PATH.

Controleer daarna:
```bash
ffmpeg -version
```

### 4. Python omgeving aanmaken

Open de **Anaconda Prompt** (niet de gewone Command Prompt of PowerShell).

```bash
conda create -n meetingai python=3.11
conda activate meetingai

# PyTorch met CUDA (pas cu124 aan naar jouw CUDA versie indien nodig)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Overige dependencies
pip install pyaudiowpatch openai-whisper numpy requests
```

### 5. Installatie controleren

Voer dit uit om te bevestigen dat alles correct is geïnstalleerd:

```bash
conda activate meetingai
python -c "import numpy, whisper, pyaudiowpatch, requests; print('Alle packages OK')"
```

Je moet `Alle packages OK` zien. Zie je in plaats daarvan een `ModuleNotFoundError`, voer dan stap 4 opnieuw uit.

---

## Gebruik

### Alles in één keer

```bash
conda activate meetingai
cd MeetingAI
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

## Problemen oplossen

### `ModuleNotFoundError: No module named 'numpy'` (of andere packages)

Dit betekent dat Python de conda omgeving **meetingai** niet gebruikt. Controleer:

1. **Gebruik Anaconda Prompt**, niet de gewone Command Prompt of PowerShell.
2. Activeer de omgeving vóór elk gebruik:
   ```bash
   conda activate meetingai
   ```
   De prompt moet veranderen naar `(meetingai) C:\...`
3. Controleer of je in de juiste omgeving bent:
   ```bash
   where python
   ```
   Het pad moet `...\envs\meetingai\python.exe` bevatten, niet `C:\Python312\python.exe`.
4. Als packages nog steeds ontbreken, installeer ze opnieuw binnen de omgeving:
   ```bash
   conda activate meetingai
   pip install pyaudiowpatch openai-whisper numpy requests
   ```

### PyTorch incompatibel met GPU (sm_86 / RTX 3080 of nieuwer)

Als je deze melding ziet tijdens transcriptie:
> NVIDIA GeForce RTX 3080 with CUDA capability sm_86 is not compatible with the current PyTorch installation.

Installeer PyTorch opnieuw met CUDA 12.4 support:

```bash
conda activate meetingai
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Whisper modellen

| Model | VRAM | Snelheid | Kwaliteit |
|---|---|---|---|
| `tiny` | ~1GB | Zeer snel | Basis |
| `medium` | ~5GB | Snel | Goed |
| `large-v3` | ~10GB | Normaal | Beste (aanbevolen) |
