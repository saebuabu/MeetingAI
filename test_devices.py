import sounddevice as sd


def list_devices():
    print("Beschikbare audio-invoerapparaten:")
    print("=" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{i}] {device['name']}")
            print(f"      Kanalen: {device['max_input_channels']}, Samplerate: {int(device['default_samplerate'])} Hz")
    print()


def test_device(device_id=None, duration=3, samplerate=16000):
    device_info = sd.query_devices(device_id, 'input')
    print(f"Test apparaat: {device_info['name']}")
    print(f"Opname van {duration} seconden... (spreek iets in)")

    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        device=device_id
    )
    sd.wait()

    volume = (recording ** 2).mean() ** 0.5
    print(f"RMS volume: {volume:.4f}")

    if volume < 0.001:
        print("WAARSCHUWING: Geen geluid gedetecteerd.")
    else:
        print("OK: Geluid gedetecteerd.")


if __name__ == "__main__":
    list_devices()

    device_id = input("Voer het apparaatnummer in om te testen (Enter voor standaard): ").strip()
    device_id = int(device_id) if device_id else None

    test_device(device_id)
