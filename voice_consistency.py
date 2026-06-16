import time
import itertools
import threading
import numpy as np
import librosa
from tqdm import tqdm
from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = "gemini-3.5-flash"

AUDIO_FILE_PATH = r"C:\Users\Asus\Desktop\Wippi\ContentQC\BT-AN02-EPIC-ALADIN-CHAPTER 3.mp3"
SCRIPT_FILE_PATH = r"C:\Users\Asus\Downloads\Script - BT-AN02-EPIC - Final .md"

SEGMENT_DURATION = 3  # seconds — shorter windows catch transient artifacts


def load_script(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fmt_ts(seconds: float) -> str:
    return f"[{int(seconds // 60):02d}:{int(seconds % 60):02d}]"


def spinner(label: str, stop_event: threading.Event):
    """Displays a spinner in the terminal until stop_event is set."""
    for char in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        if stop_event.is_set():
            break
        print(f"\r{char} {label}", end="", flush=True)
        time.sleep(0.1)
    print(f"\r✓ {label}")


def run_spinner(label: str):
    """Returns (stop_event, thread) — call stop_event.set() to stop."""
    stop_event = threading.Event()
    t = threading.Thread(target=spinner, args=(label, stop_event), daemon=True)
    t.start()
    return stop_event, t


def extract_quality_features(audio_path: str) -> str:
    # ── Step 1: Load audio ────────────────────────────────────────────────────
    stop, t = run_spinner("Loading audio file...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    total_duration = librosa.get_duration(y=y, sr=sr)
    stop.set(); t.join()

    hop_length = 256

    # ── Step 2: Extract features ──────────────────────────────────────────────
    stop, t = run_spinner("Extracting pitch (F0)...")
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        hop_length=hop_length,
    )
    stop.set(); t.join()

    stop, t = run_spinner("Separating harmonic / noise components...")
    y_harmonic, y_noise = librosa.effects.hpss(y)
    stop.set(); t.join()

    stop, t = run_spinner("Computing spectral and RMS features...")
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    rms_total    = librosa.feature.rms(y=y,          hop_length=hop_length)[0]
    rms_harmonic = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
    rms_noise    = librosa.feature.rms(y=y_noise,    hop_length=hop_length)[0]
    stop.set(); t.join()

    # Harmonic-to-Noise Ratio in dB (low HNR = scratchy / breathy)
    hnr_db = 10 * np.log10((rms_harmonic ** 2 + 1e-10) / (rms_noise ** 2 + 1e-10))

    n_frames   = len(rms_total)
    seg_frames = max(1, int(SEGMENT_DURATION * sr / hop_length))
    n_segments = int(np.ceil(n_frames / seg_frames))

    # ── Global baselines (voiced frames only) ────────────────────────────────
    voiced_mask = ~np.isnan(f0)
    f0_voiced   = f0[voiced_mask]
    hnr_voiced  = hnr_db[voiced_mask[:len(hnr_db)]]

    if len(f0_voiced) == 0:
        return "No voiced speech detected in audio."

    f0_global_mean  = float(np.mean(f0_voiced))
    f0_global_std   = float(np.std(f0_voiced))
    hnr_global_mean = float(np.mean(hnr_voiced))
    hnr_global_std  = float(np.std(hnr_voiced))
    rms_global_mean = float(np.mean(rms_total[rms_total > 0.001]))
    rms_global_std  = float(np.std(rms_total[rms_total > 0.001]))

    anomalies = []

    # ── Step 3: Scan segments ─────────────────────────────────────────────────
    for i in tqdm(range(n_segments), desc="Scanning segments", unit="seg", ncols=70):
        sf = i * seg_frames
        ef = min((i + 1) * seg_frames, n_frames)
        start_sec = sf * hop_length / sr
        end_sec   = min(ef * hop_length / sr, total_duration)

        seg_f0   = f0[sf:ef]
        seg_hnr  = hnr_db[sf:min(ef, len(hnr_db))]
        seg_flat = spectral_flatness[sf:min(ef, len(spectral_flatness))]
        seg_rms  = rms_total[sf:ef]

        voiced = seg_f0[~np.isnan(seg_f0)]
        if len(voiced) < 3:
            continue

        seg_f0_mean   = float(np.mean(voiced))
        seg_f0_std    = float(np.std(voiced))
        seg_hnr_mean  = float(np.mean(seg_hnr))
        seg_flat_mean = float(np.mean(seg_flat))
        seg_rms_mean  = float(np.mean(seg_rms))

        flags = []
        ts = fmt_ts(start_sec)

        # 1. Scratchiness: low HNR (+ high pitch check)
        hnr_z = (seg_hnr_mean - hnr_global_mean) / (hnr_global_std + 1e-6)
        is_high_pitch = seg_f0_mean > (f0_global_mean + 0.8 * f0_global_std)
        if hnr_z < -1.8:
            quality_tag = "at high pitch — likely AI voice artifact (scratchy)" if is_high_pitch else "voice quality drop"
            flags.append(
                f"Low HNR {seg_hnr_mean:.1f}dB (baseline {hnr_global_mean:.1f}dB, z={hnr_z:.2f}) — {quality_tag}"
            )

        # 2. High spectral flatness = noisy/unnatural texture
        flat_z = (seg_flat_mean - np.mean(spectral_flatness)) / (np.std(spectral_flatness) + 1e-6)
        if flat_z > 2.0:
            flags.append(f"High spectral flatness (z={flat_z:.2f}) — noisy/unnatural vocal texture")

        # 3. Pitch jitter / instability within segment
        if seg_f0_std > 0 and len(voiced) > 4:
            jitter_ratio = float(np.mean(np.abs(np.diff(voiced))) / (seg_f0_mean + 1e-6))
            if jitter_ratio > 0.08:
                flags.append(f"Pitch instability: jitter ratio {jitter_ratio:.3f} — unnatural pitch jumps")

        # 4. Sudden pitch spike vs. global mean
        f0_z = (seg_f0_mean - f0_global_mean) / (f0_global_std + 1e-6)
        if abs(f0_z) > 2.2:
            direction = "spike upward" if f0_z > 0 else "drop"
            flags.append(f"Pitch {direction}: {seg_f0_mean:.1f}Hz (baseline {f0_global_mean:.1f}Hz, z={f0_z:.2f})")

        # 5. Volume anomaly
        rms_z = (seg_rms_mean - rms_global_mean) / (rms_global_std + 1e-6)
        if rms_z < -2.5:
            flags.append(f"Volume drop (z={rms_z:.2f}) — sudden quiet, possible dropout")
        elif rms_z > 3.0:
            flags.append(f"Volume spike (z={rms_z:.2f}) — possible clipping or distortion")

        if flags:
            anomalies.append(f"{ts}–{fmt_ts(end_sec)}  " + " | ".join(flags))

    lines = [
        "ACOUSTIC QUALITY ANALYSIS",
        f"Total duration : {fmt_ts(total_duration)}",
        f"Pitch baseline : {f0_global_mean:.1f}Hz ± {f0_global_std:.1f}Hz",
        f"HNR baseline   : {hnr_global_mean:.1f}dB ± {hnr_global_std:.1f}dB",
        f"Segments analysed: {n_segments}  |  Anomalies found: {len(anomalies)}",
        "",
        "DETECTED QUALITY ISSUES:",
    ]
    lines += anomalies if anomalies else ["None detected."]
    return "\n".join(lines)


def upload_audio(path: str):
    stop, t = run_spinner("Uploading audio to Gemini File API...")
    audio_file = client.files.upload(file=path)
    stop.set(); t.join()

    with tqdm(desc="Waiting for file processing", unit=" poll", ncols=70, leave=False) as pbar:
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = client.files.get(name=audio_file.name)
            pbar.update(1)

    if audio_file.state.name == "FAILED":
        raise RuntimeError(f"Upload failed: {audio_file.state}")

    print(f"✓ File ready: {audio_file.uri}")
    return audio_file


def run_review():
    script_text     = load_script(SCRIPT_FILE_PATH)
    acoustic_report = extract_quality_features(AUDIO_FILE_PATH)
    audio_file      = upload_audio(AUDIO_FILE_PATH)

    prompt = f"""
You are a strict, objective Quality Control reviewer for children's audio content.
The lead character's voice was synthesised by an AI text-to-speech system.

You have been given:
1. The target script (with emotional stage directions).
2. The audio file.
3. A pre-computed acoustic quality report produced by librosa, listing timestamps where
   measurable degradation was detected (low HNR, pitch instability, spectral flatness spikes,
   volume anomalies). Use these as hard evidence — do not dismiss them.

Conduct a strict, line-for-line quality control review of the audio against the script.

Execution Directive: Evaluate the audio dynamically at both the macro-level (overall
structure/tone) and the micro-level (second-by-second delivery, micro-expressions, and
technical vocal shifts). Do not rely on generalized averages — if a 2-second window
contains a technical flaw, it must be explicitly exposed with a timestamp.

Output Format:
- For every parameter, output ONLY the "Issues Identified" bullet list under that heading.
- Do NOT write any preamble, analysis prose, or closing summaries — issues only.
- Use precise timestamps [MM:SS] on every bullet.
- Be specific: quote the script line, describe the flaw, state the timestamp.
- State "None" only if execution under that parameter is completely flawless.

---

**1. Kid Safe Content**
Assess the appropriateness of the material for a young audience.
- Does the overall tone, vocabulary, and intensity of the vocal delivery remain family-friendly?
- Does any portrayal of panic or distress cross the line into being genuinely frightening, or
  does it stay within a safe "Disney-style" adventure boundary?
- Flag any objectionable content or tonal over-intensity.

---

**2. Match to the Script**
Perform a line-by-line compliance check.
- Text Accuracy: Do characters speak every word exactly as written? Note any ad-libs,
  omitted words, or misread lines.
- SFX & Music Cues: Did the audio execute any requested sound design instructions?

---

**3. Consistency in Character Voice**
Evaluate vocal identity and continuity throughout the recording.
- Assess the vocal texture, accent, age profile, and timbre.
- Does the character sound like the exact same voice throughout the entire chapter?
- If singing segments are present, does the singing voice sound jarringly different or
  disconnected from the speaking persona, or is it a seamless vocal match?

---

**4. Emotion Mismatch (AI Prosody)**
- Does the AI voice's prosody match the emotional stage direction in the script?
- Flag any line where delivery is flat, over-intense, or tonally wrong for the moment.
- For a children's adventure story, emotions must be clear and expressive — neutral or flat
  delivery on an excited or fearful line is a fail.


---

**5. AI Voice Quality Degradation**
Cross-reference the acoustic report timestamps below as hard evidence.
- Scratchiness, raspiness, or breathiness — especially when pitch goes high
- Unnatural pitch jumps or instability mid-word or mid-sentence
- Robotic or synthetic texture breaking through
- Volume dropouts or distortion

For each: cite the acoustic report timestamp, describe what you hear, and state whether
it is distracting for a child listener.

---

ACOUSTIC QUALITY REPORT:
{acoustic_report}

---

TARGET SCRIPT:
{script_text}
"""

    stop, t = run_spinner("Sending to Gemini — this may take a minute...")
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type),
            types.Part.from_text(text=prompt),
        ],
    )
    stop.set(); t.join()

    print("\n" + "=" * 70)
    print("AI VOICE QUALITY REPORT")
    print("=" * 70)
    print(response.text)

    client.files.delete(name=audio_file.name)
    print("\n✓ Uploaded file deleted from server.")


if __name__ == "__main__":
    run_review()
