"""
Voice Quality Control — FastAPI backend.

Install:
    pip install fastapi "uvicorn[standard]" librosa google-genai numpy python-multipart
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000
"""

import asyncio
import json
import multiprocessing as _mp
import os
import queue as sync_queue
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from threading import Lock

import numpy as np
import librosa
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

# ── Constants ─────────────────────────────────────────────────────────────────

import secrets as _secrets
_SESSION_TOKEN = _secrets.token_hex(32)   # new token every server start
_APP_PASSWORD   = "admin"

SEGMENT_DURATION  = 3  # seconds per analysis window
SERVICE_ACCOUNT_PATH = Path(__file__).resolve().parent / "handy-compass-481307-i8-bdf6d538752a.json"
GOOGLE_SCOPES     = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

DEFAULT_PROMPT = """\
You are a strict, objective Quality Control reviewer for children's audio content.
The lead character's voice was synthesised by an AI text-to-speech system.

You have been given:
1. The target script (with emotional stage directions).
2. The audio file.
3. A pre-computed acoustic quality report produced by librosa, listing timestamps where
   measurable degradation was detected (low HNR, pitch instability, spectral flatness spikes,
   volume anomalies). Use the HNR (scratchiness) and spectral flatness flags as hard evidence. Treat pitch instability/jitter flags as low-confidence signals — only include them if you can actually hear the artifact in the audio.

Conduct a strict, line-for-line quality control review of the audio against the script.

MANDATORY: All 5 parameters below must be actively checked. You may not skip any parameter or output "None" without having completed the full procedure defined for that parameter. Passive listening is not sufficient — each parameter has a required checking procedure that you must follow.

Scope: The audio corresponds to a single chapter. Only evaluate script compliance for
lines you can actually hear within the audio's duration. Do not flag script lines from
other chapters as missing — if a line is absent, confirm it should fall within this
audio's runtime before flagging it.

{language_instruction}
Execution Directive: Evaluate the audio dynamically at both the macro-level (overall
structure/tone) and the micro-level (second-by-second delivery, micro-expressions, and
technical vocal shifts). Do not rely on generalized averages — if a 2-second window
contains a technical flaw, it must be explicitly exposed with a timestamp.

Output Format:
- For every parameter, output ONLY the "Issues Identified" bullet list under that heading.
- Do NOT write any preamble, analysis prose, or closing summaries — issues only.
- Use precise timestamps [MM:SS] on every bullet.
- Be specific: quote the script line, describe the flaw, state the timestamp.
- Only write "None" if you have completed all required checks for that parameter and found zero issues. If uncertain, flag it.
- If you determine mid-analysis that something is not a real issue (e.g. a punctuation difference), do not include it at all. Never output a bullet followed by "re-evaluating", "false flag", or "this is a punctuation difference" — simply omit it.

---

**1. Kid Safe Content**
Flag only content that genuinely crosses the line for a young audience — not normal adventure story tension.
- Flag truly age-inappropriate content: adult themes, offensive language, or graphic violence.
- Flag vocal delivery that would genuinely frighten a young child beyond what a typical Disney/Pixar-style adventure contains.
- Do NOT flag: villain menace, protagonists showing fear or determination, battle speeches, or any dramatic tone that is standard in children's adventure stories. These are expected and intentional.
- The bar is high: if the same scene could appear in a mainstream children's animated series without concern, do not flag it.

---

**2. Match to the Script**
Compare every spoken word in the audio against the script. Focus only on what can be heard — punctuation is not spoken and must never be flagged.

NOT issues (do not flag these):
- Punctuation differences: ellipses (…) vs periods (.), commas, dashes, em-dashes, quotation marks around words — these are not spoken.
- Contractions vs expanded form: "we're" / "we are", "it's" / "it is", "they're" / "they are", "I'm" / "I am", "don't" / "do not" — these mean the same thing.
- Capitalisation differences.
- Script section headers or scene labels (e.g. "TEST 1 – T-MAX vs RUTHLESS LEADER", "SCENE 5 – FINAL CLUE") — these are structural markers in the script, not spoken lines.
- Stage directions and action descriptions (e.g. "FLAPPY grabs the crystal", "exits the room") — these are not spoken.
- Duplicate script lines — if the same line appears twice in the script and is spoken once in the audio, do not flag the second occurrence as missing.

REAL issues (flag these):
- A different word spoken in place of the script word (e.g. "Team" said as "D"). Format: - [MM:SS] SCRIPT: "<exact script text>" / HEARD: "<what was said>" — substituted word
- A word present in the script but not spoken at all. Format: - [MM:SS] SCRIPT: "<exact script text>" / HEARD: "<what was said>" — omitted word
- A word spoken that does not appear in the script at that point. Format: - [MM:SS] SCRIPT: "<exact script text>" / HEARD: "<what was said>" — added word
- A script line entirely absent from the audio. Format: - [MM:SS] MISSING LINE: "<script text>"
- Lines spoken out of the sequence written in the script. Format: - [MM:SS] OUT OF ORDER: "<script text>"
- Any SFX or music cue in the script that was not executed. Format: - [MM:SS] MISSING SFX: <description>

---

**3. Consistency in Character Voice**
Every character with more than one line must be individually tracked. Listen to each character's very first line and lock in their voice profile: accent, timbre, vocal age, pitch range, and speech rhythm. Then every subsequent line from that character must match that locked profile exactly.

This check is not optional and cannot be skipped. Do not output "None" without actively comparing each character's lines against their own profile.

Important exclusions:
- Do NOT compare a Fake/Clone/Shadow version of a character against the original — they are intentionally different characters with different voices.
- Do NOT flag pitch variation caused by emotion (e.g. a character sounding higher when shocked vs. calm). Emotional delivery naturally changes pitch and energy. Only flag genuine voice drift: where the base vocal identity — accent, timbre, vocal age, or recording quality — has shifted independent of the emotion being performed.

Flag only genuine voice drift where the underlying vocal identity has changed:
- Format: - [MM:SS] VOICE INCONSISTENCY — <Character Name>: <describe the change, e.g. accent shifted, timbre became noticeably darker, sounds like a different TTS batch>

Common real causes: accent drift between lines, recording quality change mid-chapter (different background noise, different mic quality), vocal age sounding older or younger than earlier lines, timbre change that persists across multiple lines.

Also check cross-character similarity: if two distinct named characters (e.g. Narrator and a villain) sound indistinguishably similar — same timbre, same accent, same vocal age — flag it. Different characters must have clearly distinguishable voices.
- Format: - [MM:SS] CROSS-CHARACTER SIMILARITY — <Character A> and <Character B>: <describe why they sound alike>

If singing is present, flag if any character's singing voice sounds disconnected from their speaking voice.

---

**4. Emotion Mismatch (AI Prosody)**
Check every line that has an emotional stage direction. Only flag clear, obvious failures — not minor or subtle variations.

The bar for a flag: the emotion delivered must be plainly wrong — clearly missing or clearly replaced by a different emotion. If you need words like "slightly", "somewhat", "almost", "not fully", or "borderline" to describe the mismatch, it is NOT a flag.

- Flag flat/neutral delivery on a line with a strong emotional direction (e.g. a line marked "panicking" delivered completely without panic).
- Flag the wrong emotion entirely (e.g. a line marked "angry" delivered as cheerful).
- Flag over-intensity where the script specifically calls for restraint (e.g. a line marked "whisper" delivered loudly).
- Do NOT flag minor imperfections in degree (e.g. "not quite as annoyed as the script implies" or "slightly less enthusiastic").

---

**5. AI Voice Quality Degradation**
Cross-reference the acoustic report timestamps as hard evidence, and also flag anything you can hear that the report may have missed.
- Pitch spikes: sudden unnatural jumps in pitch mid-word or mid-sentence.
- Robotic wobble: synthetic, mechanical, or unstable vocal texture.
- Spectral artifacts: buzzing, distortion, or unnatural tonal coloring in the voice.
- Static or hum: background noise, hiss, or electrical hum audible under the voice.
- Scratchiness, raspiness, or breathiness — especially at high pitch.
- Volume dropouts: sudden loss of volume or silence mid-word.
- Word truncation: a word cut off abruptly before completing its final syllable — the ending of the word is missing or clipped.
- Mispronunciation: a word, character name, or script-specific term spoken with clearly incorrect pronunciation.

For each issue: cite the acoustic report timestamp if available, describe exactly what you hear, and state whether it is distracting for a child listener.

---

ACOUSTIC QUALITY REPORT:
{acoustic_report}

---

TARGET SCRIPT:
{script_text}\
"""

# ── Language instructions ─────────────────────────────────────────────────────

_LANGUAGE_INSTRUCTIONS = {
    "english": "Language: The script is in English.\n",
    "hindi": (
        "Language: The script is in Hindi (Devanagari or Hinglish romanisation).\n"
        "Hindi-specific Script Match rules:\n"
        "- Ellipsis (…) used within a spoken line as a pause or beat marker is NOT a word — do not flag 'omitted pause' under Script Match. If a dramatic pause is missing from the delivery, note it under Emotion Mismatch instead. However, if an actual content word differs within the same line (e.g. 'कर' spoken where the script says 'लड़'), still flag that word substitution — the ellipsis rule does not suppress real word-level differences.\n"
        "- The script may contain production/editorial notes (lines starting with 'Note:' or similar comments from the editor). These are not spoken content — do not flag them as missing lines.\n"
        "- A spoken stutter written in the script (e.g. 'ब-बहुत' or 'I-I') IS spoken content. Flag it as an omitted word if the stutter is not present in the audio.\n"
        "- Question intonation: if the script marks a line as a question (e.g. ends with '?') but the audio delivers it as a flat statement with no rising intonation, flag this under Emotion Mismatch, not Script Match — the words are the same.\n"
        "- Romanised Hindi and Devanagari representations of the same word are the same — do not flag a romanised hearing of a Devanagari script word as a substitution.\n"
        "Hindi vocabulary accessibility (Kid Safe Content): Flag any instance where a highly formal or literary Hindi word is used in the audio where a widely-recognised English loanword would be more natural and understandable for a child audience (e.g. a formal Sanskrit-derived Hindi word used instead of 'distract', 'order', 'team', 'mission'). This is rare — only flag when the word is genuinely likely to be unfamiliar to a young child. Format: - [MM:SS] VOCABULARY: '<word spoken>' — suggest '<simpler alternative>' for child comprehension.\n"
    ),
}


def _build_language_instruction(language: str) -> str:
    return _LANGUAGE_INSTRUCTIONS.get(language.lower().strip(), "")


# ── Prompt store (prompts.json) ───────────────────────────────────────────────

PROMPTS_PATH  = Path(__file__).resolve().parent / "prompts.json"
_prompts_lock = Lock()


def _read_prompts() -> dict:
    if not PROMPTS_PATH.exists():
        data = {"Default": DEFAULT_PROMPT}
    else:
        data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    data["Default"] = DEFAULT_PROMPT
    PROMPTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def _write_prompts(data: dict) -> None:
    PROMPTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── In-memory job store ───────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_stop_events: dict[str, threading.Event] = {}

# ── Analysis helpers ──────────────────────────────────────────────────────────


def fmt_ts(seconds: float) -> str:
    return f"[{int(seconds // 60):02d}:{int(seconds % 60):02d}]"


def _acoustic_worker(audio_path: str, result_queue: "_mp.Queue") -> None:
    """Subprocess target — runs librosa/numba in its own GIL so the event loop stays free."""
    try:
        def _noop(*_): pass
        result = extract_quality_features(audio_path, _noop, _noop)
        result_queue.put(("ok", result))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def extract_quality_features(audio_path: str, log_fn, progress_fn, stop: threading.Event | None = None) -> str:
    if stop is not None:
        # librosa.pyin uses numba which holds the GIL and freezes the event loop.
        # Running in a subprocess gives it an independent GIL.
        ctx = _mp.get_context("spawn")
        result_q = ctx.Queue()
        proc = ctx.Process(target=_acoustic_worker, args=(audio_path, result_q))
        proc.start()
        log_fn("Running acoustic analysis...")
        while proc.is_alive():
            if stop.is_set():
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
                return ""
            time.sleep(0.5)
        proc.join()
        try:
            tag, data = result_q.get_nowait()
        except sync_queue.Empty:
            raise RuntimeError("Acoustic subprocess returned no result.")
        if tag == "ok":
            return data
        raise RuntimeError(data)

    def _stopped() -> bool:
        return stop is not None and stop.is_set()

    log_fn("Loading audio file...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_duration = librosa.get_duration(y=y, sr=sr)
    log_fn(f"  Duration: {fmt_ts(total_duration)}")

    hop_length = 512

    log_fn("Extracting pitch (F0)...")
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        hop_length=hop_length,
        frame_length=1024,
    )
    if _stopped():
        return ""

    log_fn("Separating harmonic / noise components...")
    y_harmonic, y_noise = librosa.effects.hpss(y)
    if _stopped():
        return ""

    log_fn("Computing spectral and RMS features...")
    D = np.abs(librosa.stft(y, hop_length=hop_length))
    spectral_flatness = librosa.feature.spectral_flatness(S=D)[0]
    rms_total    = librosa.feature.rms(S=D)[0]
    rms_harmonic = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
    rms_noise    = librosa.feature.rms(y=y_noise,    hop_length=hop_length)[0]

    hnr_db = 10 * np.log10((rms_harmonic ** 2 + 1e-10) / (rms_noise ** 2 + 1e-10))

    n_frames   = len(rms_total)
    seg_frames = max(1, int(SEGMENT_DURATION * sr / hop_length))
    n_segments = int(np.ceil(n_frames / seg_frames))

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

    anomalies     = []
    last_pct_sent = -1

    log_fn(f"Scanning {n_segments} segments...")
    for i in range(n_segments):
        if _stopped():
            return ""
        frac = (i + 1) / n_segments
        pct  = int(frac * 100)
        if pct != last_pct_sent:
            progress_fn(frac)
            last_pct_sent = pct

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
        seg_hnr_mean  = float(np.mean(seg_hnr))
        seg_flat_mean = float(np.mean(seg_flat))
        seg_rms_mean  = float(np.mean(seg_rms))

        flags = []
        ts    = fmt_ts(start_sec)

        hnr_z         = (seg_hnr_mean - hnr_global_mean) / (hnr_global_std + 1e-6)
        is_high_pitch = seg_f0_mean > (f0_global_mean + 0.8 * f0_global_std)
        if hnr_z < -1.8:
            quality_tag = (
                "at high pitch — likely AI voice artifact (scratchy)"
                if is_high_pitch
                else "voice quality drop"
            )
            flags.append(
                f"Low HNR {seg_hnr_mean:.1f}dB "
                f"(baseline {hnr_global_mean:.1f}dB, z={hnr_z:.2f}) — {quality_tag}"
            )

        flat_z = (seg_flat_mean - np.mean(spectral_flatness)) / (np.std(spectral_flatness) + 1e-6)
        if flat_z > 2.0:
            flags.append(f"High spectral flatness (z={flat_z:.2f}) — noisy/unnatural vocal texture")

        if len(voiced) > 4:
            jitter_ratio = float(np.mean(np.abs(np.diff(voiced))) / (seg_f0_mean + 1e-6))
            if jitter_ratio > 0.10:
                flags.append(
                    f"Pitch instability: jitter ratio {jitter_ratio:.3f} — unnatural pitch jumps"
                )

        f0_z = (seg_f0_mean - f0_global_mean) / (f0_global_std + 1e-6)
        if abs(f0_z) > 2.2:
            direction = "spike upward" if f0_z > 0 else "drop"
            flags.append(
                f"Pitch {direction}: {seg_f0_mean:.1f}Hz "
                f"(baseline {f0_global_mean:.1f}Hz, z={f0_z:.2f})"
            )

        rms_z = (seg_rms_mean - rms_global_mean) / (rms_global_std + 1e-6)
        if rms_z < -2.5:
            flags.append(f"Volume drop (z={rms_z:.2f}) — sudden quiet, possible dropout")
        elif rms_z > 3.0:
            flags.append(f"Volume spike (z={rms_z:.2f}) — possible clipping or distortion")

        if flags:
            anomalies.append(f"{ts}-{fmt_ts(end_sec)}  " + " | ".join(flags))

    log_fn(f"  Scan complete — {len(anomalies)} anomaly segment(s) found.")

    lines = [
        "ACOUSTIC QUALITY ANALYSIS",
        f"Total duration   : {fmt_ts(total_duration)}",
        f"Pitch baseline   : {f0_global_mean:.1f}Hz +- {f0_global_std:.1f}Hz",
        f"HNR baseline     : {hnr_global_mean:.1f}dB +- {hnr_global_std:.1f}dB",
        f"Segments scanned : {n_segments}  |  Anomalies: {len(anomalies)}",
        "",
        "DETECTED QUALITY ISSUES:",
    ]
    lines += anomalies if anomalies else ["None detected."]
    return "\n".join(lines)


def upload_to_gemini(audio_path: str, log_fn, client) -> object:
    log_fn("Uploading audio to Gemini File API...")
    audio_file = client.files.upload(file=audio_path)

    polls = 0
    while audio_file.state.name == "PROCESSING":
        time.sleep(2)
        audio_file = client.files.get(name=audio_file.name)
        polls += 1
        log_fn(f"  Waiting for Gemini to process file... (poll {polls})")

    if audio_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini file upload failed: {audio_file.state}")

    log_fn("  File ready.")
    return audio_file


def transcribe_audio(audio_ref, client, model_id: str, log_fn) -> str:
    log_fn("Transcribing audio for script comparison...")
    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_uri(
                file_uri=audio_ref.uri,
                mime_type=audio_ref.mime_type,
            ),
            types.Part.from_text(text=(
                "Transcribe every single spoken word in this audio exactly as heard, in the order it is spoken.\n"
                "Format each line as: [MM:SS] <spoken text>\n"
                "Rules:\n"
                "- Include every line without exception.\n"
                "- Do NOT paraphrase, summarize, or correct grammar.\n"
                "- Do NOT include stage directions, sound effect descriptions, or speaker labels.\n"
                "- If a word is unclear, write your best guess and mark it (unclear).\n"
                "- Output the transcript only — no preamble, no commentary."
            )),
        ],
    )
    log_fn("  Transcript complete.")
    return response.text


# ── Single-file analysis ──────────────────────────────────────────────────────


def run_analysis(job: dict, q: sync_queue.Queue) -> None:
    audio_path   = job["audio_path"]
    script_path  = job["script_path"]
    stop: threading.Event = job["stop"]

    def log(msg: str) -> None:
        print(f"[VQC] {msg}", flush=True)
        q.put(("log", msg))

    def progress(frac: float) -> None:
        pct = int(frac * 100)
        if pct % 25 == 0:
            print(f"[VQC]   segment scan: {pct}%", flush=True)
        q.put(("progress", frac))

    try:
        api_key = job["api_key"]
        client  = genai.Client(api_key=api_key) if api_key else genai.Client()
        model_id = job["model"]
        log(f"Starting analysis  |  model: {model_id}")

        script_text = open(script_path, encoding="utf-8").read()
        if stop.is_set(): return
        acoustic_report = extract_quality_features(audio_path, log, progress, stop)
        if stop.is_set(): return
        audio_ref = upload_to_gemini(audio_path, log, client)

        try:
            final_prompt = job["prompt"].format(
                acoustic_report=acoustic_report,
                script_text=script_text,
                language_instruction=_build_language_instruction(job.get("language", "english")),
            )
        except KeyError as exc:
            raise ValueError(
                f"Prompt is missing the placeholder {{{exc}}}. "
                "Both {acoustic_report} and {script_text} must be present."
            ) from exc

        log(f"Calling {model_id} — this may take a minute...")
        response = client.models.generate_content(
            model=model_id,
            contents=[
                types.Part.from_uri(
                    file_uri=audio_ref.uri,
                    mime_type=audio_ref.mime_type,
                ),
                types.Part.from_text(text=final_prompt),
            ],
        )

        client.files.delete(name=audio_ref.name)
        log("Complete. Uploaded file deleted from Gemini server.")
        q.put(("result", response.text))

    except Exception as exc:
        q.put(("apperror", f"{type(exc).__name__}: {exc}"))
    finally:
        q.put(("done", None))
        for p in (audio_path, script_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ── Google Drive / Sheets helpers ─────────────────────────────────────────────


def _get_google_creds():
    from google.oauth2 import service_account as _sa
    return _sa.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_PATH), scopes=GOOGLE_SCOPES
    )


def _extract_drive_id(url: str, kind: str) -> str:
    patterns = {
        "folder": r"/folders/([a-zA-Z0-9_-]+)",
        "doc":    r"/document/d/([a-zA-Z0-9_-]+)",
        "sheet":  r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
    }
    m = re.search(patterns[kind], url)
    if not m:
        raise ValueError(f"Could not parse {kind} ID from URL: {url}")
    return m.group(1)


def _list_chapter_files(drive, folder_id: str) -> list:
    audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".mp4"}
    result = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,mimeType)",
        orderBy="name",
    ).execute()
    return [
        f for f in result.get("files", [])
        if Path(f["name"]).suffix.lower() in audio_exts
    ]


def _download_drive_file(drive, file_id: str, suffix: str) -> str:
    import io
    from googleapiclient.http import MediaIoBaseDownload
    req  = drive.files().get_media(fileId=file_id)
    buf  = io.BytesIO()
    dl   = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(buf.read())
        return tmp.name


def _export_google_doc(drive, doc_id: str) -> str:
    data = drive.files().export(fileId=doc_id, mimeType="text/plain").execute()
    return data.decode("utf-8") if isinstance(data, bytes) else data


def _ensure_sheet_tab(sheets, spreadsheet_id: str, sheet_name: str, headers: list = None) -> None:
    meta     = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    existing = {s["properties"]["title"] for s in meta["sheets"]}
    if sheet_name not in existing:
        sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": sheet_name}}}]},
        ).execute()
        row = headers or ["Story", "Chapter", "Category", "Timestamp", "Issue", "Model", "Date"]
        sheets.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=f"'{sheet_name}'!A1",
            valueInputOption="RAW",
            body={"values": [row]},
        ).execute()


def _append_sheet_rows(sheets, spreadsheet_id: str, sheet_name: str, rows: list) -> None:
    if not rows:
        return
    sheets.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"'{sheet_name}'!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": rows},
    ).execute()


_CATEGORY_KEYS = {
    "Kid Safe Content":         "Kid Safe Content",
    "Match to the Script":      "Script Match",
    "Consistency in Character": "Voice Consistency",
    "Emotion Mismatch":         "Emotion Mismatch",
    "AI Voice Quality":         "Quality Degradation",
}


def _parse_report_rows(report_text: str, story: str, chapter: str, model: str) -> list:
    from datetime import date as _date
    rows     = []
    cat      = None
    run_date = _date.today().isoformat()

    for line in report_text.splitlines():
        s = line.strip()
        # Detect category headings — tolerate presence or absence of ** markers
        for key, label in _CATEGORY_KEYS.items():
            if key in s:
                cat = label
                break
        # Accept both "- " and "• " bullet styles
        if (s.startswith("- ") or s.startswith("• ")) and cat:
            content = s[2:].strip()
            cl = content.lower().rstrip(".")
            if not cl or cl == "none" or cl.startswith("issues identified"):
                continue
            ts_m      = re.search(r"\[(\d{1,2}:\d{2})\]", content)
            timestamp = ts_m.group(1) if ts_m else ""
            rows.append([story, chapter, cat, timestamp, content, model, run_date])

    return rows


# ── Scoring ───────────────────────────────────────────────────────────────────

_CATEGORY_WEIGHTS = {
    "Kid Safe Content":    1.0,
    "Script Match":        1.0,
    "Voice Consistency":   1.0,
    "Emotion Mismatch":    1.0,
    "Quality Degradation": 0.25,
}
_TOTAL_WEIGHT      = sum(_CATEGORY_WEIGHTS.values())  # 4.25
_PENALTY_PER_ISSUE = 4   # 25 normalized weighted issues → score 0
_PITCH_KEYWORDS    = {"jitter", "pitch instability", "pitch spike", "pitch drop"}


def _issue_relevance(issue_text: str, category: str) -> float:
    if category == "Quality Degradation":
        if any(kw in issue_text.lower() for kw in _PITCH_KEYWORDS):
            return 0.5
    return 1.0


def _compute_score(rows: list) -> dict:
    cat_counts = {cat: 0.0 for cat in _CATEGORY_WEIGHTS}
    for row in rows:
        category   = row[2]
        issue_text = row[4]
        if category in _CATEGORY_WEIGHTS:
            cat_counts[category] += _issue_relevance(issue_text, category)
    weighted_sum = sum(cat_counts[cat] * _CATEGORY_WEIGHTS[cat] for cat in _CATEGORY_WEIGHTS)
    normalized   = weighted_sum / _TOTAL_WEIGHT
    score        = max(0.0, 100.0 - normalized * _PENALTY_PER_ISSUE)
    return {"score": round(score, 1), "cat_counts": cat_counts}


def _story_chapter_from_filename(filename: str):
    stem = Path(filename).stem
    m    = re.search(r"(.+?)[-\s]+(CHAPTER\s+\d+)$", stem, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip().upper()
    return stem, stem


def _chapter_num_from_filename(filename: str) -> int | None:
    m = re.search(r"CHAPTER\s+(\d+)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _split_script_by_chapter(script_text: str) -> dict:
    """Split a full script into per-chapter sections keyed by chapter number.

    Works with both plain-text Google Docs exports and markdown files.
    Matches headings like:
      CHAPTER 1 — "TITLE"        (plain text export)
      # **CHAPTER 1 — "TITLE"**  (markdown)

    If content exists before the first heading (e.g. the script begins directly
    with chapter 1 content without an explicit heading), it is assigned to the
    chapter number immediately before the first matched heading number.

    Returns {1: "<chapter 1 text>", 2: "<chapter 2 text>", ...}
    Returns empty dict if no chapter headings are found at all.
    """
    pattern = re.compile(
        r"^[ \t]*(?:#[ \t]+)?(?:\*+)?CHAPTER\s+(\d+)",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(pattern.finditer(script_text))
    if not matches:
        return {}

    chapters = {}

    preamble = script_text[: matches[0].start()].strip()
    if preamble:
        first_num = int(matches[0].group(1))
        orphan_num = max(1, first_num - 1)
        chapters[orphan_num] = preamble

    for i, m in enumerate(matches):
        num   = int(m.group(1))
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(script_text)
        chapters[num] = script_text[start:end].strip()

    return chapters


# ── Batch analysis ────────────────────────────────────────────────────────────


def run_batch_analysis(job: dict, q: sync_queue.Queue) -> None:
    stop: threading.Event = job["stop"]

    def log(msg: str) -> None:
        print(f"[VQC-BATCH] {msg}", flush=True)
        q.put(("log", msg))

    def progress(frac: float) -> None:
        pct = int(frac * 100)
        if pct % 25 == 0:
            print(f"[VQC-BATCH]   {pct}%", flush=True)
        q.put(("progress", frac))

    try:
        from googleapiclient.discovery import build as _build

        creds  = _get_google_creds()
        drive  = _build("drive",  "v3", credentials=creds)
        sheets = _build("sheets", "v4", credentials=creds)

        folder_id        = _extract_drive_id(job["folder_url"],         "folder")
        doc_id           = _extract_drive_id(job["doc_url"],            "doc")
        sheet_id         = _extract_drive_id(job["sheet_url"],          "sheet")
        scoring_sheet_id = _extract_drive_id(job["scoring_sheet_url"],  "sheet")
        model_id         = job["model"]
        prompt_tmpl      = job["prompt"]
        story_override   = job.get("story_name", "").strip()

        log("Listing audio files in Drive folder...")
        chapters = _list_chapter_files(drive, folder_id)
        if not chapters:
            raise ValueError("No audio files found in the Drive folder.")
        log(f"  Found {len(chapters)} file(s): {', '.join(f['name'] for f in chapters)}")

        log("Downloading script from Google Docs...")
        doc_meta       = drive.files().get(fileId=doc_id, fields="name").execute()
        doc_name       = doc_meta.get("name", "").strip()
        script_text    = _export_google_doc(drive, doc_id)
        chapter_scripts = _split_script_by_chapter(script_text)
        if chapter_scripts:
            log(f"  Script: {doc_name} ({len(script_text):,} chars) — split into chapters: {sorted(chapter_scripts.keys())}")
        else:
            log(f"  Script: {doc_name} ({len(script_text):,} chars) — WARNING: no chapter headings found, full script will be used for all chapters")

        story_name = story_override or doc_name or _story_chapter_from_filename(chapters[0]["name"])[0]

        _ensure_sheet_tab(sheets, sheet_id, story_name)
        _ensure_sheet_tab(sheets, scoring_sheet_id, story_name, headers=[
            "Date", "Story", "Chapter", "Score",
            "Kid Safe Content", "Script Match", "Voice Consistency",
            "Emotion Mismatch", "Quality Degradation", "Total Issues", "Model",
        ])

        client   = genai.Client()
        total    = len(chapters)
        all_rows = []

        for idx, ch_meta in enumerate(chapters):
            if stop.is_set():
                log("Client disconnected — stopping batch.")
                break

            _, chapter_name = _story_chapter_from_filename(ch_meta["name"])
            log(f"[{idx+1}/{total}] {chapter_name} — downloading audio...")

            suffix     = Path(ch_meta["name"]).suffix or ".mp3"
            audio_path = _download_drive_file(drive, ch_meta["id"], suffix)

            try:
                log(f"[{idx+1}/{total}] {chapter_name} — acoustic analysis...")

                off = idx / total
                sc  = 1.0 / total

                def make_prog(off=off, sc=sc):
                    def _p(frac):
                        progress(off + frac * sc)
                    return _p

                acoustic_report = extract_quality_features(audio_path, log, make_prog(), stop)
                if stop.is_set():
                    log("Client disconnected — stopping batch.")
                    break

                log(f"[{idx+1}/{total}] {chapter_name} — uploading to Gemini...")
                audio_ref = upload_to_gemini(audio_path, log, client)

                ch_num        = _chapter_num_from_filename(ch_meta["name"])
                ch_script     = chapter_scripts.get(ch_num, script_text)
                if ch_num and ch_num in chapter_scripts:
                    log(f"[{idx+1}/{total}] {chapter_name} — using script section for chapter {ch_num}")
                else:
                    log(f"[{idx+1}/{total}] {chapter_name} — no matching chapter section found, using full script")

                try:
                    final_prompt = prompt_tmpl.format(
                        acoustic_report=acoustic_report,
                        script_text=ch_script,
                        language_instruction=_build_language_instruction(job.get("language", "english")),
                    )
                except KeyError as exc:
                    raise ValueError(f"Prompt missing placeholder: {{{exc}}}")

                log(f"[{idx+1}/{total}] {chapter_name} — calling {model_id}...")
                response = client.models.generate_content(
                    model=model_id,
                    contents=[
                        types.Part.from_uri(
                            file_uri=audio_ref.uri,
                            mime_type=audio_ref.mime_type,
                        ),
                        types.Part.from_text(text=final_prompt),
                    ],
                )
                client.files.delete(name=audio_ref.name)

                rows = _parse_report_rows(response.text, story_name, chapter_name, model_id)
                _append_sheet_rows(sheets, sheet_id, story_name, rows)
                all_rows.extend(rows)

                from datetime import date as _date
                scored   = _compute_score(rows)
                run_date = _date.today().isoformat()
                cc       = scored["cat_counts"]
                _append_sheet_rows(sheets, scoring_sheet_id, story_name, [[
                    run_date, story_name, chapter_name, scored["score"],
                    round(cc["Kid Safe Content"], 2),
                    round(cc["Script Match"], 2),
                    round(cc["Voice Consistency"], 2),
                    round(cc["Emotion Mismatch"], 2),
                    round(cc["Quality Degradation"], 2),
                    len(rows), model_id,
                ]])

                log(f"[{idx+1}/{total}] {chapter_name} — {len(rows)} issue(s), score: {scored['score']}")
                q.put(("chapter_done", {
                    "chapter": chapter_name,
                    "issues":  len(rows),
                    "score":   scored["score"],
                }))

            finally:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

        progress(1.0)
        log(f"Batch complete — {total} chapter(s), {len(all_rows)} total issue(s).")
        q.put(("result", {
            "total_chapters": total,
            "total_issues":   len(all_rows),
            "sheet_url":      f"https://docs.google.com/spreadsheets/d/{sheet_id}",
            "story":          story_name,
        }))

    except Exception as exc:
        q.put(("apperror", f"{type(exc).__name__}: {exc}"))
    finally:
        q.put(("done", None))


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Voice QC")


# ── Prompt CRUD ───────────────────────────────────────────────────────────────

@app.get("/prompts")
def get_prompts():
    with _prompts_lock:
        return _read_prompts()


@app.post("/prompts")
async def save_prompt(name: str = Form(...), content: str = Form(...)):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    with _prompts_lock:
        data = _read_prompts()
        data[name.strip()] = content
        _write_prompts(data)
    return {"status": "ok"}


@app.delete("/prompts/{name}")
def delete_prompt(name: str):
    with _prompts_lock:
        data = _read_prompts()
        if name not in data:
            raise HTTPException(status_code=404, detail="Prompt not found.")
        del data[name]
        _write_prompts(data)
    return {"status": "ok"}


# ── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/auth/login")
async def auth_login(password: str = Form(...)):
    if password != _APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Incorrect password.")
    return {"token": _SESSION_TOKEN}


@app.get("/auth/check")
def auth_check(token: str = ""):
    return {"valid": token == _SESSION_TOKEN}


# ── Credentials status ────────────────────────────────────────────────────────

@app.get("/credentials/status")
def credentials_status():
    return {"configured": SERVICE_ACCOUNT_PATH.exists()}


# ── Single-file job upload + stream ──────────────────────────────────────────

@app.post("/upload")
async def upload_files(
    script:   UploadFile = File(...),
    audio:    UploadFile = File(...),
    model:    str        = Form(...),
    prompt:   str        = Form(...),
    language: str        = Form("english"),
    api_key:  str        = Form(""),
):
    audio_suffix = Path(audio.filename or "audio.mp3").suffix or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp_a:
        tmp_a.write(await audio.read())
        audio_path = tmp_a.name

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".md", mode="w", encoding="utf-8"
    ) as tmp_s:
        tmp_s.write((await script.read()).decode("utf-8"))
        script_path = tmp_s.name

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "audio_path":  audio_path,
        "script_path": script_path,
        "model":       model,
        "prompt":      prompt,
        "language":    language,
        "api_key":     api_key.strip() or None,
        "stop":        threading.Event(),
    }
    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream_analysis(job_id: str, request: Request):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = _jobs.pop(job_id)
    stop: threading.Event = job["stop"]
    _stop_events[job_id] = stop
    q: sync_queue.Queue = sync_queue.Queue()
    threading.Thread(target=run_analysis, args=(job, q), daemon=True).start()

    async def event_stream():
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    event_type, data = await loop.run_in_executor(None, q.get, True, 1.0)
                    if event_type == "done":
                        yield "event: done\ndata: {}\n\n"
                        break
                    elif event_type == "apperror":
                        yield f"event: apperror\ndata: {json.dumps({'data': data})}\n\n"
                        yield "event: done\ndata: {}\n\n"
                        break
                    else:
                        yield f"event: {event_type}\ndata: {json.dumps({'data': data})}\n\n"
                except sync_queue.Empty:
                    if await request.is_disconnected():
                        print(f"[VQC] Client disconnected — job {job_id} stopped.", flush=True)
                        break
                    yield ": heartbeat\n\n"
        finally:
            ev = _stop_events.pop(job_id, None)
            if ev:
                ev.set()
                print(f"[VQC] Stream ended for {job_id} — job stopped.", flush=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Batch job upload + stream ─────────────────────────────────────────────────

@app.post("/batch/upload")
async def batch_upload(
    folder_url:        str = Form(...),
    doc_url:           str = Form(...),
    sheet_url:         str = Form(...),
    scoring_sheet_url: str = Form(...),
    story_name:        str = Form(""),
    model:             str = Form(...),
    prompt:            str = Form(...),
    language:          str = Form("english"),
):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "type":              "batch",
        "folder_url":        folder_url,
        "doc_url":           doc_url,
        "sheet_url":         sheet_url,
        "scoring_sheet_url": scoring_sheet_url,
        "story_name":        story_name,
        "model":             model,
        "prompt":            prompt,
        "language":          language,
        "stop":              threading.Event(),
    }
    return {"job_id": job_id}


@app.get("/batch/stream/{job_id}")
async def batch_stream(job_id: str, request: Request):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = _jobs.pop(job_id)
    stop: threading.Event = job["stop"]
    _stop_events[job_id] = stop
    q: sync_queue.Queue = sync_queue.Queue()
    threading.Thread(target=run_batch_analysis, args=(job, q), daemon=True).start()

    async def event_stream():
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    event_type, data = await loop.run_in_executor(None, q.get, True, 1.0)
                    if event_type == "done":
                        yield "event: done\ndata: {}\n\n"
                        break
                    elif event_type == "apperror":
                        yield f"event: apperror\ndata: {json.dumps({'data': data})}\n\n"
                        yield "event: done\ndata: {}\n\n"
                        break
                    else:
                        yield f"event: {event_type}\ndata: {json.dumps({'data': data})}\n\n"
                except sync_queue.Empty:
                    if await request.is_disconnected():
                        print(f"[VQC-BATCH] Client disconnected — job {job_id} stopped.", flush=True)
                        break
                    yield ": heartbeat\n\n"
        finally:
            ev = _stop_events.pop(job_id, None)
            if ev:
                ev.set()
                print(f"[VQC-BATCH] Stream ended for {job_id} — job stopped.", flush=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    ev = _stop_events.get(job_id)
    if ev:
        ev.set()
        print(f"[VQC] Job {job_id} cancelled by client.", flush=True)
    return {"cancelled": job_id}


# Must be declared last — catches all routes not matched above
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True),
    name="static",
)
