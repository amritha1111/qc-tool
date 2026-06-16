# Content Reviewer

A web-based QC tool for children's audiobooks. Analyses audio against a script using Google Gemini and a local acoustic pipeline (librosa), then writes issues and scores to Google Sheets.

---

## Requirements

- Python 3.10+
- A Google Cloud service account JSON file with access to Drive and Sheets
- A Gemini API key (set as `GOOGLE_API_KEY` in your environment, or the service account is used automatically)

---

## Setup

### 1. Install dependencies

```bash
pip install fastapi "uvicorn[standard]" python-multipart google-genai numpy librosa
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### 2. Add service account credentials

Place your Google Cloud service account JSON file in the `ContentQC` folder. The filename must match the `SERVICE_ACCOUNT_PATH` constant in `server.py`:

```
ContentQC/handy-compass-481307-i8-bdf6d538752a.json
```

Share your Google Drive folder, Google Doc, and Google Sheets with the service account's `client_email`.

### 3. Run the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Login

Password: `admin`

The session persists across browser tabs. Restarting the server invalidates all sessions and requires logging in again.

---

## Batch Review

Processes an entire folder of chapter audio files against a single Google Doc script.

| Field | Description |
|---|---|
| Drive Folder URL | Google Drive folder containing chapter audio files (mp3, wav, flac, etc.) |
| Google Doc URL | Direct link to the script Google Doc |
| Issues Sheet URL | Google Sheets spreadsheet where per-issue rows are written |
| Scoring Sheet URL | Google Sheets spreadsheet where one score row per chapter is written |
| Story Name | Optional — auto-detected from filename if left blank |
| Language | English or Hindi (affects script matching strictness) |
| Model | Gemini model to use |

### Issues sheet columns

`Story | Chapter | Category | Timestamp | Issue | Model | Date`

### Scoring sheet columns

`Date | Story | Chapter | Score | Kid Safe Content | Script Match | Voice Consistency | Emotion Mismatch | Quality Degradation | Total Issues | Model`

---

## Single File Review

Upload a `.md` script file and a single audio file for a one-off analysis. The report is displayed in the browser and can be downloaded as a Markdown file.

---

## Scoring

Each chapter is scored out of 100. Issues are weighted by category:

| Category | Weight |
|---|---|
| Kid Safe Content | 1.0 |
| Script Match | 1.0 |
| Voice Consistency | 1.0 |
| Emotion Mismatch | 1.0 |
| Quality Degradation | 0.25 |

Pitch-related issues within Quality Degradation (jitter, pitch instability, pitch spike/drop) count at 0.5x. Each weighted issue deducts approximately 4 points from 100.

---

## Agents (Prompts)

The left panel contains the QC prompt sent to Gemini. Multiple prompts can be saved and switched between.

- **Default** — the built-in prompt, always kept in sync with the server code. Cannot be edited or deleted.
- **Save Changes** — saves edits to the currently selected prompt.
- **New Agent** — creates a new named prompt from a blank canvas.
- **Delete** — deletes the selected prompt (not available for Default).

---

## Notes

- Closing a browser tab stops the current file's processing but does not shut down the server.
- Audio files uploaded in Single File mode are deleted from Gemini's servers after analysis completes.
- The acoustic report (librosa) runs locally before Gemini is called. HNR and spectral flatness flags are treated as strong evidence; pitch/jitter flags are low-confidence and only reported if audibly perceptible.
