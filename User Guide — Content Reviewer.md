# Content Reviewer — User Guide

---

## What This Tool Does

The Content Reviewer automatically checks AI-generated audio against a script and produces a structured quality report. It evaluates five parameters for every chapter:

1. **Kid Safe Content** — flags anything age-inappropriate for young listeners
2. **Script Match** — detects missing lines, word substitutions, and missing SFX cues
3. **Voice Consistency** — checks that each character sounds the same across all their lines
4. **Emotion / Prosody** — verifies the vocal delivery matches the emotional stage directions
5. **AI Voice Quality** — flags pitch spikes, robotic wobble, scratchiness, and word clipping

---

## Logging In

Open the tool URL in your browser. You will see a password prompt. Enter the password and click **Sign in**.

---

## Two Modes

The tool has two modes, selectable via the tabs on the right side of the screen.

### Batch Review *(recommended for production)*

Processes an entire story — all chapters in a Google Drive folder — in one run. Results are written directly to Google Sheets.

### Single File

Upload one script and one audio file. The report appears on screen and can be downloaded as a `.md` file. Use this for spot-checking a single chapter.

---

## Running a Batch Review

**Before you start**, make sure:
- All audio files for the story are in one Google Drive folder, named consistently (e.g. `BT-TM03 CHAPTER 1.mp3`, `BT-TM03 CHAPTER 2.mp3`).
- The full story script is in a Google Doc.
- The Service Account status indicator shows **Configured** (green dot). If it shows red, contact the administrator.

**Steps:**

1. Select the **Batch Review** tab.
2. Fill in the four URL fields:
   - **Drive Folder URL** — the Google Drive folder containing all the chapter audio files.
   - **Google Doc URL** — the Google Doc containing the full story script.
   - **Issues Sheet URL** — the Google Sheet where individual issues will be logged (pre-filled).
   - **Scoring Sheet URL** — the Google Sheet where per-chapter scores will be logged (pre-filled).
3. **Story Name** — optional. If left blank, the name is taken from the Google Doc title. You can override it here (e.g. `BT-TM03-EPIC`).
4. **Language** — select English or Hindi.
5. **Model** — leave as `gemini-2.5-flash` unless instructed otherwise.
6. Click **Run Batch**.

**While running:**
The progress log shows each step — downloading audio, acoustic analysis, uploading to Gemini, calling the model. A progress bar tracks overall completion. Each chapter appears in the list as it finishes, with its issue count and score.

**When complete:**
A summary card shows the total chapters processed and total issues logged. Click **Open Google Sheet** to view the full results.

---

## Running a Single File Review

1. Select the **Single File** tab.
2. Click the **Script** upload zone and select the script file (`.md` format).
3. Click the **Audio** upload zone and select the audio file. Supported formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.opus`, `.webm`, `.mp4`.
4. Select the **Language** (English or Hindi).
5. Select the **Model**.
6. Click **Run Analysis**.

The report appears on screen when complete. Click **Download .md** to save it.

---

## Script File Format

The script must be a `.md` (Markdown) file. The tool reads it as plain text, so any readable format works. Make sure:
- Character names are clearly labelled before each line (e.g. `T-MAX:`, `NARRATOR:`).
- Emotional stage directions are included (e.g. `(nervous)`, `(commanding)`).
- SFX and music cues are present (e.g. `SFX: jungle breeze`).
- Section headers and editor notes are fine to leave in — the tool knows to ignore them.

---

## Understanding the Google Sheet Output

Each issue is logged as one row with these columns:

| Column | Description |
|---|---|
| Story | Story identifier |
| Chapter | Chapter name |
| Category | One of the 5 QC parameters |
| Timestamp | Where in the audio the issue occurs `[MM:SS]` |
| Issue | Full description of the issue |
| Model | Gemini model used |
| Date | Date the review was run |

The **Scoring Sheet** has one row per chapter with a score (0–100) and a breakdown by category. A score of 100 means no issues were detected.

---

## The Prompt Panel

The large text area on the left is the QC prompt — the instructions sent to the AI. You do not normally need to change this.

**Saved prompts (Agents):**
- The dropdown at the top lets you switch between saved prompt versions.
- **Default** is the standard prompt and cannot be edited or deleted.
- To create a custom variant: click **New Agent**, give it a name, edit the prompt text, and click **Confirm**.
- To update an existing custom agent: make your changes and click **Save Changes**.
- To delete a custom agent: select it and click **Delete**.

The selected prompt applies to both Batch and Single File runs.

---

## Tips

- **Do not close the browser tab** while a batch is running. The job will be cancelled if you navigate away.
- **Audio file names** in the Drive folder should include the chapter number so the tool can identify them correctly (e.g. `STORY NAME CHAPTER 1.mp3`).
- **Hindi scripts** can be in Devanagari, Hinglish (romanised), or a mix — the tool handles both.
- If a run fails with a connection error, check the progress log for details, then try again.
- For a long story (10+ chapters), a batch run typically takes 15–30 minutes.

---

## Contact

For access issues, configuration problems, or unexpected results, contact the administrator.
