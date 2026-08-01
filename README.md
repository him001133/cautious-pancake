# cautious-pancake# 🎬 AI Short Video Clipper v5.2

An automated Python and Gradio-based video editing pipeline that transforms long-form videos and podcasts into highly engaging 9:16 vertical shorts. 

## 🚀 Key Features

* **LLM Story Analysis Engine:** Integrates the Google GenAI SDK (Gemini 1.5 Flash) to read video transcripts and identify the most engaging narrative hooks and information gaps.
* **Speech Density Fallback:** Automatically calculates spoken word frequency to find high-energy moments if the AI API is bypassed.
* **Auto-Director Framing Engine:** Utilizes OpenCV to count faces in the frame and automatically makes directorial camera decisions.
* **Dynamic Smart Cropping:** Applies a moving X-coordinate center crop for solo speakers to keep them in frame.
* **Multi-Speaker Widescreen:** Automatically applies a blurred widescreen stack for multi-speaker podcast setups so nobody gets cut out.
* **Local Audio Transcription:** Leverages Faster-Whisper for high-speed, local word-level timestamp generation.
* **Interactive Web Dashboard:** A sleek Gradio UI for parameter tuning, video selection, and a built-in playback gallery.
* **Output Auto-Cleaner:** Automatically purges old test renders from the directory to save cloud storage space.

## 🛠️ Technology Stack

* **Python:** Core scripting and logic.
* **Gradio:** Frontend web UI framework.
* **Faster-Whisper:** Local audio transcription model.
* **OpenCV (opencv-python-headless):** Computer vision and face tracking.
* **Google GenAI SDK (google-genai):** Contextual transcript analysis.
* **FFmpeg:** High-speed video rendering and filtering.

## ⚙️ How to Run

1. Place your `.mp4`, `.mov`, or `.mkv` source videos into the `inputs/` folder.
2. (Optional) Configure your Google AI Studio key by running `echo 'export GEMINI_API_KEY="YOUR_KEY"' >> ~/.bashrc` in your terminal.
3. Launch the web dashboard by executing `python app.py` in the terminal.
4. Use the UI to execute Step 1 (Scan Hooks & Extract Timestamps).
5. Use the UI to execute Step 2 (Auto-Frame & Render Shorts) and view your videos in the gallery.