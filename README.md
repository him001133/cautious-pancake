# 🎬 AutoDirector AI Studio

AutoDirector AI is an advanced, web-based video editing suite and AI clipping tool built with Python and Streamlit. It leverages **Google's Gemini 1.5 Flash Multimodal AI** and **FFmpeg** to automate the tedious parts of video production, while offering a full custom-built Non-Linear Editor (NLE) right in the browser.

Designed with a heavy focus on the modern short-form content pipeline, it features specialized tools for generating highly accurate **Hinglish** (Hindi + English phonetic) captions, making it a tailored solution for the South Asian creator economy.

---

## ✨ Core Modules

### 1. ✂️ AI Clipper
Transform long podcasts and streams into viral Shorts, Reels, and TikToks.
* **Multimodal AI Analysis:** Uploads audio directly to Gemini 1.5 Flash to intelligently find the most engaging hooks and emotional peaks.
* **Target Durations:** Set strict minimum and maximum clip lengths (e.g., 15 to 60 seconds).
* **Auto-Framing Layouts:** Automatically crop landscape video into Vertical (9:16), Split-screen, Trio, Spotlight, or Centered layouts.
* **Cinematic Filters & Audio:** Built-in color grading presets (Vibrant, Cinematic, B&W) and automatic audio normalization (`loudnorm`).

### 2. 💬 Pro Caption Studio
A dedicated workspace for burning highly customized, viral-style subtitles.
* **Native Hinglish Engine:** Bypasses traditional English-only transcription models. Gemini natively listens and transcribes flawless Latin-script Hindi-English phonetics.
* **SRT Integration:** Import existing `.srt` files and instantly parse them into the interactive timeline.
* **Advanced Typography:** 1000% customizable. Adjust fonts, text colors, opaque background boxes, dynamic shadows, and letter spacing.
* **Viral Animations:** Includes trending presets like *Pop In*, *Bounce Drop*, *Dynamic Pop*, and *Rotate In*.
* **Smart Word Chunker:** Automatically splits long sentences into fast-paced, 3-4 word aesthetic subtitle blocks.

### 3. 🎬 Full Studio Editor (Web NLE)
A professional, multi-track video editor built to overcome Streamlit's iframe limitations.
* **Custom JS Data Bridge:** Seamlessly integrates a native Streamlit Python Project Bin with a secure, custom JavaScript timeline.
* **Zero-Lag Scrubbing:** High-performance DOM-hacking script synchronizes the HTML timeline playhead directly with the Streamlit `<video>` tag for real-time, 60fps playback.
* **Drag-and-Drop Tracks:** Add assets to `V1` or `A1` tracks and physically arrange them along the sequence.

---

## 🛠️ Tech Stack

* **Backend / UI Framework:** Python 3.10+, Streamlit
* **AI Engine:** Google Gemini API (`gemini-1.5-flash-latest`)
* **Video Processing:** FFmpeg (Subprocess execution)
* **Frontend Timeline:** Vanilla HTML, CSS, JavaScript (Injected via Streamlit Components)

---

## 🚀 Installation & Setup

### Prerequisites
1. **Python 3.x** installed on your system.
2. **FFmpeg** installed and added to your system's PATH variables.
3. A **Google Gemini API Key** (Get one free from Google AI Studio).

### Local Deployment

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/autodirector-ai.git](https://github.com/yourusername/autodirector-ai.git)
cd autodirector-ai
