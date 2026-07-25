import os
import subprocess
import gradio as gr
from faster_whisper import WhisperModel

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Whisper Model...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

# ASS Style Definitions (ASS Colors are &HAABBGGRR)
STYLES = {
    "TikTok Viral Yellow": {
        "font": "Arial",
        "size": 80,
        "primary": "&H00FFFFFF&",     # White
        "active_color": "&H0000FFFF&", # Yellow
        "outline": 5,
        "shadow": 3,
        "outline_color": "&H00000000&" # Black
    },
    "Hormozi Green": {
        "font": "Impact",
        "size": 85,
        "primary": "&H00FFFFFF&",
        "active_color": "&H0000FF00&", # Neon Green
        "outline": 7,
        "shadow": 4,
        "outline_color": "&H00000000&"
    },
    "Cyber Cyan": {
        "font": "Arial",
        "size": 80,
        "primary": "&H00FFFFFF&",
        "active_color": "&H00FFFF00&", # Cyan
        "outline": 5,
        "shadow": 3,
        "outline_color": "&H00000000&"
    },
    "Crimson Pop": {
        "font": "Arial",
        "size": 80,
        "primary": "&H00FFFFFF&",
        "active_color": "&H000000FF&", # Crimson Red
        "outline": 6,
        "shadow": 3,
        "outline_color": "&H00000000&"
    }
}

def seconds_to_ass_time(sec):
    sec = max(0, sec)
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    centiseconds = int(round((sec - int(sec)) * 100))
    if centiseconds >= 100:
        centiseconds = 0
        seconds += 1
    return f"{hours:01d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

def generate_ass_subtitles(words, output_ass_path, clip_start, clip_end, style_name="TikTok Viral Yellow"):
    clip_words = [w for w in words if w["end"] >= clip_start and w["start"] <= clip_end]
    if not clip_words:
        return

    config = STYLES.get(style_name, STYLES["TikTok Viral Yellow"])

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CustomStyle,{config['font']},{config['size']},{config['primary']},&H000000FF,{config['outline_color']},&H80000000,1,0,0,0,100,100,0,0,1,{config['outline']},{config['shadow']},2,50,50,350,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    
    chunks = []
    current_chunk = []
    for i, w in enumerate(clip_words):
        current_chunk.append(w)
        gap_break = False
        if i + 1 < len(clip_words):
            if clip_words[i+1]["start"] - w["end"] > 0.4:
                gap_break = True
                
        if len(current_chunk) >= 4 or gap_break or i == len(clip_words) - 1:
            chunks.append(current_chunk)
            current_chunk = []

    for chunk in chunks:
        for idx, active_word in enumerate(chunk):
            start_t = max(0, active_word["start"] - clip_start)
            
            if idx < len(chunk) - 1:
                end_t = max(0, chunk[idx+1]["start"] - clip_start)
            else:
                end_t = max(0, active_word["end"] - clip_start + 0.25)
                
            text_parts = []
            for w in chunk:
                clean_word = w["word"].strip()
                if w == active_word:
                    # Highlight active word with configured color
                    text_parts.append(f"{{\\c{config['active_color']}}}{clean_word}{{\\c{config['primary']}}}")
                else:
                    text_parts.append(clean_word)
                    
            line_text = " ".join(text_parts)
            start_str = seconds_to_ass_time(start_t)
            end_str = seconds_to_ass_time(end_t)
            events.append(f"Dialogue: 0,{start_str},{end_str},CustomStyle,,0,0,0,,{line_text}")

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(events))

def get_input_videos():
    extensions = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    if os.path.exists(INPUT_DIR):
        files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(extensions)]
        return sorted(files) if files else ["No videos found in inputs/ folder"]
    return ["No videos found in inputs/ folder"]

def generate_multiple_clips(selected_file, num_clips, clip_duration, caption_style, progress=gr.Progress()):
    if not selected_file or selected_file == "No videos found in inputs/ folder":
        return None, "❌ Please place a video inside the 'inputs/' folder and select it."
    
    input_path = os.path.join(INPUT_DIR, selected_file)
    if not os.path.exists(input_path):
        return None, f"❌ Error: File '{input_path}' not found."

    progress(0.1, desc="🎙️ Transcribing audio and extracting word timestamps...")
    segments, _ = whisper_model.transcribe(input_path, word_timestamps=True)
    
    all_words = []
    for segment in segments:
        for w in segment.words:
            all_words.append({"word": w.word, "start": w.start, "end": w.end})
            
    if not all_words:
        return None, "❌ No speech detected in video!"

    total_duration = all_words[-1]["end"]
    clip_duration = float(clip_duration)
    num_clips = int(num_clips)
    
    progress(0.3, desc="🧠 Analyzing speech density for high-energy hooks...")
    step = 10.0
    candidates = []
    for start_t in range(0, int(max(0, total_duration - clip_duration)), int(step)):
        end_t = start_t + clip_duration
        word_count = sum(1 for w in all_words if w["start"] >= start_t and w["end"] <= end_t)
        candidates.append({"start": start_t, "end": end_t, "score": word_count})
        
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected_windows = []
    for cand in candidates:
        overlap = any(max(cand["start"], sel["start"]) < min(cand["end"], sel["end"]) for sel in selected_windows)
        if not overlap:
            selected_windows.append(cand)
            if len(selected_windows) == num_clips:
                break
                
    selected_windows.sort(key=lambda x: x["start"])
    
    generated_videos = []
    transcript_summary = []
    
    for i, win in enumerate(selected_windows):
        progress(0.4 + (0.5 * (i / num_clips)), desc=f"🎨 Burning '{caption_style}' captions on Clip {i+1} of {len(selected_windows)}...")
        
        base_name = os.path.splitext(selected_file)[0]
        output_filename = f"viral_{base_name}_clip{i+1}_{int(win['start'])}s_to_{int(win['end'])}s.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 1. Generate subtitle script with selected style
        ass_filename = f"temp_captions_clip{i+1}.ass"
        ass_path = os.path.join(OUTPUT_DIR, ass_filename)
        generate_ass_subtitles(all_words, ass_path, win["start"], win["end"], style_name=caption_style)
        
        # 2. Render 9:16 vertical short with burned subtitles
        filter_chain = f"crop=ih*(9/16):ih,scale=1080:1920,ass='{ass_path}'"
        
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(win["start"]),
            "-to", str(win["end"]),
            "-vf", filter_chain,
            "-c:v", "libx264",
            "-c:a", "aac",
            output_path
        ]
        
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        caption_label = f"Clip {i+1}: {int(win['start'])}s - {int(win['end'])}s ({win['score']} words)"
        generated_videos.append((output_path, caption_label))
        transcript_summary.append(f"✨ {caption_label} [{caption_style}]")
        
    progress(1.0, desc="🚀 All Viral Captioned Clips Generated!")
    return generated_videos, "\n".join(transcript_summary)

def refresh_file_list():
    files = get_input_videos()
    return gr.Dropdown(choices=files, value=files[0] if files else None)

# CSS for Live Hover Preview Cards
custom_css = """
.style-preview-container {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 15px;
}
.style-card {
    background: #1e1e2e;
    border: 2px solid #313244;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}
.style-card:hover {
    border-color: #89b4fa;
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.4);
}
.style-title {
    color: #cdd6f4;
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 8px;
}
.preview-text {
    font-size: 20px;
    font-weight: 900;
    color: #ffffff;
    text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000;
}
.yellow-highlight:hover .hl-y { color: #ffff00; }
.green-highlight:hover .hl-g { color: #00ff00; }
.cyan-highlight:hover .hl-c { color: #00ffff; }
.red-highlight:hover .hl-r { color: #ff0000; }
"""

# HTML Hover Preview Grid Component
hover_html = """
<div class="style-preview-container">
    <div class="style-card yellow-highlight">
        <div class="style-title">TikTok Viral Yellow</div>
        <div class="preview-text" style="font-family: Arial;">THIS IS <span class="hl-y">VIRAL</span> TEXT</div>
    </div>
    <div class="style-card green-highlight">
        <div class="style-title">Hormozi Green</div>
        <div class="preview-text" style="font-family: Impact;">GET MORE <span class="hl-g">VIEWS</span> NOW</div>
    </div>
    <div class="style-card cyan-highlight">
        <div class="style-title">Cyber Cyan</div>
        <div class="preview-text" style="font-family: Arial;">FUTURE <span class="hl-c">AI</span> CONTENT</div>
    </div>
    <div class="style-card red-highlight">
        <div class="style-title">Crimson Pop</div>
        <div class="preview-text" style="font-family: Arial;">HIGH <span class="hl-r">HOOK</span> SCORE</div>
    </div>
</div>
<p style="text-align: center; color: #a6adc8; font-size: 12px; margin-top: -5px;">💡 Hover over any card above to preview word-highlighting animations!</p>
"""

with gr.Blocks(title="AI Short Video Clipper v2.1", css=custom_css) as demo:
    gr.Markdown("# 🎬 AI Multi-Clip Dashboard v2.1")
    gr.Markdown("Pick a caption style preset with live hover previews and render 9:16 Shorts with customized typography.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Choose Source Video")
            with gr.Row():
                workspace_dropdown = gr.Dropdown(
                    choices=get_input_videos(),
                    label="Select from inputs/ Folder",
                    value=get_input_videos()[0] if get_input_videos() else None,
                    scale=4
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)
            
            gr.Markdown("### 2️⃣ Caption Style Selector")
            gr.HTML(hover_html)
            
            style_selector = gr.Radio(
                choices=["TikTok Viral Yellow", "Hormozi Green", "Cyber Cyan", "Crimson Pop"],
                value="TikTok Viral Yellow",
                label="Select Render Style"
            )
            
            gr.Markdown("### 3️⃣ Clip Settings")
            num_clips_input = gr.Slider(minimum=1, maximum=6, value=3, step=1, label="Number of Viral Clips")
            clip_duration_input = gr.Slider(minimum=15, maximum=60, value=30, step=5, label="Clip Duration (seconds)")
                
            submit_btn = gr.Button("🔥 Generate Captioned Shorts", variant="primary")
            
            transcript_output = gr.Textbox(label="Processing Summary", lines=5, interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 📺 Captioned Video Gallery")
            video_gallery = gr.Gallery(
                label="Generated Shorts",
                show_label=True,
                columns=2,
                rows=2,
                object_fit="contain",
                height=600
            )

    refresh_btn.click(fn=refresh_file_list, outputs=[workspace_dropdown])
    
    submit_btn.click(
        fn=generate_multiple_clips,
        inputs=[workspace_dropdown, num_clips_input, clip_duration_input, style_selector],
        outputs=[video_gallery, transcript_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)