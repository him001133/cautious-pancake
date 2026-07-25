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

# ASS Style Definitions
STYLES = {
    "TikTok Viral Yellow": {
        "font": "Arial", "size": 80, "primary": "&H00FFFFFF&", "active_color": "&H0000FFFF&",
        "outline": 5, "shadow": 3, "outline_color": "&H00000000&"
    },
    "Hormozi Green": {
        "font": "Impact", "size": 85, "primary": "&H00FFFFFF&", "active_color": "&H00FF00&",
        "outline": 7, "shadow": 4, "outline_color": "&H00000000&"
    },
    "Cyber Cyan": {
        "font": "Arial", "size": 80, "primary": "&H00FFFFFF&", "active_color": "&H00FFFF00&",
        "outline": 5, "shadow": 3, "outline_color": "&H00000000&"
    },
    "Crimson Pop": {
        "font": "Arial", "size": 80, "primary": "&H00FFFFFF&", "active_color": "&H000000FF&",
        "outline": 6, "shadow": 3, "outline_color": "&H00000000&"
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

def generate_ass_subtitles(clip_words, output_ass_path, clip_start, clip_end, style_name="TikTok Viral Yellow"):
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

# --- STEP 1: TRANSCRIBE & POPULATE EDITOR TABLE ---
def step1_transcribe_and_detect(selected_file, num_clips, clip_duration, progress=gr.Progress()):
    if not selected_file or selected_file == "No videos found in inputs/ folder":
        return None, None, "❌ Please place a video inside the 'inputs/' folder."
    
    input_path = os.path.join(INPUT_DIR, selected_file)
    if not os.path.exists(input_path):
        return None, None, f"❌ Error: File '{input_path}' not found."

    progress(0.2, desc="🎙️ Transcribing audio and extracting micro-timestamps...")
    segments, _ = whisper_model.transcribe(input_path, word_timestamps=True)
    
    all_words = []
    for segment in segments:
        for w in segment.words:
            all_words.append({"word": w.word, "start": w.start, "end": w.end})
            
    if not all_words:
        return None, None, "❌ No speech detected in video!"

    total_duration = all_words[-1]["end"]
    clip_duration = float(clip_duration)
    num_clips = int(num_clips)
    
    progress(0.6, desc="🧠 Scoring speech density to find viral hooks...")
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
    
    # Populate the Editable Dataframe Table
    table_rows = []
    for i, win in enumerate(selected_windows):
        clip_label = f"Clip {i+1} ({int(win['start'])}s-{int(win['end'])}s)"
        for w in all_words:
            if w["start"] >= win["start"] and w["end"] <= win["end"]:
                table_rows.append([clip_label, w["word"].strip(), round(w["start"], 2), round(w["end"], 2)])
                
    progress(1.0, desc="✅ Hooks Extracted! Review and edit words in the table below.")
    return table_rows, selected_windows, f"✅ Step 1 Complete! Extracted {len(selected_windows)} clips. You can now edit any typo directly in the table below!"

# --- STEP 2: RENDER FROM EDITED TABLE ---
def step2_render_clips(selected_file, edited_table, selected_windows, caption_style, progress=gr.Progress()):
    if not edited_table or not selected_windows:
        return None, "❌ Please run Step 1 (Transcribe & Detect Hooks) before rendering!"
        
    input_path = os.path.join(INPUT_DIR, selected_file)
    generated_videos = []
    transcript_summary = []
    
    for i, win in enumerate(selected_windows):
        progress(0.2 + (0.7 * (i / len(selected_windows))), desc=f"🎨 Rendering Clip {i+1} of {len(selected_windows)} with '{caption_style}'...")
        
        clip_label = f"Clip {i+1} ({int(win['start'])}s-{int(win['end'])}s)"
        
        # Reconstruct word list strictly from the user's edited Dataframe rows
        clip_words = []
        for row in edited_table:
            if str(row[0]) == clip_label:
                clip_words.append({"word": str(row[1]), "start": float(row[2]), "end": float(row[3])})
                
        base_name = os.path.splitext(selected_file)[0]
        output_filename = f"viral_v3_{base_name}_clip{i+1}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Generate ASS subtitle file from edited words
        ass_path = os.path.join(OUTPUT_DIR, f"temp_v3_clip{i+1}.ass")
        generate_ass_subtitles(clip_words, ass_path, win["start"], win["end"], style_name=caption_style)
        
        filter_chain = f"crop=ih*(9/16):ih,scale=1080:1920,ass='{ass_path}'"
        command = [
            "ffmpeg", "-y", "-i", input_path, "-ss", str(win["start"]), "-to", str(win["end"]),
            "-vf", filter_chain, "-c:v", "libx264", "-c:a", "aac", output_path
        ]
        
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        caption_text = f"Clip {i+1}: {int(win['start'])}s - {int(win['end'])}s ({len(clip_words)} words)"
        generated_videos.append((output_path, caption_text))
        transcript_summary.append(f"🔥 {caption_text} [{caption_style}] - Burned from Custom Edits!")
        
    progress(1.0, desc="🚀 All Edited Shorts Generated Successfully!")
    return generated_videos, "\n".join(transcript_summary)

def refresh_file_list():
    files = get_input_videos()
    return gr.Dropdown(choices=files, value=files[0] if files else None)

# CSS & HTML Showroom
custom_css = """
.style-preview-container { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 10px; }
.style-card { background: #1e1e2e; border: 2px solid #313244; border-radius: 8px; padding: 12px; text-align: center; cursor: pointer; transition: all 0.3s; }
.style-card:hover { border-color: #89b4fa; transform: translateY(-2px); }
.style-title { color: #cdd6f4; font-size: 12px; font-weight: bold; margin-bottom: 4px; }
.preview-text { font-size: 18px; font-weight: 900; color: #fff; text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000; }
.yellow-highlight:hover .hl-y { color: #ffff00; }
.green-highlight:hover .hl-g { color: #00ff00; }
.cyan-highlight:hover .hl-c { color: #00ffff; }
.red-highlight:hover .hl-r { color: #ff0000; }
"""

hover_html = """
<div class="style-preview-container">
    <div class="style-card yellow-highlight"><div class="style-title">TikTok Yellow</div><div class="preview-text" style="font-family: Arial;">VIRAL <span class="hl-y">TEXT</span></div></div>
    <div class="style-card green-highlight"><div class="style-title">Hormozi Green</div><div class="preview-text" style="font-family: Impact;">MORE <span class="hl-g">VIEWS</span></div></div>
    <div class="style-card cyan-highlight"><div class="style-title">Cyber Cyan</div><div class="preview-text" style="font-family: Arial;">FUTURE <span class="hl-c">AI</span></div></div>
    <div class="style-card red-highlight"><div class="style-title">Crimson Pop</div><div class="preview-text" style="font-family: Arial;">HIGH <span class="hl-r">HOOK</span></div></div>
</div>
"""

with gr.Blocks(title="AI Short Video Clipper v3.0", css=custom_css) as demo:
    # State variable to silently pass detected clip timestamps between Step 1 and Step 2
    clip_windows_state = gr.State()
    
    gr.Markdown("# 🎬 AI Video Clipper v3.0: 2-Step Pro Editor")
    gr.Markdown("Step 1: Extract viral hooks and **manually edit typos in the data table**. Step 2: Burn karaoke captions and render!")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Step 1: Target & Transcribe")
            with gr.Row():
                workspace_dropdown = gr.Dropdown(choices=get_input_videos(), label="Select Source Video", value=get_input_videos()[0] if get_input_videos() else None, scale=4)
                refresh_btn = gr.Button("🔄", scale=1)
            
            num_clips_input = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Number of Clips")
            clip_duration_input = gr.Slider(minimum=15, maximum=60, value=25, step=5, label="Duration (seconds)")
            
            step1_btn = gr.Button("🎙️ Step 1: Detect Hooks & Transcribe", variant="primary")
            status_box = gr.Textbox(label="Status", value="⏳ Waiting for Step 1...", interactive=False, lines=2)
            
            gr.Markdown("---")
            gr.Markdown("### 🔥 Step 2: Style & Render")
            gr.HTML(hover_html)
            style_selector = gr.Radio(choices=["TikTok Viral Yellow", "Hormozi Green", "Cyber Cyan", "Crimson Pop"], value="TikTok Viral Yellow", label="Caption Preset")
            
            step2_btn = gr.Button("🔥 Step 2: Burn Captions & Render Shorts", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Interactive Transcript Editor (Click any word cell to edit!)")
            # 🔥 CHANGED: height=300 -> max_height=300 to conform to Gradio 5.x standards
            transcript_editor = gr.Dataframe(
                headers=["Clip Label", "Word (Click to Edit)", "Start (s)", "End (s)"],
                datatype=["str", "str", "number", "number"],
                type="array",
                interactive=True,
                row_count=(10, "dynamic"),
                col_count=(4, "fixed"),
                max_height=300
            )
            
            gr.Markdown("### 📺 Final Render Gallery")
            video_gallery = gr.Gallery(label="Generated 9:16 Shorts", show_label=True, columns=2, rows=1, object_fit="contain", height=450)

    # Event Listeners
    refresh_btn.click(fn=refresh_file_list, outputs=[workspace_dropdown])
    
    step1_btn.click(
        fn=step1_transcribe_and_detect,
        inputs=[workspace_dropdown, num_clips_input, clip_duration_input],
        outputs=[transcript_editor, clip_windows_state, status_box]
    )
    
    step2_btn.click(
        fn=step2_render_clips,
        inputs=[workspace_dropdown, transcript_editor, clip_windows_state, style_selector],
        outputs=[video_gallery, status_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)