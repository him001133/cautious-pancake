import os
import subprocess
import numpy as np
import gradio as gr
from faster_whisper import WhisperModel

# Import OpenCV for computer vision face tracking
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Whisper Model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- OUTPUT FOLDER CLEANER ---
def clear_output_folder():
    """Removes all files inside the outputs directory."""
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                pass

# --- AUTO-DIRECTOR: SMART FRAMING CALCULATOR ---
def get_auto_framing_filter(video_path, start_t, end_t):
    """
    Scans the clip to count faces.
    - 1 Face: Face-tracked 9:16 crop.
    - 2+ Faces: Blurred Widescreen Fit (prevents chopping active speakers).
    - 0 Faces: Center Crop.
    """
    if not CV2_AVAILABLE or not os.path.exists(video_path):
        return "crop=ih*(9/16):ih,scale=1080:1920", "Standard Center Crop (Fallback)"

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width == 0 or height == 0:
        cap.release()
        return "crop=ih*(9/16):ih,scale=1080:1920", "Standard Center Crop (Fallback)"

    crop_w = int(height * (9 / 16))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    sample_count = 10
    duration = max(1.0, end_t - start_t)
    step = duration / (sample_count + 1)
    
    face_counts = []
    detected_x_centers = []
    
    for i in range(1, sample_count + 1):
        sample_time = start_t + (i * step)
        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Tuned to ignore small background noise
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))
        
        face_counts.append(len(faces))
        
        if len(faces) == 1:
            fx, fy, fw, fh = faces[0]
            detected_x_centers.append(fx + (fw // 2))
            
    cap.release()
    
    # Calculate average number of faces detected across the clip
    median_faces = int(np.median(face_counts)) if face_counts else 0

    # AUTO-DECISION LOGIC
    if median_faces >= 2:
        print("🎬 Auto-Director: Detected Multiple Faces! Using Blurred Widescreen.")
        filter_chain = "split[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:10[bg_blur];[fg]scale=1080:-1[fg_scale];[bg_blur][fg_scale]overlay=0:(H-h)/2"
        return filter_chain, "Blurred Widescreen (Multi-Speaker)"
        
    elif median_faces == 1 and detected_x_centers:
        print("🎯 Auto-Director: Detected Single Face! Tracking X-Coordinate.")
        median_x = float(np.median(detected_x_centers))
        filtered_centers = [x for x in detected_x_centers if abs(x - median_x) < (width * 0.25)]
        final_x = int(np.mean(filtered_centers)) if filtered_centers else int(median_x)
        
        crop_x = max(0, min(int(final_x - (crop_w // 2)), width - crop_w))
        filter_chain = f"crop={crop_w}:{height}:{crop_x}:0,scale=1080:1920"
        return filter_chain, "Face-Tracked Solo Crop"
        
    else:
        print("↔️ Auto-Director: No clear faces found. Defaulting to Center Crop.")
        default_x = (width - crop_w) // 2
        filter_chain = f"crop={crop_w}:{height}:{default_x}:0,scale=1080:1920"
        return filter_chain, "Center Crop"

def get_input_videos():
    extensions = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    if os.path.exists(INPUT_DIR):
        files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(extensions)]
        return sorted(files) if files else ["No videos found in inputs/ folder"]
    return ["No videos found in inputs/ folder"]

# --- STEP 1: HOOK DETECTION ---
def step1_detect(selected_file, language_mode, num_clips, clip_duration, progress=gr.Progress()):
    if not selected_file or selected_file == "No videos found in inputs/ folder":
        return None, None, "❌ Please place a video inside the 'inputs/' folder."
    
    input_path = os.path.join(INPUT_DIR, selected_file)
    if not os.path.exists(input_path):
        return None, None, f"❌ Error: File '{input_path}' not found."

    progress(0.2, desc=f"🎙️ Transcribing audio ({language_mode})...")
    
    lang_code = "hi" if "Hindi" in language_mode else "en" if "English" in language_mode else None
    segments, _ = whisper_model.transcribe(input_path, language=lang_code, word_timestamps=True)
    
    all_words = []
    for segment in segments:
        for w in segment.words:
            all_words.append({"word": w.word, "start": w.start, "end": w.end})
            
    if not all_words:
        return None, None, "❌ No speech detected in video!"

    total_duration = all_words[-1]["end"]
    clip_duration = float(clip_duration)
    num_clips = int(num_clips)
    
    progress(0.6, desc=f"🧠 Scanning for top {num_clips} viral hooks...")
    step = 5.0 
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
    
    table_rows = []
    for i, win in enumerate(selected_windows):
        clip_label = f"Clip {i+1} ({int(win['start'])}s-{int(win['end'])}s)"
        clip_words = [w["word"].strip() for w in all_words if w["start"] >= win["start"] and w["end"] <= win["end"]]
        snippet = " ".join(clip_words[:15]) + "..." if clip_words else "No spoken text detected"
        table_rows.append([clip_label, round(win["start"], 1), round(win["end"], 1), snippet])
                
    progress(1.0, desc="✅ Hooks Extracted!")
    return table_rows, selected_windows, f"✅ Step 1 Complete! Extracted {len(selected_windows)} hooks."

# --- STEP 2: RENDERING ENGINE ---
def step2_render_clips(selected_file, selected_windows, auto_clean, progress=gr.Progress()):
    if not selected_windows:
        return None, "❌ Please run Step 1 (Detect Hooks) before rendering!"
    
    if auto_clean:
        clear_output_folder()
        
    input_path = os.path.join(INPUT_DIR, selected_file)
    generated_videos = []
    render_summary = []
    
    for i, win in enumerate(selected_windows):
        progress(0.1 + (0.8 * (i / len(selected_windows))), desc=f"🎬 Analyzing frames & rendering Clip {i+1} of {len(selected_windows)}...")
        
        start_t = win["start"]
        end_t = win["end"]
        clip_duration_real = end_t - start_t
        
        # Let the Auto-Director decide the framing!
        filter_chain, decision_label = get_auto_framing_filter(input_path, start_t, end_t)
        
        base_name = os.path.splitext(selected_file)[0]
        output_filename = f"viral_v4.3_{base_name}_clip{i+1}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        command = [
            "ffmpeg", "-y", "-ss", str(start_t), "-i", input_path, "-t", str(clip_duration_real),
            "-vf", filter_chain, "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", output_path
        ]
        
        subprocess.run(command)
        
        caption_text = f"Clip {i+1}: {int(start_t)}s - {int(end_t)}s ({decision_label})"
        generated_videos.append((output_path, caption_text))
        render_summary.append(f"⚡ {caption_text}")
        
    progress(1.0, desc="🚀 All Shorts Rendered Successfully!")
    return generated_videos, "\n".join(render_summary)

def refresh_file_list():
    files = get_input_videos()
    return gr.Dropdown(choices=files, value=files[0] if files else None)

custom_css = """
.stat-box { background: #1e1e2e; border: 2px solid #313244; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 10px; }
.stat-title { color: #cdd6f4; font-size: 14px; font-weight: bold; }
.stat-desc { color: #a6adc8; font-size: 12px; }
"""

with gr.Blocks(title="AI Short Video Clipper v4.3", css=custom_css) as demo:
    clip_windows_state = gr.State()
    
    gr.Markdown("# 🎬 AI Video Clipper v4.3: Fully Automated Auto-Director Engine")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Step 1: Target & Detect")
            with gr.Row():
                workspace_dropdown = gr.Dropdown(choices=get_input_videos(), label="Select Source Video", value=get_input_videos()[0] if get_input_videos() else None, scale=4)
                refresh_btn = gr.Button("🔄", scale=1)
            
            language_selector = gr.Dropdown(
                choices=["English", "Hindi", "Auto-Detect"],
                value="English",
                label="Audio Language Mode"
            )
            
            num_clips_input = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Viral Clips to Extract")
            clip_duration_input = gr.Slider(minimum=15, maximum=60, value=25, step=5, label="Clip Duration (seconds)")
            
            step1_btn = gr.Button("🎙️ Step 1: Scan Hooks & Extract Timestamps", variant="primary")
            status_box = gr.Textbox(label="System Status", value="⏳ Ready. Select a video and click Step 1.", interactive=False, lines=2)
            
            with gr.Row():
                stop_btn = gr.Button("🛑 Stop Active Process", variant="stop", scale=2)
            
            gr.Markdown("---")
            gr.Markdown("### 🎬 Step 2: Auto-Direct & Render")
            gr.HTML('<div class="stat-box"><div class="stat-title">🤖 AI Auto-Director Active</div><div class="stat-desc">The engine will automatically scan the clip. If it sees a solo speaker, it face-tracks them. If it sees a podcast (2+ people), it switches to blurred widescreen so nobody gets cut!</div></div>')
            
            auto_clean_checkbox = gr.Checkbox(value=True, label="Auto-clear outputs folder before new render")
            step2_btn = gr.Button("⚡ Step 2: Auto-Frame & Render Shorts", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Extracted Viral Hooks Overview")
            hooks_table = gr.Dataframe(
                headers=["Clip Label", "Start (s)", "End (s)", "Spoken Text Preview"],
                datatype=["str", "number", "number", "str"],
                type="array",
                interactive=False,
                row_count=(5, "dynamic"),
                col_count=(4, "fixed")
            )
            
            gr.Markdown("### 📺 Rendered Gallery")
            video_gallery = gr.Gallery(label="Generated 9:16 Shorts", show_label=True, columns=3, rows=2, object_fit="contain", height=500)

    # Event Listeners
    refresh_btn.click(fn=refresh_file_list, outputs=[workspace_dropdown])
    
    step1_event = step1_btn.click(
        fn=step1_detect,
        inputs=[workspace_dropdown, language_selector, num_clips_input, clip_duration_input],
        outputs=[hooks_table, clip_windows_state, status_box]
    )
    
    step2_event = step2_btn.click(
        fn=step2_render_clips,
        inputs=[workspace_dropdown, clip_windows_state, auto_clean_checkbox],
        outputs=[video_gallery, status_box]
    )
    
    stop_btn.click(
        fn=lambda: "🛑 Process interrupted and stopped by user!",
        inputs=None,
        outputs=[status_box],
        cancels=[step1_event, step2_event]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)import os
import subprocess
import numpy as np
import gradio as gr
from faster_whisper import WhisperModel

# Import OpenCV for computer vision face tracking
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Whisper Model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- OUTPUT FOLDER CLEANER ---
def clear_output_folder():
    """Removes all files inside the outputs directory."""
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                pass

# --- AUTO-DIRECTOR: SMART FRAMING CALCULATOR ---
def get_auto_framing_filter(video_path, start_t, end_t):
    """
    Scans the clip to count faces.
    - 1 Face: Face-tracked 9:16 crop.
    - 2+ Faces: Blurred Widescreen Fit (prevents chopping active speakers).
    - 0 Faces: Center Crop.
    """
    if not CV2_AVAILABLE or not os.path.exists(video_path):
        return "crop=ih*(9/16):ih,scale=1080:1920", "Standard Center Crop (Fallback)"

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width == 0 or height == 0:
        cap.release()
        return "crop=ih*(9/16):ih,scale=1080:1920", "Standard Center Crop (Fallback)"

    crop_w = int(height * (9 / 16))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    sample_count = 10
    duration = max(1.0, end_t - start_t)
    step = duration / (sample_count + 1)
    
    face_counts = []
    detected_x_centers = []
    
    for i in range(1, sample_count + 1):
        sample_time = start_t + (i * step)
        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Tuned to ignore small background noise
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))
        
        face_counts.append(len(faces))
        
        if len(faces) == 1:
            fx, fy, fw, fh = faces[0]
            detected_x_centers.append(fx + (fw // 2))
            
    cap.release()
    
    # Calculate average number of faces detected across the clip
    median_faces = int(np.median(face_counts)) if face_counts else 0

    # AUTO-DECISION LOGIC
    if median_faces >= 2:
        print("🎬 Auto-Director: Detected Multiple Faces! Using Blurred Widescreen.")
        filter_chain = "split[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:10[bg_blur];[fg]scale=1080:-1[fg_scale];[bg_blur][fg_scale]overlay=0:(H-h)/2"
        return filter_chain, "Blurred Widescreen (Multi-Speaker)"
        
    elif median_faces == 1 and detected_x_centers:
        print("🎯 Auto-Director: Detected Single Face! Tracking X-Coordinate.")
        median_x = float(np.median(detected_x_centers))
        filtered_centers = [x for x in detected_x_centers if abs(x - median_x) < (width * 0.25)]
        final_x = int(np.mean(filtered_centers)) if filtered_centers else int(median_x)
        
        crop_x = max(0, min(int(final_x - (crop_w // 2)), width - crop_w))
        filter_chain = f"crop={crop_w}:{height}:{crop_x}:0,scale=1080:1920"
        return filter_chain, "Face-Tracked Solo Crop"
        
    else:
        print("↔️ Auto-Director: No clear faces found. Defaulting to Center Crop.")
        default_x = (width - crop_w) // 2
        filter_chain = f"crop={crop_w}:{height}:{default_x}:0,scale=1080:1920"
        return filter_chain, "Center Crop"

def get_input_videos():
    extensions = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    if os.path.exists(INPUT_DIR):
        files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(extensions)]
        return sorted(files) if files else ["No videos found in inputs/ folder"]
    return ["No videos found in inputs/ folder"]

# --- STEP 1: HOOK DETECTION ---
def step1_detect(selected_file, language_mode, num_clips, clip_duration, progress=gr.Progress()):
    if not selected_file or selected_file == "No videos found in inputs/ folder":
        return None, None, "❌ Please place a video inside the 'inputs/' folder."
    
    input_path = os.path.join(INPUT_DIR, selected_file)
    if not os.path.exists(input_path):
        return None, None, f"❌ Error: File '{input_path}' not found."

    progress(0.2, desc=f"🎙️ Transcribing audio ({language_mode})...")
    
    lang_code = "hi" if "Hindi" in language_mode else "en" if "English" in language_mode else None
    segments, _ = whisper_model.transcribe(input_path, language=lang_code, word_timestamps=True)
    
    all_words = []
    for segment in segments:
        for w in segment.words:
            all_words.append({"word": w.word, "start": w.start, "end": w.end})
            
    if not all_words:
        return None, None, "❌ No speech detected in video!"

    total_duration = all_words[-1]["end"]
    clip_duration = float(clip_duration)
    num_clips = int(num_clips)
    
    progress(0.6, desc=f"🧠 Scanning for top {num_clips} viral hooks...")
    step = 5.0 
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
    
    table_rows = []
    for i, win in enumerate(selected_windows):
        clip_label = f"Clip {i+1} ({int(win['start'])}s-{int(win['end'])}s)"
        clip_words = [w["word"].strip() for w in all_words if w["start"] >= win["start"] and w["end"] <= win["end"]]
        snippet = " ".join(clip_words[:15]) + "..." if clip_words else "No spoken text detected"
        table_rows.append([clip_label, round(win["start"], 1), round(win["end"], 1), snippet])
                
    progress(1.0, desc="✅ Hooks Extracted!")
    return table_rows, selected_windows, f"✅ Step 1 Complete! Extracted {len(selected_windows)} hooks."

# --- STEP 2: RENDERING ENGINE ---
def step2_render_clips(selected_file, selected_windows, auto_clean, progress=gr.Progress()):
    if not selected_windows:
        return None, "❌ Please run Step 1 (Detect Hooks) before rendering!"
    
    if auto_clean:
        clear_output_folder()
        
    input_path = os.path.join(INPUT_DIR, selected_file)
    generated_videos = []
    render_summary = []
    
    for i, win in enumerate(selected_windows):
        progress(0.1 + (0.8 * (i / len(selected_windows))), desc=f"🎬 Analyzing frames & rendering Clip {i+1} of {len(selected_windows)}...")
        
        start_t = win["start"]
        end_t = win["end"]
        clip_duration_real = end_t - start_t
        
        # Let the Auto-Director decide the framing!
        filter_chain, decision_label = get_auto_framing_filter(input_path, start_t, end_t)
        
        base_name = os.path.splitext(selected_file)[0]
        output_filename = f"viral_v4.3_{base_name}_clip{i+1}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        command = [
            "ffmpeg", "-y", "-ss", str(start_t), "-i", input_path, "-t", str(clip_duration_real),
            "-vf", filter_chain, "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", output_path
        ]
        
        subprocess.run(command)
        
        caption_text = f"Clip {i+1}: {int(start_t)}s - {int(end_t)}s ({decision_label})"
        generated_videos.append((output_path, caption_text))
        render_summary.append(f"⚡ {caption_text}")
        
    progress(1.0, desc="🚀 All Shorts Rendered Successfully!")
    return generated_videos, "\n".join(render_summary)

def refresh_file_list():
    files = get_input_videos()
    return gr.Dropdown(choices=files, value=files[0] if files else None)

custom_css = """
.stat-box { background: #1e1e2e; border: 2px solid #313244; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 10px; }
.stat-title { color: #cdd6f4; font-size: 14px; font-weight: bold; }
.stat-desc { color: #a6adc8; font-size: 12px; }
"""

with gr.Blocks(title="AI Short Video Clipper v4.3", css=custom_css) as demo:
    clip_windows_state = gr.State()
    
    gr.Markdown("# 🎬 AI Video Clipper v4.3: Fully Automated Auto-Director Engine")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Step 1: Target & Detect")
            with gr.Row():
                workspace_dropdown = gr.Dropdown(choices=get_input_videos(), label="Select Source Video", value=get_input_videos()[0] if get_input_videos() else None, scale=4)
                refresh_btn = gr.Button("🔄", scale=1)
            
            language_selector = gr.Dropdown(
                choices=["English", "Hindi", "Auto-Detect"],
                value="English",
                label="Audio Language Mode"
            )
            
            num_clips_input = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Viral Clips to Extract")
            clip_duration_input = gr.Slider(minimum=15, maximum=60, value=25, step=5, label="Clip Duration (seconds)")
            
            step1_btn = gr.Button("🎙️ Step 1: Scan Hooks & Extract Timestamps", variant="primary")
            status_box = gr.Textbox(label="System Status", value="⏳ Ready. Select a video and click Step 1.", interactive=False, lines=2)
            
            with gr.Row():
                stop_btn = gr.Button("🛑 Stop Active Process", variant="stop", scale=2)
            
            gr.Markdown("---")
            gr.Markdown("### 🎬 Step 2: Auto-Direct & Render")
            gr.HTML('<div class="stat-box"><div class="stat-title">🤖 AI Auto-Director Active</div><div class="stat-desc">The engine will automatically scan the clip. If it sees a solo speaker, it face-tracks them. If it sees a podcast (2+ people), it switches to blurred widescreen so nobody gets cut!</div></div>')
            
            auto_clean_checkbox = gr.Checkbox(value=True, label="Auto-clear outputs folder before new render")
            step2_btn = gr.Button("⚡ Step 2: Auto-Frame & Render Shorts", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Extracted Viral Hooks Overview")
            hooks_table = gr.Dataframe(
                headers=["Clip Label", "Start (s)", "End (s)", "Spoken Text Preview"],
                datatype=["str", "number", "number", "str"],
                type="array",
                interactive=False,
                row_count=(5, "dynamic"),
                col_count=(4, "fixed")
            )
            
            gr.Markdown("### 📺 Rendered Gallery")
            video_gallery = gr.Gallery(label="Generated 9:16 Shorts", show_label=True, columns=3, rows=2, object_fit="contain", height=500)

    # Event Listeners
    refresh_btn.click(fn=refresh_file_list, outputs=[workspace_dropdown])
    
    step1_event = step1_btn.click(
        fn=step1_detect,
        inputs=[workspace_dropdown, language_selector, num_clips_input, clip_duration_input],
        outputs=[hooks_table, clip_windows_state, status_box]
    )
    
    step2_event = step2_btn.click(
        fn=step2_render_clips,
        inputs=[workspace_dropdown, clip_windows_state, auto_clean_checkbox],
        outputs=[video_gallery, status_box]
    )
    
    stop_btn.click(
        fn=lambda: "🛑 Process interrupted and stopped by user!",
        inputs=None,
        outputs=[status_box],
        cancels=[step1_event, step2_event]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)