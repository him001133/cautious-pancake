import os
import re
import json
import uuid
import subprocess
import numpy as np
import requests
import time
import textwrap
import streamlit as st
import streamlit.components.v1 as components
from google import genai

# Page Configuration - Keep Sidebar Expanded by Default
st.set_page_config(page_title="AutoDirector AI", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# --- NATIVE UI: FOOTER & STYLING ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem !important; padding-bottom: 80px !important; }
    
    .custom-footer { position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0e1117; padding: 12px 0; text-align: center; border-top: 1px solid #2e303e; color: #6b7280; font-size: 0.85rem; z-index: 99999; }
    .custom-footer a { color: #a3a8b8; text-decoration: none; margin: 0 8px; transition: 0.2s; }
    .custom-footer a:hover { color: #ff4b4b; }
    </style>
    <div class="custom-footer">
        &copy; 2026 AutoDirector AI Studio. All rights reserved.  
        <span style="opacity: 0.3; margin: 0 5px;">|</span>
        <a href="privacy" target="_self">Privacy Policy</a>
        <span style="opacity: 0.3; margin: 0 5px;">|</span>
        <a href="terms" target="_self">Terms & Conditions</a>
    </div>
""", unsafe_allow_html=True)

# Session State Initialization
if "current_step" not in st.session_state: st.session_state.current_step = "workspace_home"
if "editor_mode" not in st.session_state: st.session_state.editor_mode = None
if "active_video" not in st.session_state: st.session_state.active_video = None
if "detected_clips" not in st.session_state: st.session_state.detected_clips = []
if "selected_clip_idx" not in st.session_state: st.session_state.selected_clip_idx = 0
if "framing_mode" not in st.session_state: st.session_state.framing_mode = "Vertical"
if "crop_x_percent" not in st.session_state: st.session_state.crop_x_percent = 50
if "current_subs" not in st.session_state: st.session_state.current_subs = []
if "last_clip_idx" not in st.session_state: st.session_state.last_clip_idx = -1
if "workflow_step" not in st.session_state: st.session_state.workflow_step = 1

# Studio NLE State Initialization
if "studio_media_pool" not in st.session_state: st.session_state.studio_media_pool = []
if "studio_v1_track" not in st.session_state: st.session_state.studio_v1_track = []
if "studio_a1_track" not in st.session_state: st.session_state.studio_a1_track = []
if "studio_selected_clip" not in st.session_state: st.session_state.studio_selected_clip = None

def reset_project_state():
    st.session_state.detected_clips = []
    st.session_state.selected_clip_idx = 0
    st.session_state.current_subs = []
    st.session_state.last_clip_idx = -1
    st.session_state.workflow_step = 1

# --- CUSTOM HEADER ---
nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([5, 1.2, 1, 1, 1.5], vertical_alignment="center")
with nav_c1: st.markdown("<h3 style='margin: 0;'>🎬 AutoDirector AI</h3>", unsafe_allow_html=True)
with nav_c2: 
    if st.button("🏠 Dashboard", key="nav_home", use_container_width=True):
        st.session_state.current_step = "workspace_home"
        st.rerun()
with nav_c3: st.page_link("pages/pricing.py", label="Pricing", icon="💳")
with nav_c4: st.page_link("pages/support.py", label="Support", icon="🎧")
st.markdown("---")

# --- SIDEBAR API KEY INPUT ---
st.sidebar.markdown("### 🔑 API Settings")
sidebar_key = st.sidebar.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password", key="sidebar_gemini_key")
if sidebar_key:
    os.environ["GEMINI_API_KEY"] = sidebar_key

st.sidebar.markdown("---")
if st.sidebar.button("➕ Edit New Video", type="primary", use_container_width=True):
    st.session_state.current_step = "workspace_home"
    st.session_state.editor_mode = None
    st.session_state.active_video = None
    reset_project_state()
    st.rerun()

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
PREVIEW_DIR = "previews"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# --- FFmpeg Path Helper ---
def get_ffmpeg_sub_path(filepath):
    return f"'{filepath}'"

# --- SUBTITLE GENERATOR LOGIC ---
def hex_to_ass_color(hex_str):
    h = hex_str.lstrip('#')
    if len(h) != 6: return "&H00FFFFFF"
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}"

def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def chunk_subtitles_by_words(raw_subs, max_words=4):
    new_subs = []
    for item in raw_subs:
        words = str(item['text']).strip().split()
        if not words: continue
        total_words = len(words)
        dur = item['end'] - item['start']
        
        if total_words <= max_words:
            new_subs.append(item)
        else:
            chunks = [words[i:i + max_words] for i in range(0, total_words, max_words)]
            chunk_dur = dur / len(chunks)
            for idx, chunk in enumerate(chunks):
                c_start = item['start'] + (idx * chunk_dur)
                c_end = c_start + chunk_dur
                new_subs.append({
                    "start": round(c_start, 2),
                    "end": round(c_end, 2),
                    "text": " ".join(chunk)
                })
    return new_subs

def parse_srt_file(content):
    """Parses standard SRT string content into our JSON array format."""
    subs = []
    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            times = lines[1].split(' --> ')
            if len(times) == 2:
                def t2s(t):
                    t = t.replace(',', '.')
                    h, m, s = t.split(':')
                    return int(h)*3600 + int(m)*60 + float(s)
                try:
                    start = t2s(times[0].strip())
                    end = t2s(times[1].strip())
                    text = " ".join(lines[2:])
                    text = re.sub('<[^<]+>', '', text) # Strip HTML tags
                    subs.append({"start": start, "end": end, "text": text})
                except: pass
    return subs

def create_ass_file(
    subs, 
    filepath="subs.ass", 
    font="Arial", 
    size=90, 
    color="&H0000FFFF", 
    outline="&H000000FF", 
    back_color="&H00000000",
    anim="None", 
    max_chars=20, 
    pos_y=1300,
    bold=True,
    italic=False,
    border_style=1,
    letter_spacing=0,
    uppercase=False
):
    b_val = -1 if bold else 0
    i_val = -1 if italic else 0

    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{color},&H000000FF,{outline},{back_color},{b_val},{i_val},0,0,100,100,{letter_spacing},0,{border_style},5,2,5,10,10,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    formatted_subs = chunk_subtitles_by_words(subs, max_words=max(2, max_chars // 5))

    for sub in formatted_subs:
        start_str = format_ass_time(sub['start'])
        end_str = format_ass_time(sub['end'])
        
        raw_text = str(sub['text']).replace('\n', ' ').strip()
        if uppercase:
            raw_text = raw_text.upper()
            
        wrapped_text = "\\N".join(textwrap.wrap(raw_text, width=max_chars))
        tags = f"\\pos(540,{pos_y})"
        
        if anim == "Fade In/Out":
            tags += "\\fad(200,200)"
        elif anim == "Pop In (Trending)":
            tags += "\\fscx50\\fscy50\\t(0,200,\\fscx100\\fscy100)"
        elif anim == "Bounce Drop":
            tags += "\\fscx150\\fscy150\\t(0,200,\\fscx100\\fscy100)"
        elif anim == "Dynamic Pop":
            tags += "\\fscx50\\fscy50\\t(0,150,\\fscx120\\fscy120)\\t(150,250,\\fscx100\\fscy100)"
        elif anim == "Rotate In":
            tags += "\\frz-15\\t(0,200,\\frz0)\\fad(200,0)"
            
        ass_content += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{{{tags}}}{wrapped_text}\n"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(ass_content)

def build_layout_ffmpeg_filter(mode, crop_x_pct=50, color_grade="None", aspect_ratio="9:16"):
    x_factor = crop_x_pct / 100.0
    
    if aspect_ratio == "9:16":
        out_w, out_h = 1080, 1920
        if mode == "Vertical": 
            base_filter = f"crop=ih*(9/16):ih:(iw-ih*(9/16))*{x_factor}:0,scale={out_w}:{out_h}"
        elif mode == "Split": 
            base_filter = f"split=2[top][bot];[top]crop=iw/2:ih:0:0,scale={out_w}:960[t_scaled];[bot]crop=iw/2:ih:iw/2:0,scale={out_w}:960[b_scaled];[t_scaled][b_scaled]vstack=inputs=2"
        elif mode == "Trio": 
            base_filter = f"split=3[p1][p2][p3];[p1]crop=iw/3:ih:0:0,scale={out_w}:640[v1];[p2]crop=iw/3:ih:iw/3:0,scale={out_w}:640[v2];[p3]crop=iw/3:ih:(iw/3)*2:0,scale={out_w}:640[v3];[v1][v2][v3]vstack=inputs=3"
        elif mode == "Spotlight": 
            base_filter = f"split[bg][fg];[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur=20:10[bg_blur];[fg]scale={out_w}:-1[fg_scale];[bg_blur][fg_scale]overlay=0:(H-h)/2"
        elif mode == "Centered": 
            base_filter = f"crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,scale={out_w}:{out_h}"
        elif mode == "Horizontal": 
            base_filter = f"split[bg][fg];[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur=25:10[b_blur];[fg]scale={out_w}:-1[f_scale];[b_blur][f_scale]overlay=0:(H-h)/2"
        else:
            base_filter = f"crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,scale={out_w}:{out_h}"
    elif aspect_ratio == "16:9":
        base_filter = "scale=1920:1080"
    elif aspect_ratio == "1:1":
        base_filter = f"crop=min(iw\,ih):min(iw\,ih):(iw-min(iw\,ih))*{x_factor}:(ih-min(iw\,ih))/2,scale=1080:1080"
    elif aspect_ratio == "4:5":
        base_filter = f"crop=ih*(4/5):ih:(iw-ih*(4/5))*{x_factor}:0,scale=1080:1350"
    else:
        # Default fallback for uncropped caption studio
        base_filter = "scale=1080:-2"

    grade_filter = ""
    if color_grade == "Cinematic":
        grade_filter = ",eq=contrast=1.15:saturation=1.2:gamma=0.9"
    elif color_grade == "Vibrant":
        grade_filter = ",eq=saturation=1.45:contrast=1.05"
    elif color_grade == "Warm Tone":
        grade_filter = ",colorbalance=rs=0.1:gs=-0.03:bs=-0.12"
    elif color_grade == "Cool Tone":
        grade_filter = ",colorbalance=rs=-0.1:gs=0.0:bs=0.15"
    elif color_grade == "B&W Dramatic":
        grade_filter = ",hue=s=0,eq=contrast=1.25"

    return base_filter + grade_filter

def generate_frame_preview(video_path, timestamp_s, filter_str):
    preview_path = os.path.join(PREVIEW_DIR, "preview_frame.jpg")
    cmd = ["ffmpeg", "-y", "-ss", str(timestamp_s), "-i", video_path, "-vf", filter_str, "-vframes", "1", "-q:v", "2", preview_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return preview_path if os.path.exists(preview_path) else None

def render_dynamic_tracker(step_num, container):
    container.empty()
    with container.container():
        step_cols = st.columns(4)
        with step_cols[0]: 
            st.markdown("✅ **1. Upload Video**" if step_num >= 1 else "⚪ 1. Upload Video")
        with step_cols[1]:
            if step_num > 2: st.markdown("✅ **2. Extract Audio**")
            elif step_num == 2: st.markdown("🔄 **2. Extract Audio**")
            else: st.markdown("⚪ 2. Extract Audio")
        with step_cols[2]:
            if step_num > 3: st.markdown("✅ **3. AI Analysis**")
            elif step_num == 3: st.markdown("🔄 **3. AI Analysis**")
            else: st.markdown("⚪ 3. AI Analysis")
        with step_cols[3]:
            if step_num >= 4: st.markdown("✅ **4. Export Clips**")
            else: st.markdown("⚪ 4. Export Clips")
        st.markdown("---")

# ==========================================
# SCREEN 0: UNIFIED DASHBOARD GATEWAY
# ==========================================
if st.session_state.current_step == "workspace_home":
    st.markdown("<h1 style='text-align: center; padding: 3rem 0 1rem 0;'>Select Your Workflow</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_space1, col1, col2, col3, col_space2 = st.columns([0.5, 2, 2, 2, 0.5], gap="large")
    
    with col1:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>✂️ AI Clipper</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #a3a8b8;'>Upload a long podcast. Let AI find viral hooks and auto-frame them for Shorts/Reels.</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Start Auto-Clipping", type="primary", use_container_width=True):
                st.session_state.editor_mode = "clipper"
                st.session_state.current_step = "upload"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>💬 Pro Caption Studio</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #a3a8b8;'>Add viral captions to an existing video. Upload an SRT or use AI Hinglish Generation.</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open Caption Studio", type="primary", use_container_width=True):
                st.session_state.editor_mode = "caption_studio"
                st.session_state.current_step = "caption_upload" 
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>🎬 Full Studio Editor</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #a3a8b8;'>Open the multi-track timeline. Drag, drop, trim, and manually build your video.</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open Studio Workspace", type="primary", use_container_width=True):
                st.session_state.editor_mode = "full_studio"
                st.session_state.current_step = "studio_editor" 
                st.rerun()

# ==========================================
# SCREEN 1.5: CAPTION STUDIO UPLOAD
# ==========================================
elif st.session_state.current_step == "caption_upload":
    st.markdown("<h2 style='text-align: center; padding: 2rem 0;'>Upload Video for Captioning</h2>", unsafe_allow_html=True)
    
    col_space1, col_main, col_space2 = st.columns([1, 2, 1])
    with col_main:
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload your finished video", type=["mp4", "mov", "webm", "mkv"])
            if uploaded_file is not None:
                save_path = os.path.join(INPUT_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.active_video = uploaded_file.name
                reset_project_state()
                st.session_state.current_step = "caption_editor"
                st.rerun()
            
            existing_files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".mp4", ".mov", ".mkv", ".webm"))]
            if existing_files:
                st.markdown("<div style='text-align: center; margin: 15px 0; color: #6b7280; font-size: 0.9rem;'>— or open a recent video —</div>", unsafe_allow_html=True)
                selected_existing = st.selectbox("Your Videos", existing_files, label_visibility="collapsed")
                if st.button("🚀 Process Selected Video", use_container_width=True):
                    st.session_state.active_video = selected_existing
                    reset_project_state()
                    st.session_state.current_step = "caption_editor"
                    st.rerun()

# ==========================================
# SCREEN 1: AI CLIPPER UPLOAD
# ==========================================
elif st.session_state.current_step == "upload":
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0 2rem 0;'>
            <h1 style='font-size: 3rem; font-weight: 900; margin-bottom: 15px; line-height: 1.2;'>
                Transform Long Videos into <br><span style='color: #ff4b4b;'>Viral AI Clips</span>
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    col_space1, col_main, col_space2 = st.columns([1, 2, 1])
    with col_main:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center; margin-bottom: 15px;'>📥 Add Video to Workspace</h3>", unsafe_allow_html=True)
            
            tab_upload, tab_youtube = st.tabs(["📁 Upload File", "🔗 YouTube Link"])
            
            with tab_upload:
                uploaded_file = st.file_uploader("Upload a new video", type=["mp4", "mov", "webm", "mkv"], label_visibility="collapsed")
                if uploaded_file is not None:
                    save_path = os.path.join(INPUT_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.active_video = uploaded_file.name
                    reset_project_state()
                    st.session_state.current_step = "dashboard"
                    st.rerun()
            
            with tab_youtube:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                yt_url = st.text_input("Paste YouTube URL here", placeholder="https://www.youtube.com/watch?v=...")
                if st.button("📥 Download via API", type="primary", use_container_width=True):
                    if yt_url:
                        with st.spinner("Connecting to external API & streaming to disk..."):
                            try:
                                api_url = "https://api.cobalt.tools/api/json"
                                headers = {"Accept": "application/json", "Content-Type": "application/json"}
                                payload = {"url": yt_url, "videoQuality": "1080"}
                                api_res = requests.post(api_url, json=payload, headers=headers)
                                
                                if api_res.status_code == 200:
                                    direct_link = api_res.json().get("url")
                                    if direct_link:
                                        filename = f"yt_{uuid.uuid4().hex[:8]}.mp4"
                                        save_path = os.path.join(INPUT_DIR, filename)
                                        with requests.get(direct_link, stream=True) as r:
                                            r.raise_for_status()
                                            with open(save_path, 'wb') as f:
                                                for chunk in r.iter_content(chunk_size=8192):
                                                    f.write(chunk)
                                        st.session_state.active_video = filename
                                        reset_project_state()
                                        st.session_state.current_step = "dashboard"
                                        st.rerun()
                                    else:
                                        st.error("No download link found.")
                                else:
                                    st.error(f"API Error ({api_res.status_code})")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Please paste a valid YouTube link first.")
            
            existing_files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".mp4", ".mov", ".mkv", ".webm"))]
            if existing_files:
                st.markdown("<div style='text-align: center; margin: 15px 0; color: #6b7280; font-size: 0.9rem;'>— or open a recent project —</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    selected_existing = st.selectbox("Your Videos", existing_files, label_visibility="collapsed")
                    if st.button("🚀 Process Selected Video", use_container_width=True):
                        st.session_state.active_video = selected_existing
                        reset_project_state()
                        st.session_state.current_step = "dashboard"
                        st.rerun()

# ==========================================
# SCREEN 2 & 3: AI CLIP DASHBOARD
# ==========================================
elif st.session_state.current_step == "dashboard":
    tracker_placeholder = st.empty()
    render_dynamic_tracker(st.session_state.workflow_step, tracker_placeholder)

    top_col1, top_col2 = st.columns([1, 8])
    with top_col1:
        if st.button("← Back"):
            st.session_state.current_step = "upload"
            st.rerun()
    with top_col2: st.subheader(f"Project: {st.session_state.active_video}")

    video_path = os.path.join(INPUT_DIR, st.session_state.active_video)
    col_left, col_right = st.columns([3, 2], gap="medium")
    
    with col_left: st.video(video_path)
        
    with col_right:
        st.markdown("### ✦ Find a moment")
        
        main_key = st.text_input("🔑 Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password", help="Enter your Gemini key here or in the sidebar to activate AI Transcription.")
        if main_key:
            os.environ["GEMINI_API_KEY"] = main_key

        prompt_input = st.text_input("Describe what you're looking for or hit Auto-Scan", placeholder='"the funniest part", "when they get emotional"...')
        
        c_cfg1, c_cfg2 = st.columns(2)
        with c_cfg1:
            num_clips = st.slider("Number of clips", min_value=1, max_value=250, value=4)
        with c_cfg2:
            dur_range = st.slider("Target Duration (sec)", min_value=10, max_value=120, value=(15, 60))
            
        if st.button("⚡ Generate AI Clips", type="primary", use_container_width=True):
            st.session_state.workflow_step = 2
            render_dynamic_tracker(2, tracker_placeholder)
            
            start_time = time.time()
            ESTIMATED_TOTAL_SECONDS = 90 
            
            def get_progress_text(step_name, percent):
                elapsed = int(time.time() - start_time)
                time_left = max(0, int((elapsed / max(1, percent)) * (100 - percent)) if percent > 0 else ESTIMATED_TOTAL_SECONDS)
                return f"{percent}% | {step_name} | Elapsed: {elapsed // 60}:{elapsed % 60:02d} | ETA: ~{time_left // 60}:{time_left % 60:02d}"

            progress_bar = st.progress(0, text=get_progress_text("Initializing...", 0))
            
            try:
                progress_bar.progress(10, text=get_progress_text("Cleaning up temporary files...", 10))
                audio_path = os.path.join(INPUT_DIR, "temp_audio.mp3")
                if os.path.exists(audio_path):
                    try: os.remove(audio_path)
                    except: pass

                progress_bar.progress(25, text=get_progress_text("Extracting audio via FFmpeg...", 25))
                cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k", audio_path]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if res.returncode != 0:
                    st.error("Audio extraction failed.")
                else:
                    active_key = os.environ.get("GEMINI_API_KEY")
                    if not active_key:
                        st.warning("⚠️ No Gemini API Key provided. Please enter your key above or in the sidebar.")
                    else:
                        progress_bar.progress(40, text=get_progress_text("Uploading audio securely to Gemini...", 40))
                        client = genai.Client(api_key=active_key)
                        
                        uploaded_audio = client.files.upload(file=audio_path)
                        time.sleep(2) 
                        
                        progress_bar.progress(60, text=get_progress_text("Gemini natively listening & transcribing...", 60))
                        st.session_state.workflow_step = 3
                        render_dynamic_tracker(3, tracker_placeholder)
                        
                        prompt = f"""You are a master video editor. Listen to the uploaded audio.
Find the {num_clips} most engaging, viral-worthy short clips (e.g., strong hooks, funny moments, or emotional peaks).
Each clip MUST be between {dur_range[0]} and {dur_range[1]} seconds long.

For each clip, you must provide:
1. The exact 'start' and 'end' timestamps in seconds.
2. A catchy 'title'.
3. The EXACT word-for-word transcription for that specific clip, broken down into short subtitle segments ('subs'). Each subtitle segment should be 2 to 4 seconds long, and its timestamps must be RELATIVE to the start of the clip (i.e., the first sub starts at 0.0).

Return ONLY a valid JSON array matching this exact structure, with NO extra markdown formatting:
[
  {{
    "title": "Example Title",
    "start": 45.0,
    "end": 65.0,
    "subs": [
      {{"start": 0.0, "end": 2.5, "text": "My people go surfing,"}},
      {{"start": 2.5, "end": 5.0, "text": "and they always tell me"}}
    ]
  }}
]"""
                        try:
                            res_gemini = client.models.generate_content(
                                model='gemini-1.5-flash-latest', 
                                contents=[uploaded_audio, prompt]
                            )
                            clean_json = re.sub(r'```json\n|\n```|```', '', res_gemini.text).strip()
                            st.session_state.detected_clips = json.loads(clean_json)
                        except Exception as e: 
                            st.error(f"Gemini API Error: {e}")
                        finally:
                            try: client.files.delete(name=uploaded_audio.name)
                            except: pass

                    if not st.session_state.detected_clips:
                        st.warning("⚠️ Gemini analysis failed. Falling back to mathematically sliced clips.")
                        try:
                            dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
                            total_dur = float(subprocess.check_output(dur_cmd).decode('utf-8').strip())
                        except:
                            total_dur = 300.0 
                            
                        slice_len = dur_range[0]
                        st.session_state.detected_clips = [{
                            "title": f"Fallback Clip #{i+1}", 
                            "start": float(i*slice_len), 
                            "end": float(i*slice_len + slice_len - 2),
                            "subs": [{"start": 0.0, "end": 3.0, "text": "AI transcription failed."}]
                        } for i in range(min(num_clips, int(total_dur // slice_len)))]
                    
                    min_req_dur = float(dur_range[0])
                    max_req_dur = float(dur_range[1])
                    for clip in st.session_state.detected_clips:
                        c_dur = clip['end'] - clip['start']
                        if c_dur < min_req_dur:
                            needed = min_req_dur - c_dur
                            clip['start'] = max(0.0, round(clip['start'] - (needed / 2.0), 2))
                            clip['end'] = round(clip['start'] + min_req_dur, 2)
                        elif c_dur > max_req_dur:
                            clip['end'] = round(clip['start'] + max_req_dur, 2)
                        
                    st.session_state.workflow_step = 4
                    render_dynamic_tracker(4, tracker_placeholder)
                    progress_bar.progress(100, text=get_progress_text("Done!", 100))
                    time.sleep(1)
                    progress_bar.empty()
                    st.rerun()

            except Exception as e:
                progress_bar.empty()
                st.error(f"Failed: {e}")
        
        if st.session_state.detected_clips:
            st.markdown(f"#### Clips ({len(st.session_state.detected_clips)})")
            for idx, clip in enumerate(st.session_state.detected_clips):
                with st.container(border=True):
                    st.markdown(f"**{idx+1}. {clip['title']}**")
                    st.caption(f"⏱️ {int(clip['start'])}s → {int(clip['end'])}s ({int(clip['end'] - clip['start'])}s)")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Clip Horizontal 📺", key=f"horiz_{idx}"):
                            st.session_state.selected_clip_idx = idx; st.session_state.framing_mode = "Horizontal"; st.session_state.current_step = "editor"; st.rerun()
                    with c2:
                        if st.button("Clip Vertical 📱", key=f"vert_{idx}", type="primary"):
                            st.session_state.selected_clip_idx = idx; st.session_state.framing_mode = "Vertical"; st.session_state.current_step = "editor"; st.rerun()

# ==========================================
# SCREEN 4: POSITION & SUBTITLES (CLIPPER)
# ==========================================
elif st.session_state.current_step == "editor":
    tracker_placeholder = st.empty()
    render_dynamic_tracker(4, tracker_placeholder)
    
    top_col1, top_col2 = st.columns([1, 8])
    with top_col1:
        if st.button("← Back"):
            st.session_state.current_step = "dashboard"
            st.rerun()
    with top_col2: st.subheader("Clip Editor")

    clip = st.session_state.detected_clips[st.session_state.selected_clip_idx]
    video_path = os.path.join(INPUT_DIR, st.session_state.active_video)
    
    st.markdown("### Framing Strategy & Export Ratio")
    c_ratio1, c_ratio2 = st.columns([2, 4])
    with c_ratio1:
        export_aspect = st.selectbox("Output Ratio", ["9:16 (Shorts/Reels)", "16:9 (YouTube)", "1:1 (Square)", "4:5 (Instagram)"], index=0)
        aspect_code = export_aspect.split(" ")[0]
    with c_ratio2:
        preset_cols = st.columns(6)
        modes = ["Vertical", "Split", "Trio", "Spotlight", "Centered", "Horizontal"]
        for i, m in enumerate(modes):
            with preset_cols[i]:
                btn_type = "primary" if st.session_state.framing_mode == m else "secondary"
                if st.button(m, key=f"mode_{m}", type=btn_type, use_container_width=True):
                    st.session_state.framing_mode = m; st.rerun()

    col_edit, col_prev = st.columns([3, 2], gap="large")
    with col_edit:
        st.markdown("#### Adjust Camera & Video Filters")
        st.session_state.crop_x_percent = st.slider("Horizontal Position (X-Axis)", min_value=0, max_value=100, value=st.session_state.crop_x_percent)
        
        c_flt1, c_flt2 = st.columns(2)
        with c_flt1:
            color_grade_preset = st.selectbox("Color Grade Filter", ["None", "Cinematic", "Vibrant", "Warm Tone", "Cool Tone", "B&W Dramatic"], index=0)
        with c_flt2:
            enable_audio_boost = st.checkbox("Normalize & Boost Audio", value=True, help="Applies FFmpeg loudnorm filter for studio audio levels.")

        st.markdown("---")
        
        st.markdown("#### 💬 Subtitle Engine")
        enable_subs = st.checkbox("Burn Viral Subtitles onto video", value=True)
        
        if enable_subs:
            preset_choice = st.selectbox("🎯 Quick Preset Templates", [
                "Custom", 
                "Hormozi Viral (Yellow/Black)", 
                "Clean White Banner", 
                "Neon Cyberpunk (Cyan)", 
                "Minimalist Bold"
            ], index=1)

            f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Impact", "#FFFF00", "#000000", "#000000", "Outline", True, True
            
            if preset_choice == "Hormozi Viral (Yellow/Black)":
                f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Impact", "#FFFF00", "#000000", "#000000", "Outline", True, True
            elif preset_choice == "Clean White Banner":
                f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Arial", "#FFFFFF", "#000000", "#000000", "Opaque Box", True, False
            elif preset_choice == "Neon Cyberpunk (Cyan)":
                f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Courier New", "#00FFFF", "#FF00FF", "#000000", "Outline", True, True
            elif preset_choice == "Minimalist Bold":
                f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Verdana", "#FFFFFF", "#000000", "#000000", "Outline", True, False

            with st.expander("🎨 Advanced Styling Options", expanded=True):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    sub_font = st.selectbox("Font Family", ["Impact", "Arial", "Courier New", "Verdana", "Tahoma"], index=["Impact", "Arial", "Courier New", "Verdana", "Tahoma"].index(f_font))
                    sub_color = st.color_picker("Primary Text Color", f_color) 
                    sub_border_type = st.radio("Style Mode", ["Outline + Shadow", "Opaque Box"], index=0 if f_border_st == "Outline" else 1, horizontal=True)
                with col_f2:
                    sub_anim = st.selectbox("Animation Effect", ["None", "Pop In (Trending)", "Fade In/Out", "Bounce Drop", "Dynamic Pop", "Rotate In"], index=1)
                    sub_outline = st.color_picker("Outline / Border Color", f_outline)
                    sub_bg_color = st.color_picker("Box Background Color", f_bg) if sub_border_type == "Opaque Box" else "#000000"

                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    sub_bold = st.checkbox("Bold Text", value=f_bold)
                with col_t2:
                    sub_italic = st.checkbox("Italic Text", value=False)
                with col_t3:
                    sub_uppercase = st.checkbox("ALL CAPS", value=f_uppercase)

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    sub_size = st.slider("Font Size", 30, 150, 90)
                with col_s2:
                    sub_chars = st.slider("Max Chars/Line", 10, 50, 18)
                with col_s3:
                    sub_pos_y = st.slider("Vertical Pos (Y)", 200, 1800, 1300)
                with col_s4:
                    sub_spacing = st.slider("Letter Spacing", 0, 20, 0)

        if st.session_state.last_clip_idx != st.session_state.selected_clip_idx:
            st.session_state.current_subs = clip.get('subs', [])
            st.session_state.last_clip_idx = st.session_state.selected_clip_idx

        edited_subs = []
        if enable_subs and st.session_state.current_subs:
            edited_subs = st.data_editor(st.session_state.current_subs, column_config={
                "start": st.column_config.NumberColumn("Start (s)", format="%.2f", disabled=True),
                "end": st.column_config.NumberColumn("End (s)", format="%.2f", disabled=True),
                "text": st.column_config.TextColumn("Subtitle Text")
            }, num_rows="dynamic", use_container_width=True)

        st.markdown("---")
        if st.button("⚡ Render Final Short", type="primary", use_container_width=True):
            out_path = os.path.join(OUTPUT_DIR, f"clipzi_{st.session_state.framing_mode}_{st.session_state.selected_clip_idx+1}.mp4")
            filter_str = build_layout_ffmpeg_filter(st.session_state.framing_mode, st.session_state.crop_x_percent, color_grade_preset, aspect_code)
            
            if enable_subs and edited_subs:
                ass_color = hex_to_ass_color(sub_color)
                ass_outline = hex_to_ass_color(sub_outline)
                ass_bg = hex_to_ass_color(sub_bg_color)
                b_style_val = 3 if sub_border_type == "Opaque Box" else 1
                
                create_ass_file(
                    edited_subs, 
                    filepath="subs.ass", 
                    font=sub_font, 
                    size=sub_size, 
                    color=ass_color, 
                    outline=ass_outline, 
                    back_color=ass_bg,
                    anim=sub_anim, 
                    max_chars=sub_chars, 
                    pos_y=sub_pos_y,
                    bold=sub_bold,
                    italic=sub_italic,
                    border_style=b_style_val,
                    letter_spacing=sub_spacing,
                    uppercase=sub_uppercase
                )
                filter_str += f",subtitles={get_ffmpeg_sub_path('subs.ass')}"
                
            cmd = ["ffmpeg", "-y", "-ss", str(clip['start']), "-i", video_path, "-t", str(clip['end'] - clip['start']), "-vf", filter_str]
            if enable_audio_boost: cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
            cmd += ["-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", out_path]
            
            with st.spinner("Rendering video..."):
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode != 0: st.error(f"FFmpeg failed:\n\n{res.stderr}")
                else:
                    st.success(f"Saved to {out_path}")
                    with open(out_path, 'rb') as v_file: st.video(v_file.read())

    with col_prev:
        st.markdown("#### Live Preview Frame")
        preview_filter = build_layout_ffmpeg_filter(st.session_state.framing_mode, st.session_state.crop_x_percent, color_grade_preset, aspect_code)
        
        if enable_subs and edited_subs:
            ass_color = hex_to_ass_color(sub_color)
            ass_outline = hex_to_ass_color(sub_outline)
            ass_bg = hex_to_ass_color(sub_bg_color)
            b_style_val = 3 if sub_border_type == "Opaque Box" else 1
            
            create_ass_file(
                edited_subs, 
                filepath="subs.ass", 
                font=sub_font, 
                size=sub_size, 
                color=ass_color, 
                outline=ass_outline, 
                back_color=ass_bg,
                anim=sub_anim, 
                max_chars=sub_chars, 
                pos_y=sub_pos_y,
                bold=sub_bold,
                italic=sub_italic,
                border_style=b_style_val,
                letter_spacing=sub_spacing,
                uppercase=sub_uppercase
            )
            preview_filter += f",subtitles={get_ffmpeg_sub_path('subs.ass')}"
            
        preview_time = clip['start'] + 1.0
        if preview_time > clip['end']: preview_time = clip['start']
        
        preview_img = generate_frame_preview(video_path, preview_time, preview_filter)
        if preview_img: 
            with open(preview_img, "rb") as img_file: st.image(img_file.read(), caption=f"Ratio: {aspect_code} | Mode: {st.session_state.framing_mode}", use_container_width=True)


# ==========================================
# SCREEN 6: PRO CAPTION STUDIO
# ==========================================
elif st.session_state.current_step == "caption_editor":
    top_col1, top_col2 = st.columns([1, 8])
    with top_col1:
        if st.button("← Change Video"):
            st.session_state.current_step = "caption_upload"
            st.rerun()
    with top_col2: st.subheader("Pro Caption Studio")

    video_path = os.path.join(INPUT_DIR, st.session_state.active_video)
    
    col_prev, col_edit = st.columns([2, 3], gap="large")
    
    with col_edit:
        st.markdown("### 1. Load Subtitles")
        tab_ai, tab_srt = st.tabs(["🎙️ AI Hinglish Gen (Gemini)", "📁 Upload .SRT File"])
        
        with tab_ai:
            st.info("Hinglish Engine: Uses Gemini to listen to audio and output perfectly spelled Latin-script Hindi-English phonetics.")
            if st.button("Generate Hinglish Captions", type="primary", use_container_width=True):
                active_key = os.environ.get("GEMINI_API_KEY")
                if not active_key:
                    st.error("Please add your Gemini API Key in the sidebar.")
                else:
                    with st.spinner("Extracting audio..."):
                        audio_path = os.path.join(INPUT_DIR, "temp_audio_caption.mp3")
                        if os.path.exists(audio_path): os.remove(audio_path)
                        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                    with st.spinner("Uploading to Gemini & Transcribing Hinglish..."):
                        try:
                            client = genai.Client(api_key=active_key)
                            uploaded_audio = client.files.upload(file=audio_path)
                            time.sleep(2)
                            
                            prompt = """You are an expert bilingual transcriber specializing in 'Hinglish' (a natural mix of Hindi and English).
Listen to the uploaded audio carefully. Write the exact word-for-word transcription.
CRITICAL: For Hindi words, use standard English alphabets (Latin script) with accurate phonetic spelling (e.g., "bhai", "kya", "kar", "rahe", "ho").
Break the transcription into short, fast-paced subtitle segments of 2 to 4 seconds each.

Return ONLY a valid JSON array matching this exact format:
[
  {"start": 0.0, "end": 2.5, "text": "bhai ye video edit karna"},
  {"start": 2.5, "end": 5.0, "text": "is very important for the campaign"}
]"""
                            res_gemini = client.models.generate_content(model='gemini-1.5-flash-latest', contents=[uploaded_audio, prompt])
                            clean_json = re.sub(r'```json\n|\n```|```', '', res_gemini.text).strip()
                            st.session_state.current_subs = json.loads(clean_json)
                            try: client.files.delete(name=uploaded_audio.name)
                            except: pass
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        with tab_srt:
            uploaded_srt = st.file_uploader("Upload .SRT File", type=["srt"])
            if uploaded_srt:
                content = uploaded_srt.getvalue().decode("utf-8")
                if st.button("Parse SRT", use_container_width=True):
                    st.session_state.current_subs = parse_srt_file(content)
                    st.rerun()

        st.markdown("### 2. Editor & Styling")
        edited_subs = []
        if st.session_state.current_subs:
            edited_subs = st.data_editor(st.session_state.current_subs, column_config={
                "start": st.column_config.NumberColumn("Start (s)", format="%.2f"),
                "end": st.column_config.NumberColumn("End (s)", format="%.2f"),
                "text": st.column_config.TextColumn("Subtitle Text")
            }, num_rows="dynamic", use_container_width=True, height=200)
            
            st.markdown("---")
            preset_choice = st.selectbox("🎯 Quick Preset Templates", [
                "Custom", "Hormozi Viral (Yellow/Black)", "Clean White Banner", "Neon Cyberpunk (Cyan)", "Minimalist Bold"
            ], index=1)

            f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Impact", "#FFFF00", "#000000", "#000000", "Outline", True, True
            if preset_choice == "Clean White Banner":
                f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Arial", "#FFFFFF", "#000000", "#000000", "Opaque Box", True, False
            elif preset_choice == "Neon Cyberpunk (Cyan)":
                f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Courier New", "#00FFFF", "#FF00FF", "#000000", "Outline", True, True
            elif preset_choice == "Minimalist Bold":
                f_font, f_color, f_outline, f_bg, f_border_st, f_bold, f_uppercase = "Verdana", "#FFFFFF", "#000000", "#000000", "Outline", True, False

            with st.expander("🎨 Advanced Styling Options", expanded=True):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    sub_font = st.selectbox("Font Family", ["Impact", "Arial", "Courier New", "Verdana", "Tahoma"], index=["Impact", "Arial", "Courier New", "Verdana", "Tahoma"].index(f_font))
                    sub_color = st.color_picker("Primary Text Color", f_color) 
                    sub_border_type = st.radio("Style Mode", ["Outline + Shadow", "Opaque Box"], index=0 if f_border_st == "Outline" else 1, horizontal=True)
                with col_f2:
                    sub_anim = st.selectbox("Animation Effect", ["None", "Pop In (Trending)", "Fade In/Out", "Bounce Drop", "Dynamic Pop", "Rotate In"], index=1)
                    sub_outline = st.color_picker("Outline / Border Color", f_outline)
                    sub_bg_color = st.color_picker("Box Background Color", f_bg) if sub_border_type == "Opaque Box" else "#000000"

                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1: sub_bold = st.checkbox("Bold Text", value=f_bold)
                with col_t2: sub_italic = st.checkbox("Italic Text", value=False)
                with col_t3: sub_uppercase = st.checkbox("ALL CAPS", value=f_uppercase)

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1: sub_size = st.slider("Font Size", 30, 150, 90)
                with col_s2: sub_chars = st.slider("Max Chars/Line", 10, 50, 18)
                with col_s3: sub_pos_y = st.slider("Vertical Pos (Y)", 200, 1800, 1300)
                with col_s4: sub_spacing = st.slider("Letter Spacing", 0, 20, 0)
                
            st.markdown("---")
            if st.button("⚡ Render Video with Captions", type="primary", use_container_width=True):
                out_path = os.path.join(OUTPUT_DIR, f"captioned_{uuid.uuid4().hex[:6]}.mp4")
                
                # Keep original aspect ratio, just attach subs
                filter_str = "scale=-1:-1" 
                
                ass_color = hex_to_ass_color(sub_color)
                ass_outline = hex_to_ass_color(sub_outline)
                ass_bg = hex_to_ass_color(sub_bg_color)
                b_style_val = 3 if sub_border_type == "Opaque Box" else 1
                
                create_ass_file(
                    edited_subs, filepath="subs.ass", font=sub_font, size=sub_size, 
                    color=ass_color, outline=ass_outline, back_color=ass_bg, anim=sub_anim, 
                    max_chars=sub_chars, pos_y=sub_pos_y, bold=sub_bold, italic=sub_italic, 
                    border_style=b_style_val, letter_spacing=sub_spacing, uppercase=sub_uppercase
                )
                filter_str += f",subtitles={get_ffmpeg_sub_path('subs.ass')}"
                
                cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", filter_str, "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "copy", out_path]
                
                with st.spinner("Rendering final captioned video..."):
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if res.returncode != 0: st.error(f"FFmpeg failed:\n\n{res.stderr}")
                    else:
                        st.success(f"Saved to {out_path}")
                        with open(out_path, 'rb') as v_file: st.video(v_file.read())

    with col_prev:
        st.markdown("#### Preview (Original Aspect Ratio)")
        if st.session_state.current_subs:
            preview_filter = "scale=-1:-1"
            ass_color = hex_to_ass_color(sub_color)
            ass_outline = hex_to_ass_color(sub_outline)
            ass_bg = hex_to_ass_color(sub_bg_color)
            b_style_val = 3 if sub_border_type == "Opaque Box" else 1
            
            create_ass_file(
                edited_subs, filepath="subs.ass", font=sub_font, size=sub_size, 
                color=ass_color, outline=ass_outline, back_color=ass_bg, anim=sub_anim, 
                max_chars=sub_chars, pos_y=sub_pos_y, bold=sub_bold, italic=sub_italic, 
                border_style=b_style_val, letter_spacing=sub_spacing, uppercase=sub_uppercase
            )
            preview_filter += f",subtitles={get_ffmpeg_sub_path('subs.ass')}"
            
            # Preview at 1s into the first subtitle
            preview_time = edited_subs[0]['start'] + 0.5
            
            preview_img = generate_frame_preview(video_path, preview_time, preview_filter)
            if preview_img: 
                with open(preview_img, "rb") as img_file: st.image(img_file.read(), use_container_width=True)
        else:
            st.video(video_path)

# ==========================================
# SCREEN 5: FULL STUDIO EDITOR (NLE)
# ==========================================
elif st.session_state.current_step == "studio_editor":
    top_col1, top_col2 = st.columns([1, 8])
    with top_col1:
        if st.button("← Back to Workspace"):
            st.session_state.current_step = "workspace_home"
            st.rerun()
    with top_col2: 
        st.subheader("Studio Workspace")
    
    col_eff, col_prog, col_bin = st.columns([1, 2, 1])
    
    with col_eff:
        with st.container(border=True, height=350):
            st.markdown("**🎛️ Effect Controls**")
            st.divider()
            if st.session_state.studio_selected_clip:
                st.caption(f"Editing: {st.session_state.studio_selected_clip['file']}")
            else:
                st.caption("Select a clip on the timeline to edit properties.")
            
    with col_prog:
        with st.container(border=True, height=350):
            st.markdown("**📺 Program Monitor**")
            
            st.markdown("""
                <style>
                    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
                        overflow-y: hidden !important;
                    }
                    div[data-testid="stVideo"], 
                    div[data-testid="stVideo"] > div,
                    div[data-testid="stVideo"] video {
                        height: 240px !important;
                        max-height: 240px !important; 
                        width: 100% !important;
                        object-fit: contain !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            active_preview = None
            if st.session_state.studio_selected_clip:
                active_preview = st.session_state.studio_selected_clip["file"]
            elif st.session_state.studio_v1_track:
                active_preview = st.session_state.studio_v1_track[0]["file"]
                
            if active_preview and active_preview.endswith((".mp4", ".mov", ".webm")):
                st.video(os.path.join(INPUT_DIR, active_preview))
            else:
                st.markdown("<div style='height: 240px; background-color: #000; width: 100%; border: 1px solid #333; display: flex; align-items: center; justify-content: center;'><span style='color: #444'>NO MEDIA</span></div>", unsafe_allow_html=True)
            
    with col_bin:
        with st.container(border=True, height=350):
            st.markdown("**📁 Project Bin**")
            upload_studio = st.file_uploader("Import Media", type=["mp4", "mov", "wav", "mp3", "jpg", "png"], label_visibility="collapsed")
            
            if upload_studio is not None:
                if upload_studio.name not in st.session_state.studio_media_pool:
                    save_path = os.path.join(INPUT_DIR, upload_studio.name)
                    with open(save_path, "wb") as f: f.write(upload_studio.getbuffer())
                    st.session_state.studio_media_pool.append(upload_studio.name)
                    st.rerun()
            
            st.divider()
            st.markdown("**Assets:**")
            if st.session_state.studio_media_pool:
                for asset in st.session_state.studio_media_pool:
                    c1, c2 = st.columns([4, 1])
                    with c1: st.markdown(f"<div style='font-size: 13px; padding: 2px 0; color: #ccc;'>🎞️ {asset[:15]}...</div>", unsafe_allow_html=True)
                    with c2:
                        if st.button("➕", key=f"add_{asset}", help="Add to Timeline"):
                            is_video = asset.endswith((".mp4", ".mov", ".jpg", ".png"))
                            track_len = len(st.session_state.studio_v1_track) if is_video else len(st.session_state.studio_a1_track)
                            
                            new_clip = {
                                "id": str(uuid.uuid4())[:8],
                                "file": asset,
                                "left": 100 + (track_len * 210), 
                                "width": 200
                            }
                            
                            if is_video: st.session_state.studio_v1_track.append(new_clip)
                            else: st.session_state.studio_a1_track.append(new_clip)
                            st.rerun()
            else:
                st.caption("No media imported yet.")

    st.markdown("<br>", unsafe_allow_html=True)

    v1_html = ""
    for clip in st.session_state.studio_v1_track:
        v1_html += f'<div class="clip video draggable" id="{clip["id"]}" style="left: {clip["left"]}px; width: {clip["width"]}px;" onclick="selectClip(this)">{clip["file"]}</div>\n'

    a1_html = ""
    for clip in st.session_state.studio_a1_track:
        a1_html += f'<div class="clip audio draggable" id="{clip["id"]}" style="left: {clip["left"]}px; width: {clip["width"]}px;" onclick="selectClip(this)">{clip["file"]}</div>\n'

    timeline_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <style>
        :root {
            --bg-panel: #252526;
            --border: #3e3e42;
            --accent: #007acc;
            --track-v: #4a5c68;
            --track-a: #3a6858;
        }
        body, html {
            margin: 0; padding: 0; 
            background-color: transparent;
            color: #cccccc;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            user-select: none;
        }
        .timeline { 
            background-color: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 4px;
            display: flex; flex-direction: column;
            height: 350px;
        }
        .panel-header {
            background-color: #2d2d30;
            padding: 5px 10px;
            font-size: 13px;
            font-weight: bold;
            border-bottom: 1px solid var(--border);
            color: #fff;
            display: flex; justify-content: space-between;
        }
        .timeline-toolbar {
            height: 30px; background-color: #333333;
            display: flex; align-items: center; padding: 0 10px; gap: 10px;
            border-bottom: 1px solid var(--border);
        }
        .tool-icon {
            width: 20px; height: 20px; background-color: #555;
            border-radius: 3px; cursor: pointer; text-align: center;
            line-height: 20px; font-size: 12px;
        }
        .tool-icon:hover, .tool-icon.active { background-color: var(--accent); color: white;}
        .track-container {
            flex-grow: 1; display: flex; flex-direction: column;
            padding: 5px 0; position: relative; overflow-x: auto;
        }
        .track {
            height: 40px; margin-bottom: 2px; background-color: #2a2a2a;
            display: flex; position: relative;
            border-top: 1px solid #333; border-bottom: 1px solid #111;
        }
        .track-header {
            width: 80px; background-color: #333;
            border-right: 1px solid var(--border); display: flex;
            align-items: center; justify-content: center;
            font-size: 11px; font-weight: bold; z-index: 20; position: sticky; left: 0;
        }
        .clip {
            position: absolute; height: 100%; border-radius: 2px;
            border: 1px solid #111; display: flex; align-items: center;
            padding-left: 5px; font-size: 10px; white-space: nowrap;
            overflow: hidden; cursor: grab; transition: box-shadow 0.1s;
        }
        .clip.selected { border: 1px solid #fff; box-shadow: 0 0 5px rgba(255,255,255,0.5); z-index: 15; }
        .clip:active { cursor: grabbing; z-index: 16; }
        .clip.video { background-color: var(--track-v); }
        .clip.audio { background-color: var(--track-a); }
        
        .playhead {
            position: absolute; top: 0; bottom: 0; width: 2px;
            background-color: red; left: 80px; z-index: 25; cursor: ew-resize;
        }
        .playhead::before {
            content: ''; position: absolute; top: -5px; left: -4px;
            width: 10px; height: 10px; background-color: red;
            clip-path: polygon(0 0, 100% 0, 50% 100%); cursor: ew-resize;
        }
    </style>
    </head>
    <body>
        <div class="timeline">
            <div class="panel-header">
                <span>Sequence 01</span>
                <span id="timecode" style="color: #888;">00:00:00:00</span>
            </div>
            <div class="timeline-toolbar">
                <div class="tool-icon active" title="Selection Tool (V)">↖</div>
                <div class="tool-icon" title="Razor Tool (C)">✂</div>
                <div class="tool-icon" title="Ripple Edit (B)">⇹</div>
            </div>
            
            <div class="track-container" id="timeline-container">
                <div class="playhead" id="playhead"></div>
                
                <div class="track">
                    <div class="track-header">V2</div>
                </div>
                <div class="track">
                    <div class="track-header">V1</div>
                    <!-- V1_CLIPS -->
                </div>
                
                <div style="height: 10px; border-bottom: 2px solid #111;"></div>
                
                <div class="track">
                    <div class="track-header">A1</div>
                    <!-- A1_CLIPS -->
                </div>
                <div class="track">
                    <div class="track-header">A2</div>
                </div>
            </div>
        </div>

        <script>
            const getStreamlitVideo = () => {
                try {
                    return window.parent.document.querySelector('video');
                } catch(e) { return null; }
            };

            const PX_PER_SEC = 20; 

            function formatTimecode(seconds) {
                const h = Math.floor(seconds / 3600);
                const m = Math.floor((seconds % 3600) / 60);
                const s = Math.floor(seconds % 60);
                const f = Math.floor((seconds % 1) * 30); 
                return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}:${f.toString().padStart(2, '0')}`;
            }

            setInterval(() => {
                if(!isScrubbing) {
                    const vid = getStreamlitVideo();
                    if(vid && !vid.paused) {
                        const newPos = 80 + (vid.currentTime * PX_PER_SEC);
                        document.getElementById('playhead').style.left = newPos + 'px';
                        document.getElementById('timecode').innerText = formatTimecode(vid.currentTime);
                    }
                }
            }, 50);

            window.addEventListener('keydown', (e) => {
                if(e.code === 'Space') {
                    e.preventDefault();
                    const vid = getStreamlitVideo();
                    if(vid) {
                        if(vid.paused) vid.play();
                        else vid.pause();
                    }
                }
            });

            function selectClip(element) {
                document.querySelectorAll('.clip').forEach(c => c.classList.remove('selected'));
                element.classList.add('selected');
            }

            const draggables = document.querySelectorAll('.draggable');
            let isDragging = false, currentElement = null, startX = 0, initialLeft = 0;

            draggables.forEach(draggable => {
                draggable.addEventListener('mousedown', function(e) {
                    isDragging = true; currentElement = this;
                    startX = e.clientX; initialLeft = parseInt(window.getComputedStyle(this).left, 10) || 0;
                    selectClip(this);
                    e.stopPropagation();
                });
            });

            const playhead = document.getElementById('playhead');
            let isScrubbing = false, playheadStartX = 0, playheadInitialLeft = 0;

            playhead.addEventListener('mousedown', function(e) {
                isScrubbing = true; playheadStartX = e.clientX;
                playheadInitialLeft = parseInt(window.getComputedStyle(this).left, 10) || 80;
                e.stopPropagation();
            });

            document.addEventListener('mousemove', function(e) {
                if (isDragging && currentElement) {
                    let newLeft = initialLeft + (e.clientX - startX);
                    if (newLeft < 80) newLeft = 80; 
                    currentElement.style.left = newLeft + 'px';
                }
                
                if (isScrubbing && playhead) {
                    let newLeft = playheadInitialLeft + (e.clientX - playheadStartX);
                    if (newLeft < 80) newLeft = 80;
                    playhead.style.left = newLeft + 'px';
                    
                    const timeInSeconds = (newLeft - 80) / PX_PER_SEC;
                    document.getElementById('timecode').innerText = formatTimecode(timeInSeconds);
                    
                    const vid = getStreamlitVideo();
                    if(vid) {
                        vid.pause(); 
                        vid.currentTime = timeInSeconds;
                    }
                }
            });

            document.addEventListener('mouseup', () => { isDragging = false; isScrubbing = false; currentElement = null; });
        </script>
    </body>
    </html>
    """
    
    components.html(timeline_html, height=370)