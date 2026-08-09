import os
import re
import json
import subprocess
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

# Page Configuration
st.set_page_config(page_title="AutoDirector AI", page_icon="🎬", layout="wide")

# --- CUSTOM UI: HEADER & FOOTER ---
st.markdown("""
    <style>
    /* Fixed Top Header */
    .custom-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        padding: 15px 40px;
        z-index: 99999;
        border-bottom: 1px solid #2e303e;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-logo {
        font-size: 1.4rem;
        font-weight: 800;
        color: white;
        text-decoration: none;
        letter-spacing: 0.5px;
    }
    .header-links a {
        color: #a3a8b8;
        text-decoration: none;
        margin-left: 25px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: color 0.2s ease;
    }
    .header-links a:hover {
        color: #ff4b4b;
    }
    
    /* Push main content down so header doesn't overlap it */
    .block-container {
        padding-top: 90px !important;
        padding-bottom: 80px !important;
    }
    
    /* Fixed Bottom Footer */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        padding: 12px 0;
        text-align: center;
        border-top: 1px solid #2e303e;
        color: #6b7280;
        font-size: 0.85rem;
        z-index: 99999;
    }
    </style>
    
    <div class="custom-header">
        <div class="header-logo">🎬 AutoDirector AI</div>
        <div class="header-links">
            <a href="#">Dashboard</a>
            <a href="#">Pricing</a>
            <a href="#">Support</a>
        </div>
    </div>
    
    <div class="custom-footer">
        &copy; 2026 AutoDirector AI Studio. All rights reserved.
    </div>
""", unsafe_allow_html=True)
# ----------------------------------

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
PREVIEW_DIR = "previews"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

@st.cache_resource
def load_whisper_model():
    # Optimized for GitHub Codespaces / Streamlit Cloud CPU speed
    return WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4, num_workers=2)

whisper_model = load_whisper_model()

# Session State Initialization
if "current_step" not in st.session_state:
    st.session_state.current_step = "upload" # 'upload', 'dashboard', 'editor'
if "active_video" not in st.session_state:
    st.session_state.active_video = None
if "detected_clips" not in st.session_state:
    st.session_state.detected_clips = []
if "selected_clip_idx" not in st.session_state:
    st.session_state.selected_clip_idx = 0
if "framing_mode" not in st.session_state:
    st.session_state.framing_mode = "Vertical"
if "crop_x_percent" not in st.session_state:
    st.session_state.crop_x_percent = 50

# --- FFMPEG FILTER BUILDER FOR SCREEN 4 LAYOUT PRESETS ---
def build_layout_ffmpeg_filter(mode, crop_x_pct=50):
    x_factor = crop_x_pct / 100.0
    
    if mode == "Vertical":
        return f"crop=ih*(9/16):ih:n*(iw-crop_w)*{x_factor}:0,scale=1080:1920"
        
    elif mode == "Split":
        return (
            "split=2[top][bot];"
            f"[top]crop=iw/2:ih:0:0,scale=1080:960[t_scaled];"
            f"[bot]crop=iw/2:ih:iw/2:0,scale=1080:960[b_scaled];"
            "[t_scaled][b_scaled]vstack=inputs=2"
        )
        
    elif mode == "Trio":
        return (
            "split=3[p1][p2][p3];"
            "[p1]crop=iw/3:ih:0:0,scale=1080:640[v1];"
            "[p2]crop=iw/3:ih:iw/3:0,scale=1080:640[v2];"
            "[p3]crop=iw/3:ih:(iw/3)*2:0,scale=1080:640[v3];"
            "[v1][v2][v3]vstack=inputs=3"
        )
        
    elif mode == "Spotlight":
        return "split[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:10[bg_blur];[fg]scale=1080:-1[fg_scale];[bg_blur][fg_scale]overlay=0:(H-h)/2"
        
    elif mode == "Centered":
        return "crop=ih*(9/16):ih:(iw-crop_w)/2:0,scale=1080:1920"
        
    elif mode == "Horizontal":
        return "split[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:10[b_blur];[fg]scale=1080:-1[f_scale];[b_blur][f_scale]overlay=0:(H-h)/2"
        
    return "crop=ih*(9/16):ih,scale=1080:1920"

def generate_frame_preview(video_path, timestamp_s, filter_str):
    preview_path = os.path.join(PREVIEW_DIR, "preview_frame.jpg")
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp_s), "-i", video_path,
        "-vf", filter_str, "-vframes", "1", "-q:v", "2", preview_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return preview_path if os.path.exists(preview_path) else None

# ==========================================
# SCREEN 1: FULL LANDING PAGE & UPLOAD
# ==========================================
if st.session_state.current_step == "upload":
    
    # 1. Giant Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 4rem 0 2rem 0;'>
            <h1 style='font-size: 4rem; font-weight: 900; margin-bottom: 15px; line-height: 1.2;'>
                Turn Podcasts into <br><span style='color: #ff4b4b;'>Viral 9:16 Shorts</span>
            </h1>
            <p style='font-size: 1.3rem; color: #a3a8b8; max-width: 800px; margin: 0 auto;'>
                Stop scrubbing through hours of footage. Our AI story engine finds the highest-retention hooks and auto-frames the action instantly.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Main Call-to-Action (Upload Area)
    st.markdown("<br>", unsafe_allow_html=True)
    col_space1, col_main, col_space2 = st.columns([1, 2, 1])
    with col_main:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center; margin-bottom: 0;'>Start Clipping for Free</h3>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload a new video", type=["mp4", "mov", "webm", "mkv"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                save_path = os.path.join(INPUT_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.active_video = uploaded_file.name
                st.session_state.current_step = "dashboard"
                st.rerun()
                
        existing_files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".mp4", ".mov", ".mkv", ".webm"))]
        if existing_files:
            st.markdown("<div style='text-align: center; margin: 15px 0; color: #6b7280; font-size: 0.9rem;'>— or open a recent project —</div>", unsafe_allow_html=True)
            with st.container(border=True):
                selected_existing = st.selectbox("Your Videos", existing_files, label_visibility="collapsed")
                if st.button("🚀 Process Selected Video", type="primary", use_container_width=True):
                    st.session_state.active_video = selected_existing
                    st.session_state.current_step = "dashboard"
                    st.rerun()

    # 3. How It Works
    st.markdown("<br><br><br><h2 style='text-align: center;'>How It Works</h2><hr style='border-color: #2e303e;'>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("### 1️⃣ Upload")
        st.write("Drop in your long-form MP4. We support podcasts, interviews, and gaming VODs up to 800MB directly in the browser.")
    with s2:
        st.markdown("### 2️⃣ AI Analysis")
        st.write("Our Gemini-powered engine reads the transcript to find emotional spikes, jokes, and high-retention viral hooks.")
    with s3:
        st.markdown("### 3️⃣ Auto-Frame")
        st.write("Choose your layout. We automatically crop the landscape video to a perfect 9:16 and export a watermark-free MP4.")

    # 4. Pricing Table
    st.markdown("<br><br><br><h2 style='text-align: center;'>Simple Pricing</h2><hr style='border-color: #2e303e;'>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        with st.container(border=True):
            st.markdown("### Hobby")
            st.markdown("## $0<span style='font-size: 1rem; color: gray;'>/mo</span>", unsafe_allow_html=True)
            st.markdown("- 5 uploads per month\n- 720p Export\n- Standard AI Engine")
            st.button("Current Plan", disabled=True, use_container_width=True)
    with p2:
        with st.container(border=True):
            st.markdown("### Creator 🚀")
            st.markdown("## $15<span style='font-size: 1rem; color: gray;'>/mo</span>", unsafe_allow_html=True)
            st.markdown("- 50 uploads per month\n- 1080p Export\n- Face Tracking Crop")
            st.button("Upgrade to Creator", type="primary", use_container_width=True)
    with p3:
        with st.container(border=True):
            st.markdown("### Studio")
            st.markdown("## $49<span style='font-size: 1rem; color: gray;'>/mo</span>", unsafe_allow_html=True)
            st.markdown("- Unlimited uploads\n- 4K ProRes Export\n- Custom Watermarks")
            st.button("Contact Sales", use_container_width=True)

    # 5. FAQ Section
    st.markdown("<br><br><br><h2 style='text-align: center;'>Frequently Asked Questions</h2><hr style='border-color: #2e303e;'>", unsafe_allow_html=True)
    with st.expander("Does this work for non-English videos?"):
        st.write("Yes! Our built-in Whisper AI model automatically translates and transcribes over 90 languages with extreme accuracy.")
    with st.expander("How long does processing take?"):
        st.write("Typically, a 10-minute video takes about 2 to 3 minutes to transcribe and clip on our cloud servers.")
    with st.expander("Are there watermarks on the free tier?"):
        st.write("No. We believe in providing value first. All exports from AutoDirector AI are 100% watermark-free, even on the Hobby plan.")
    
    # Extra padding at the bottom so the sticky footer doesn't cover content
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# ==========================================
# SCREEN 2 & 3: AI CLIP DASHBOARD
# ==========================================
elif st.session_state.current_step == "dashboard":
    top_col1, top_col2 = st.columns([1, 8])
    with top_col1:
        if st.button("← Back"):
            st.session_state.current_step = "upload"
            st.rerun()
    with top_col2:
        st.subheader(f"Project: {st.session_state.active_video}")

    video_path = os.path.join(INPUT_DIR, st.session_state.active_video)
    
    col_left, col_right = st.columns([3, 2], gap="medium")
    
    with col_left:
        st.video(video_path)
        
    with col_right:
        st.markdown("### ✦ Find a moment")
        prompt_input = st.text_input("Describe what you're looking for or hit Auto-Scan", placeholder='"the funniest part", "when they get emotional"...')
        
        if st.button("⚡ Generate AI Clips", type="primary", use_container_width=True):
            with st.spinner("Transcribing and searching story arcs..."):
                lang_code = "en"
                segments, _ = whisper_model.transcribe(video_path, word_timestamps=True)
                
                transcript_lines = []
                all_words = []
                for s in segments:
                    for w in s.words:
                        all_words.append({"word": w.word, "start": w.start, "end": w.end})
                    transcript_lines.append(f"[{s.start:.1f}s - {s.end:.1f}s] {s.text.strip()}")
                
                active_key = os.environ.get("GEMINI_API_KEY")
                
                if active_key:
                    try:
                        client = genai.Client(api_key=active_key)
                        prompt = f"""
                        Read this transcript and find the 4 most engaging short clips.
                        Return ONLY a valid JSON array:
                        [
                            {{"title": "Why surfing is the ultimate metaphor", "start": 12.0, "end": 36.0}}
                        ]
                        Transcript:
                        {"\n".join(transcript_lines)}
                        """
                        res = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
                        clean_json = re.sub(r'```json\n|\n```|```', '', res.text).strip()
                        st.session_state.detected_clips = json.loads(clean_json)
                    except Exception as e:
                        st.error(f"API Error: {e}")
                
                if not st.session_state.detected_clips and all_words:
                    total_dur = all_words[-1]["end"]
                    st.session_state.detected_clips = [
                        {"title": f"Viral Hook Segment #{i+1}", "start": float(i*30), "end": float(i*30 + 25)}
                        for i in range(min(4, int(total_dur // 30)))
                    ]
        
        if st.session_state.detected_clips:
            st.markdown(f"#### Clips ({len(st.session_state.detected_clips)})")
            
            for idx, clip in enumerate(st.session_state.detected_clips):
                with st.container(border=True):
                    st.markdown(f"**{idx+1}. {clip['title']}**")
                    st.caption(f"⏱️ {int(clip['start'])}s → {int(clip['end'])}s ({int(clip['end'] - clip['start'])}s)")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Clip Horizontal 📺", key=f"horiz_{idx}"):
                            st.session_state.selected_clip_idx = idx
                            st.session_state.framing_mode = "Horizontal"
                            st.session_state.current_step = "editor"
                            st.rerun()
                    with c2:
                        if st.button("Clip Vertical 📱", key=f"vert_{idx}", type="primary"):
                            st.session_state.selected_clip_idx = idx
                            st.session_state.framing_mode = "Vertical"
                            st.session_state.current_step = "editor"
                            st.rerun()

# ==========================================
# SCREEN 4: POSITION THE CROP (FRAMING STUDIO)
# ==========================================
elif st.session_state.current_step == "editor":
    top_col1, top_col2 = st.columns([1, 8])
    with top_col1:
        if st.button("← Back"):
            st.session_state.current_step = "dashboard"
            st.rerun()
    with top_col2:
        st.subheader("Position the Crop")

    clip = st.session_state.detected_clips[st.session_state.selected_clip_idx]
    video_path = os.path.join(INPUT_DIR, st.session_state.active_video)
    
    st.markdown("### Framing Strategy")
    preset_cols = st.columns(6)
    modes = ["Vertical", "Split", "Trio", "Spotlight", "Centered", "Horizontal"]
    
    for i, m in enumerate(modes):
        with preset_cols[i]:
            btn_type = "primary" if st.session_state.framing_mode == m else "secondary"
            if st.button(m, key=f"mode_{m}", type=btn_type, use_container_width=True):
                st.session_state.framing_mode = m
                st.rerun()

    col_edit, col_prev = st.columns([3, 2], gap="large")
    
    with col_edit:
        st.markdown("#### Adjust Camera Position")
        st.session_state.crop_x_percent = st.slider(
            "Horizontal Position (X-Axis)",
            min_value=0, max_value=100, value=st.session_state.crop_x_percent,
            help="Drag to move the 9:16 framing box across the source video."
        )
        
        filter_str = build_layout_ffmpeg_filter(st.session_state.framing_mode, st.session_state.crop_x_percent)
        
        st.markdown("---")
        if st.button("⚡ Render Final Short", type="primary", use_container_width=True):
            out_name = f"clipzi_{st.session_state.framing_mode}_{st.session_state.selected_clip_idx+1}.mp4"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            with st.spinner("Rendering short with selected layout..."):
                cmd = [
                    "ffmpeg", "-y", "-ss", str(clip['start']), "-i", video_path,
                    "-t", str(clip['end'] - clip['start']),
                    "-vf", filter_str, "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", out_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.success(f"Rendered! Saved to {out_path}")
                st.video(out_path)

    with col_prev:
        st.markdown("#### Live Preview (9:16)")
        filter_str = build_layout_ffmpeg_filter(st.session_state.framing_mode, st.session_state.crop_x_percent)
        mid_point = (clip['start'] + clip['end']) / 2.0
        
        preview_img = generate_frame_preview(video_path, mid_point, filter_str)
        if preview_img:
            st.image(preview_img, caption=f"Mode: {st.session_state.framing_mode}", use_container_width=True)