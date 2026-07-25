import subprocess
import os
from faster_whisper import WhisperModel

def transcribe_video(video_path):
    print("🎙️ Transcribing audio and extracting timestamps...")
    # Using CPU and int8 quantization for optimized cloud container performance
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(video_path, word_timestamps=True)
    
    transcript_data = []
    for segment in segments:
        for word in segment.words:
            transcript_data.append({
                "word": word.word,
                "start": word.start,
                "end": word.end
            })
    return transcript_data

def create_vertical_clip(input_path, output_path, start_time, end_time):
    print(f"✂️ Slicing from {start_time}s to {end_time}s and cropping to 9:16...")
    
    # FFmpeg filter: crops center 9:16 vertical window from a 16:9 landscape canvas
    crop_filter = "crop=ih*(9/16):ih"
    
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"🚀 Success! Vertical clip saved to: {output_path}")

if __name__ == "__main__":
    # 1. Point directly to your uploaded podcast file
    sample_video = "podcast 1.mp4"
    
    # Safety check to ensure the file exists and is completely uploaded
    if not os.path.exists(sample_video):
        print(f"❌ Error: Could not find '{sample_video}'. Make sure your file upload is 100% complete!")
        exit()

    # 2. Extract word-level timestamps from your podcast
    words = transcribe_video(sample_video)
    print(f"✅ Transcribed {len(words)} words.")

    # 3. Define your highlight window in seconds
    # Current setting: Slices a 30-second clip from 1 minute in (60.0s) to 1 min 30 seconds (90.0s)
    # Change these numbers to target the exact timestamps of the hook you want!
    clip_start = 60.0
    clip_end = 90.0

    # 4. Process and export the vertical short
    create_vertical_clip(sample_video, "viral_short_output.mp4", clip_start, clip_end)