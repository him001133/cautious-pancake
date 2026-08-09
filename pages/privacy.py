import streamlit as st

st.set_page_config(page_title="Privacy Policy | AutoDirector AI", page_icon="🎬", layout="centered")

st.markdown("""
# Privacy Policy
**Last Updated: August 10, 2026**

### 1. Information We Collect
We collect the video files, audio transcripts, and YouTube URLs you provide in order to process and render your AI clips. We do not require account creation for the free tier.

### 2. How We Use Your Information
Your files are stored temporarily on our servers exclusively for processing. All uploaded media and rendered clips are automatically purged from our servers periodically to free up storage. 

### 3. Third-Party Services
We use Google Gemini AI to analyze your transcripts and extract highlights. We use the Cobalt API to process external video links. By using our service, you acknowledge that your transcripts and links may be processed by these third-party providers.

*If you have any questions or require data deletion, please contact us via the Support page.*
""")
