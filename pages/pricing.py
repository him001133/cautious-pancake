import streamlit as st

st.set_page_config(page_title="Pricing | AutoDirector AI", page_icon="🎬", layout="wide")

# --- CUSTOM UI: HEADER & FOOTER ---
st.markdown("""
    <style>
    /* Hide the default Streamlit header so it doesn't overlap our custom one */
    [data-testid="stHeader"] { display: none !important; }
    
    .custom-header { position: fixed; top: 0; left: 0; width: 100%; background-color: #0e1117; padding: 15px 40px; z-index: 99999; border-bottom: 1px solid #2e303e; display: flex; justify-content: space-between; align-items: center; }
    .header-logo { font-size: 1.4rem; font-weight: 800; color: white; text-decoration: none; }
    .header-logo:hover { color: #ff4b4b; }
    .header-links a { color: #a3a8b8; text-decoration: none; margin-left: 25px; font-weight: 600; font-size: 0.95rem; transition: color 0.2s ease; }
    .header-links a:hover { color: #ff4b4b; }
    .block-container { padding-top: 90px !important; padding-bottom: 80px !important; }
    .custom-footer { position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0e1117; padding: 12px 0; text-align: center; border-top: 1px solid #2e303e; color: #6b7280; font-size: 0.85rem; z-index: 99999; }
    </style>
    
    <div class="custom-header">
        <a href="/" target="_top" class="header-logo">🎬 AutoDirector AI</a>
        <div class="header-links">
            <a href="/" target="_top">Dashboard</a>
            <a href="/pricing" target="_top">Pricing</a>
            <a href="/support" target="_top">Support</a>
        </div>
    </div>
    <div class="custom-footer">&copy; 2026 AutoDirector AI Studio. All rights reserved.</div>
""", unsafe_allow_html=True)
# ----------------------------------

st.markdown("<br><h1 style='text-align: center;'>Simple, Transparent Pricing</h1><br>", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
with p1:
    with st.container(border=True):
        st.markdown("### Hobby\n## $0<span style='font-size: 1rem; color: gray;'>/mo</span>\n- 5 uploads per month\n- 720p Export", unsafe_allow_html=True)
        st.button("Current Plan", disabled=True, use_container_width=True)
with p2:
    with st.container(border=True):
        st.markdown("### Creator 🚀\n## $15<span style='font-size: 1rem; color: gray;'>/mo</span>\n- 50 uploads per month\n- 1080p Export", unsafe_allow_html=True)
        st.button("Upgrade to Creator", type="primary", use_container_width=True)
with p3:
    with st.container(border=True):
        st.markdown("### Studio\n## $49<span style='font-size: 1rem; color: gray;'>/mo</span>\n- Unlimited uploads\n- 4K ProRes Export", unsafe_allow_html=True)
        st.button("Contact Sales", use_container_width=True)
