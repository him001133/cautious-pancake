import streamlit as st

st.set_page_config(page_title="Support | AutoDirector AI", page_icon="🎬", layout="wide")

# --- CUSTOM UI: HEADER & FOOTER ---
st.markdown("""
    <style>
    .custom-header { position: fixed; top: 0; left: 0; width: 100%; background-color: #0e1117; padding: 15px 40px; z-index: 99999; border-bottom: 1px solid #2e303e; display: flex; justify-content: space-between; align-items: center; }
    .header-logo { font-size: 1.4rem; font-weight: 800; color: white; text-decoration: none; }
    .header-links a { color: #a3a8b8; text-decoration: none; margin-left: 25px; font-weight: 600; font-size: 0.95rem; transition: color 0.2s ease; }
    .header-links a:hover { color: #ff4b4b; }
    .block-container { padding-top: 90px !important; padding-bottom: 80px !important; }
    .custom-footer { position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0e1117; padding: 12px 0; text-align: center; border-top: 1px solid #2e303e; color: #6b7280; font-size: 0.85rem; z-index: 99999; }
    </style>
    <div class="custom-header">
        <div class="header-logo">🎬 AutoDirector AI</div>
        <div class="header-links">
            <a href="/" target="_top">Dashboard</a>
            <a href="/pricing" target="_top">Pricing</a>
            <a href="/support" target="_top">Support</a>
        </div>
    </div>
    <div class="custom-footer">&copy; 2026 AutoDirector AI Studio. All rights reserved.</div>
""", unsafe_allow_html=True)
# ----------------------------------

st.markdown("<br><h1 style='text-align: center;'>How can we help?</h1><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.form("support_form"):
        st.text_input("Email Address")
        st.selectbox("What do you need help with?", ["Billing", "Technical Issue", "Feature Request", "Other"])
        st.text_area("Describe your issue")
        submit = st.form_submit_button("Send Message", type="primary", use_container_width=True)
        
        if submit:
            st.success("Thanks! Our team will get back to you within 24 hours.")
