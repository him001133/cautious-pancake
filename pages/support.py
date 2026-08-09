import streamlit as st

st.set_page_config(page_title="Support | AutoDirector AI", page_icon="🎬", layout="wide")

# --- NATIVE UI: HEADER & FOOTER ---
st.markdown("""
    <style>
    /* Hide default header */
    [data-testid="stHeader"] { display: none !important; }
    .block-container { padding-top: 1rem !important; padding-bottom: 80px !important; }
    
    /* Sticky Footer */
    .custom-footer { position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0e1117; padding: 12px 0; text-align: center; border-top: 1px solid #2e303e; color: #6b7280; font-size: 0.85rem; z-index: 99999; }
    </style>
    <div class="custom-footer">&copy; 2026 AutoDirector AI Studio. All rights reserved.</div>
""", unsafe_allow_html=True)

# Native Streamlit Navigation
nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([6, 1, 1, 1], vertical_alignment="center")
with nav_c1: st.markdown("<h3 style='margin: 0;'>🎬 AutoDirector AI</h3>", unsafe_allow_html=True)
with nav_c2: st.page_link("app.py", label="Dashboard", icon="🏠")
with nav_c3: st.page_link("pages/pricing.py", label="Pricing", icon="💳")
with nav_c4: st.page_link("pages/support.py", label="Support", icon="🎧")
st.markdown("---")
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
