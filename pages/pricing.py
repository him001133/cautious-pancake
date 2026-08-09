import streamlit as st

st.set_page_config(page_title="Pricing | AutoDirector AI", page_icon="🎬", layout="wide")

st.markdown("<br><br><br><h1 style='text-align: center;'>Simple, Transparent Pricing</h1><br>", unsafe_allow_html=True)

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
