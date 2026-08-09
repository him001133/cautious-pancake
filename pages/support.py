import streamlit as st

st.set_page_config(page_title="Support | AutoDirector AI", page_icon="🎬", layout="wide")

st.markdown("<br><br><br><h1 style='text-align: center;'>How can we help?</h1><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.form("support_form"):
        st.text_input("Email Address")
        st.selectbox("What do you need help with?", ["Billing", "Technical Issue", "Feature Request", "Other"])
        st.text_area("Describe your issue")
        submit = st.form_submit_button("Send Message", type="primary", use_container_width=True)
        
        if submit:
            st.success("Thanks! Our team will get back to you within 24 hours.")
