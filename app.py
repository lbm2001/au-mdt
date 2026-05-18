import streamlit as st

pg = st.navigation([
    st.Page("pages/0_Policy_Explorer.py", title="Policy Explorer", icon="📊"),
    st.Page("pages/1_Backward_Induction_Steps.py", title="Backward Induction Steps", icon="🔢"),
])
pg.run()
