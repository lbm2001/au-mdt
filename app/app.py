import streamlit as st

pg = st.navigation([
    st.Page("pages/0_Settings.py",                title="Settings",            icon="⚙️"),
    st.Page("pages/1_Policy_Explorer.py",          title="Policy Explorer",     icon="📊"),
    st.Page("pages/2_Policy_Rollout.py",           title="Policy Rollout",      icon="🎲"),
    st.Page("pages/3_Backward_Induction_Steps.py", title="Backward Induction",  icon="🔢"),
])
pg.run()
