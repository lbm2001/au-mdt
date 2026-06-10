import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Prevent both OpenMP runtimes (numpy's libomp + PyTorch's libiomp5) from
# spawning thread pools that conflict inside Streamlit's threaded server on
# macOS.  These must be set before numpy / torch are first imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import streamlit as st

pg = st.navigation([
    st.Page("pages/0_Settings.py",                title="Settings",            icon="⚙️"),
    st.Page("pages/1_Policy_Explorer.py",          title="Policy Explorer",     icon="📊"),
    st.Page("pages/2_Policy_Rollout.py",           title="Policy Rollout",      icon="🎲"),
    st.Page("pages/3_Backward_Induction_Steps.py", title="Backward Induction",  icon="🔢"),
    st.Page("pages/4_Sensitivity_Analysis.py",     title="Sensitivity Analysis", icon="📐"),
])
pg.run()
