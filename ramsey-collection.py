from ramsey.streamlit import collect, train
from streamlit import caching
import streamlit as st

st.set_page_config(layout='wide')
st.sidebar.header('Navigation')
page = st.sidebar.selectbox('', ['Collect Data', 'Train Data'])


if page == 'Collect Data':
    caching.clear_cache()
    collect()

elif page == 'Train Data':
    caching.clear_cache()
    train()
