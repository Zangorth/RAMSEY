from urllib.request import urlopen
from helpers.ramsey import upload
from pydub.playback import play
from pydub import AudioSegment
from time import sleep
from io import BytesIO
import streamlit as st
import pyodbc as sql
import pandas as pd
import numpy as np

personalities = ['Ramsey', 'Deloney', 'Coleman', 'AO', 'Cruze', 'Wright', 'Kamel']
api = 'AIzaSyAftHJhz8-5UUOACb46YBLchKL78yrXpbw'

st.title('The Ramsey Highlights')
st.header('Model Training')

with st.sidebar.expander('Credentials'):
    login = st.form('Login', clear_on_submit=True)
    username = login.text_input('Username:', 'guest_login')
    password = login.text_input('Password:', 'ReadOnly!23', type='password')
    submit = login.form_submit_button()
    
st.write('')
st.sidebar.subheader('Models to Train')
models = st.sidebar.multiselect('Select All That Apply', ['Speaker', 'Gender'], ['Speaker', 'Gender'])
models = [m.lower() for m in models]

st.write('')
st.sidebar.subheader('Filters')    
channel = st.sidebar.multiselect('Channel', personalities, personalities)

channel = [f"'{c.lower()}'" for c in channel]

left_side, right_side = st.sidebar.columns(2)
equality = left_side.selectbox('Year (Optional)', ['=', '>', '<'])
year = right_side.text_input('')

year_filter = 'YEAR(publish_date) IS NOT NULL' if year == '' else f'YEAR(publish_date) {equality} {year}'
channel_filter = f'({", ".join(channel)})'

begin = st.sidebar.button('BEGIN TRAINING')

if begin:
    query = f'''
    SELECT * 
    FROM ramsey.metadata
    WHERE seconds <= 900
        AND channel IN {channel_filter}
        AND {year_filter}
    '''
    
    with st.spinner('Reading Data from Azure'):
        connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                              'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                              f'UID={username};PWD={password}')
        con = sql.connect(connection_string)
        collected = pd.read_sql(query, con)
        checked = pd.read_sql('SELECT * FROM ramsey.training', con)
        con.close()
        
        
        
        collected = collected.sample(100000, replace=True, weights=collected['seconds']).reset_index(drop=True)
        
        st.session_state['panda'] = collected
        st.session_state['i'] = 0
        st.session_state['seed'] = np.random.randint(1, 1000000)
        st.session_state['sound'] = ''
        
        
if 'panda' in st.session_state:
    i = st.session_state['i']
    st.session_state['old_eye'] = i
    rng = np.random.default_rng(st.session_state['seed']+i)
    
    personality = st.session_state['panda']['channel'][i]
    sample = st.session_state['panda']['id'][i]
    second = rng.integers(0, st.session_state['panda']['seconds'][i]-1)
    link = st.session_state['panda']['link'][i]
    drive = st.session_state['panda']['drive'][i]
    
    if st.session_state['sound'] == '':
        sound_byte = BytesIO(urlopen(f'https://www.googleapis.com/drive/v3/files/{drive}?key={api}&alt=media').read())
        st.session_state['sound'] = AudioSegment.from_file(sound_byte)
    
        sound = st.session_state['sound']
        lead = sound[second*1000-3000: second*1000+3000]
        sound = sound[second*1000:second*1000+1000]
        
        play(sound)
        
    else:
        sound = st.session_state['sound']
        lead = sound[second*1000-3000: second*1000+3000]
        sound = sound[second*1000:second*1000+1000]
    
    st.subheader(f'Iteration {i}: {personality} {sample} - Second {second}')
    
    left, middle, right = st.columns(3)
    
    replay = left.button('REPLAY')
    context = middle.button('CONTEXT')
    
    if replay:
        play(sound)
    
    if context:
        play(lead)
    
    if 'speaker' in models and 'gender' in models:
        upload_form = st.form('upload_both', clear_on_submit=True)
        left, right = upload_form.columns(2)
        speaker_upload = left.radio('Speaker', personalities + ['Hogan', 'Guest', 'None'])
        gender_upload = right.radio('Gender', ['Man', 'Woman', 'None'])
        send = upload_form.form_submit_button()
        
    elif 'speaker' in models:
        upload_form = st.form('upload_speaker', clear_on_submit=True)
        speaker_upload = upload_form.radio('Speaker', personalities + ['Hogan', 'Guest', 'None'])
        gender_upload = None
        send = upload_form.form_submit_button()
        
    elif 'gender' in models:
        upload_form = st.form('upload_gender', clear_on_submit=True)
        speaker_upload = None
        gender_upload = upload_form.radio('Gender', ['Man', 'Woman', 'None'])
        send = upload_form.form_submit_button()
        
    if send:
        loadup = pd.DataFrame({'channel': [personality], 'id': [sample], 'second': [second],
                               'speaker': speaker_upload, 'gender': gender_upload})
        
        if username == 'zangorth':
            with st.spinner('Uploading Data to Azure'):
                upload(loadup, 'training', username, password)
                
        else:
            with st.spinner('Guest Users are not Allowed to Upload Data'):
                sleep(3)
        
        st.session_state['i'] = i + 1
        st.session_state['sound'] = ''
        st.write(f'Speaker: {speaker_upload} | Gender: {gender_upload}')
        
        st.experimental_rerun()