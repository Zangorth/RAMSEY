from urllib.request import urlopen
from pydub.playback import play
from pydub import AudioSegment
from io import BytesIO
import streamlit as st
import pyodbc as sql
import pandas as pd

personalities = ['Ramsey', 'Deloney', 'Coleman', 'AO', 'Cruze', 'Wright', 'Kamel']
api = 'AIzaSyAftHJhz8-5UUOACb46YBLchKL78yrXpbw'

st.title('The Ramsey Highlights')
st.header('Model Training')

with st.sidebar.expander('Credentials'):
    login = st.form('Login', clear_on_submit=True)
    username = login.text_input('Username:', 'guest_login')
    password = login.text_input('Password:', 'ReadOnly!23')
    submit = login.form_submit_button()
    
st.write('')
st.sidebar.subheader('Models to Train')
models = st.sidebar.multiselect('Select All That Apply', ['Speaker', 'Gender'])
models = [m.lower() for m in models]

st.write('')
st.sidebar.subheader('Filters')    
channel = st.sidebar.multiselect('Channel', personalities, personalities)

channel = [f"'{c.lower()}'" for c in channel]

left_side, right_side = st.sidebar.columns(2)
equality = left_side.selectbox('Year (Optional)', ['=', '>', '<'])
year = right_side.text_input('')

year_filter = 'YEAR(publish_date) IS NOT NULL' if year == '' else f'AND YEAR(publish_date) {equality} {year}'
channel_filter = f'({", ".join(channel)})'

begin = st.sidebar.button('BEGIN TRAINING')

if begin:
    query = f'''
    SELECT audio.channel, audio.id, [second],
        YEAR(publish_date) AS 'year', link, drive
    FROM
        (SELECT channel, id, MAX(second) AS [second]
        FROM ramsey.audio
        WHERE audio.channel IN {channel_filter}
        GROUP BY channel, id) audio
    LEFT JOIN ramsey.metadata
        ON metadata.channel = audio.channel
        AND metadata.id = audio.id
    WHERE {year_filter}
    '''
    
    with st.spinner('Reading Data from Azure... This could take several minutes...'):
        connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                              'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                              f'UID={username};PWD={password}')
        con = sql.connect(connection_string)
        collected = pd.read_sql(query, con)
        checked = pd.read_sql('SELECT * FROM ramsey.training', con)
        con.close()
        
        collected = collected.sample(frac=1).reset_index(drop=True)
        
        st.session_state['panda'] = collected
        st.session_state['i'] = 0
        
        
if 'panda' in st.session_state:
    i = st.session_state['i']
    personality = st.session_state['panda']['channel'][i]
    sample = st.session_state['panda']['id'][i]
    second = st.session_state['panda']['second'][i]
    link = st.session_state['panda']['link'][i]
    drive = st.session_state['panda']['drive'][i]
    
    sound_byte = BytesIO(urlopen(f'https://www.googleapis.com/drive/v3/files/{drive}?key={api}&alt=media').read())
    sound = AudioSegment.from_file(sound_byte)
    
    lead = sound[second*1000-3000: second*1000+3000]
    sound = sound[second*1000:second*1000+1000]
    
    play(sound)
    
    context = st.button('CONTEXT')
    
    if context:
        play(lead)
    
    if 'speaker' in models and 'gender' in models:
        upload_form = st.form('upload_both', clear_on_submit=True)
        left, right = upload_form.columns(2)
        left.radio('Speaker', personalities + ['Hogan', 'None'])
        right.radio('Gender', ['Man', 'Woman', 'None'])
        send = upload_form.form_submit_button()
        
        st.session_state['i'] += 1
        
    elif 'speaker' in models:
        upload_form = st.form('upload_speaker', clear_on_submit=True)
        upload_form.radio('Speaker', personalities + ['Hogan', 'None'])
        send = upload_form.form_submit_button()
        
        st.session_state['i'] += 1
        
    elif 'gender' in models:
        upload_form = st.form('upload_gender', clear_on_submit=True)
        upload_form.radio('Gender', personalities + ['Hogan', 'None'])
        send = upload_form.form_submit_button()
        
        st.session_state['i'] += 1
    
    
    
    
    
        
    
        
        # checked['identified_anywhere'] = 1
        
        # collected = collected.merge(checked, on=['channel', 'id', 'second'], how='left')
        # collected = collected.loc[collected['identified_anywhere'].isnull()]
        
        # cols = ['channel', 'id', 'second'] + models
        
        # checked = checked[models].dropna()
        # checked['identified_model'] = 1
        
        # collected = collected.merge(checked, on=['channel', 'id', 'second'], how='left')

    
    # i, stop = -1, False
    # while not stop:
    #     i += 1
        
    #     personality = collected['channel'][i]
    #     sample = collected['id'][i]
    #     second = collected['second'][i]
    #     link = collected['link'][i]
        
    #     sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{personality}\\{personality}\\{sample} {personality}.mp3')
    #     lead = sound[second*1000-3000: second*1000+3000]
    #     sound = sound[second*1000:second*1000+1000]

    #     left, middle, right = st.columns(3)
        
    #     ramsey = left.button('Ramsey')
    #     ao = left.button('AO')
    #     kamel = left.button('Kamel')
    #     deloney = middle.button('Deloney')
    #     cruze = middle.button('Cruze')
    #     lead = left.button('CONTEXT')
    #     coleman = right.button('Coleman')
    #     wright = right.button('Wright')
    #     close = right.button('Stop Training')
        
    #     while not any([ramsey, ao, kamel, deloney, cruze, lead, coleman, wright, close]):
    #         pass


































        



