from pydub.playback import play
from pydub import AudioSegment
from ramsey.ramsey import upload
import streamlit as st
import pyodbc as sql
import pandas as pd

personalities = ['Ramsey', 'Deloney', 'Coleman', 'AO', 'Cruze', 'Wright', 'Kamel']
personality_choices = personalities + ['Hogan', 'Guest', 'None']
api = 'AIzaSyAftHJhz8-5UUOACb46YBLchKL78yrXpbw'

st.title('The Ramsey Highlights')
st.header('Model Training')

st.write('')
st.sidebar.subheader('Filters')
channel = st.sidebar.multiselect('Channel', personalities, ['Ramsey'])
channel = [f"'{c.lower()}'" for c in channel]

left_side, right_side = st.sidebar.columns(2)
equality = left_side.selectbox('Year (Optional)', ['=', '>', '<'])
year = right_side.text_input('')

keywords = st.sidebar.selectbox('Train Hogan?', ['No', 'Yes'])

year_filter = 'YEAR(audio.publish_date) IS NOT NULL' if year == '' else f'YEAR(audio.publish_date) {equality} {year}'
channel_filter = f'({", ".join(channel)})'
key_filter = "AND metadata.keywords LIKE '%hogan%'" if keywords == 'Yes' else ''

begin = st.sidebar.button('BEGIN TRAINING')

st.session_state['complete'] = False if 'complete' not in st.session_state else st.session_state['complete']

if begin:
    with st.spinner('Reading Data from SQL'):
        connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                             'Server=ZANGORTH;' + 
                             'DATABASE=HomeBase;' +
                             'Trusted_Connection=yes;')
        con = sql.connect(connection_string)
        query = open(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\Queries\Training.txt').read()
        panda = pd.read_sql(query.format(channel_filter, year_filter, key_filter), con)
        trained = pd.read_sql('SELECT * FROM ramsey.train', con)
        con.close()
        
        panda.columns = ['channel', 'publish_date', 'random_id', 'second', 'speaker_pred', 'gender_pred', 'link']
        panda['speaker_pred'] = panda['speaker_pred'].fillna('missing')
        panda['gender_pred'] = panda['gender_pred'].fillna('missing')
        
        panda = panda.merge(trained, how='left', on=['channel', 'publish_date', 'random_id', 'second'])
        panda = panda.loc[(panda['speaker'].isnull()) | (panda['gender'].isnull())]
        panda = panda.sample(frac=1).reset_index(drop=True)
        
        st.session_state['panda'] = panda
        st.session_state['trained'] = pd.DataFrame(columns=trained.columns)
        st.session_state['i'] = 0
        st.session_state['sound'] = ''
        
        st.session_state['restrict_year'] = 'all' if year_filter == 'YEAR(publish_date) IS NOT NULL' else f'{equality}{year}'
        st.session_state['restrict_channel'] = 'all' if len(channel) == 7 else '|'.join(channel)
        st.session_state['restrict_key'] = 'all' if keywords == 'No' else 'hogan'
        st.session_state['complete'] = False
        

if 'panda' in st.session_state and not st.session_state['complete']:
    i = st.session_state['i']
    
    personality = st.session_state['panda']['channel'][i]
    publish_date = st.session_state['panda']['publish_date'][i]
    sample = st.session_state['panda']['random_id'][i]
    second = st.session_state['panda']['second'][i]
    video_link = st.session_state['panda']['link'][i].split('v=')[-1]
    
    
    if st.session_state['sound'] == '':
        with st.spinner('Reading Audio File'):
            sound_byte = f'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Audio\\Audio Full\\{personality}\\{personality} {publish_date} {sample}.mp3'
                
            st.session_state['sound'] = AudioSegment.from_file(sound_byte)
        
            sound = st.session_state['sound']
            lead = sound[second*1000-3000: second*1000+3000]
            sound = sound[second*1000:second*1000+1000]
        
        play(sound)
        
    else:
        sound = st.session_state['sound']
        lead = sound[second*1000-3000: second*1000+3000]
        sound = sound[second*1000:second*1000+1000]
    
    st.subheader(f'Iteration {i}: {personality} {publish_date} - Second {second}')
    
    left, middle, right = st.columns(3)
    
    replay = left.button('REPLAY')
    context = middle.button('CONTEXT')
    link = right.button('LINK')
    
    if replay:
        play(sound)
    
    if context:
        play(lead)
        
    if link:
        st.write(f'https://youtu.be/{video_link}?t={second-3}')
    
    upload_form = st.form('upload', clear_on_submit=True)
    left, middle, right = upload_form.columns(3)
    
    default_speaker = 0 if st.session_state['panda']['speaker_pred'][i] == 'missing' else personality_choices.index(st.session_state['panda']['speaker_pred'][i])
    default_gender = 0 if st.session_state['panda']['gender_pred'][i] == 'missing' else ['Man', 'Woman', 'None'].index(st.session_state['panda']['gender_pred'][i])
    
    speaker_upload = left.radio('Speaker', personality_choices, index=default_speaker)
    gender_upload = middle.radio('Gender', ['Man', 'Woman', 'None'], index=default_gender)
    slce = f'{st.session_state["restrict_channel"]}-{st.session_state["restrict_year"]}-{st.session_state["restrict_key"]}'        
    
    send = upload_form.form_submit_button()
        
    if send:
        new = pd.DataFrame({'channel': [personality], 'publish_date': [publish_date], 
                            'random_id': [sample], 'second': [second],
                            'speaker': [speaker_upload], 'gender': [gender_upload],
                            'slice': [slce.replace("'", "")]})
        
        st.session_state['trained'] = st.session_state['trained'].append(new, ignore_index=True, sort=False)
        
        st.session_state['i'] = i + 1
        st.session_state['sound'] = ''
        
        st.experimental_rerun()
        
    if st.session_state['i'] > 0:
        homebase = st.button('SQL UPLOAD')
        
        if homebase:
            upload(st.session_state['trained'], 'ramsey', 'train')
            st.session_state['complete'] = True
            st.experimental_rerun()
            
if st.session_state['complete']:
    st.subheader('Data Successfully Uploaded')