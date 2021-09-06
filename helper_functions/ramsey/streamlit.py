from ramsey.ramsey import upload, reindex
from ramsey.ramsey import data_collect
from urllib.request import urlopen
from pydub.playback import play
from pydub import AudioSegment
from pytube import Playlist
from io import BytesIO
import streamlit as st
import pyodbc as sql
import pandas as pd
import numpy as np

#################
# Default Login #
#################

try:
    default_username = 'zangorth'
    default_password = open(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\password.txt', 'r').read()
except FileNotFoundError:
    default_username = 'guest_login'
    default_password = 'ReadOnly!23'

###############
# Data Scrape #
###############
def collect():
    st.title('The Ramsey Highlights')
    st.header('New Data Collection')
    
    with st.sidebar.expander('Credentials'):
        login = st.form('Login', clear_on_submit=True)
        username = login.text_input('Username:', default_username)
        password = login.text_input('Password:', default_password, type='password')
        login.form_submit_button()
        
    options = st.sidebar.radio('', ['All Data', 'Link'])
    
    if options == 'Link':
        video_link = st.sidebar.text_input('Link:', 'https://www.youtube.com/watch?v=NpKZTVehzbE')
        audio_location = st.sidebar.text_input('Audio Path:', r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\Audio\Audio Full\Ramsey')
        transcript_location = st.sidebar.text_input('Transcript Path:', r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\Audio\Transcript\Ramsey')
        
        run = st.sidebar.button('BEGIN COLLECTION')
        
        if run:
            data_collect(video_link, username, password, audio_location, transcript_location)
    
    if options == 'All Data':
        personality = st.sidebar.selectbox('Personality', ['ramsey', 'deloney', 'coleman', 'ao', 'cruze', 'wright', 'kamel'])
        audio_location = st.sidebar.text_input('Audio Path:', f'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Audio\\Audio Full\\{personality}')
        transcript_location = st.sidebar.text_input('Transcript Path:', f'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Audio\\Transcript\\{personality}')
        
        personalities = {'ramsey': 'https://www.youtube.com/watch?v=0JUw1agDjoA&list=UU7eBNeDW1GQf2NJQ6G6gAxw&index=2',
                         'deloney': 'https://www.youtube.com/watch?v=_wWc1Tc19qA&list=UU4HiMKM8WLcNbt9ae_XNRNQ&index=2',
                         'coleman': 'https://www.youtube.com/watch?v=aKRSyxnE3C4&list=UU0tVfiyBpMOQLA3FAanPGJA&index=2',
                         'ao': 'https://www.youtube.com/watch?v=adTnzyz7deI&list=UUaW51g-nmLfq703TPZC7Gsg&index=2',
                         'cruze': 'https://www.youtube.com/watch?v=PvwDX69CsAQ&list=UUt59W0ScV709iwy2h-oiulQ&index=2',
                         'wright': 'https://www.youtube.com/watch?v=bdXVQGZtYy4&list=UU1CHQyZ5-MTJzuSCvSVw_qg&index=2',
                         'kamel': 'https://www.youtube.com/watch?v=u5ufvVsaW4M&list=UUKFrkFOwmiXMuZtQJXuG5OQ&index=2'}
        
        run = st.sidebar.button('BEGIN COLLECTION')
            
        if run:
            with st.spinner('Identifying New Videos'):
                videos = Playlist(personalities[personality]).video_urls
                videos = list(videos)
                
                connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                                     'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                                     f'UID={username};PWD={password}')
                con = sql.connect(connection_string)
                query = 'SELECT * FROM ramsey.metadata'
                collected = pd.read_sql(query, con)
                con.close()
                
                videos = list(set(videos) - set(collected['link']))
                
            if len(videos) <= 0:
                st.write('No New Videos to Upload')
                
            else:
                first = True
                iteration_meta = st.empty()
                i, pb_meta = 0, st.progress(0)
                for video in videos:
                    iteration_meta.text(f'Processing Video: {i+1}/{len(videos)}')
                    pb_meta.progress((i+1)/len(videos))
                    i += 1
                    
                    try:
                        out = data_collect(video, username, password, audio_location, transcript_location, verbose=False)
                    
                        if first:
                            meta_frame = out[0]
                            audio_frame = out[1]
                            
                            first = False
                            
                        else:
                            meta_frame = meta_frame.append(out[0], ignore_index=True, sort=False)
                            audio_frame = audio_frame.append(out[1], ignore_index=True, sort=False)
                            
                    except:
                        pass
                    
                st.write('Meta Data')
                st.dataframe(meta_frame)
                st.write('')
                st.write('Audio Code')
                st.dataframe(audio_frame)

#######################
# Supervised Training #
#######################
def train():
    personalities = ['Ramsey', 'Deloney', 'Coleman', 'AO', 'Cruze', 'Wright', 'Kamel']
    api = 'AIzaSyAftHJhz8-5UUOACb46YBLchKL78yrXpbw'
    
    st.title('The Ramsey Highlights')
    st.header('Model Training')
    
    with st.sidebar.expander('Credentials'):
        login = st.form('Login', clear_on_submit=True)
        username = login.text_input('Username:', default_username)
        password = login.text_input('Password:', default_password, type='password')
        gdrive = login.selectbox('Use Google Drive:', ['No', 'Yes'])
        login.form_submit_button()
    
    local = True if gdrive == 'No' else False
    
    st.write('')
    st.sidebar.subheader('Filters')
    channel = st.sidebar.multiselect('Channel', personalities, ['Ramsey'])
    channel = [f"'{c.lower()}'" for c in channel]
    
    left_side, right_side = st.sidebar.columns(2)
    equality = left_side.selectbox('Year (Optional)', ['=', '>', '<'])
    year = right_side.text_input('')
    
    keywords = st.sidebar.selectbox('Train Hogan?', ['No', 'Yes'])
    
    year_filter = 'YEAR(publish_date) IS NOT NULL' if year == '' else f'YEAR(publish_date) {equality} {year}'
    channel_filter = f'({", ".join(channel)})'
    key_filter = "AND keywords LIKE '%hogan%'" if keywords == 'Yes' else ''
    
    begin = st.sidebar.button('BEGIN TRAINING')
    
    if begin:
        query = f'''
        SELECT * 
        FROM ramsey.metadata
        WHERE drive IS NOT NULL
            AND channel IN {channel_filter}
            AND {year_filter} {key_filter}
        '''
        
        with st.spinner('Reading Data from Azure'):
            connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                                  'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                                  f'UID={username};PWD={password}')
            con = sql.connect(connection_string)
            collected = pd.read_sql(query, con)
            trained = pd.read_sql('SELECT * FROM ramsey.training', con)
            con.close()
            
            collected = collected.loc[collected.index.repeat(collected.seconds)]
            collected['second'] = collected.groupby(['channel', 'id']).cumcount()
            collected = collected.merge(trained[['channel', 'id', 'second', 'speaker', 'gender']], how='left')
            collected = collected.loc[(collected['speaker'].isnull()) | (collected['gender'].isnull())]
            collected = collected.sample(frac=1).reset_index(drop=True)
            
            st.session_state['panda'] = collected
            st.session_state['trained'] = pd.DataFrame(columns=trained.columns)
            st.session_state['i'] = 0
            st.session_state['sound'] = ''
            
            st.session_state['restrict_year'] = 'all' if year_filter == 'YEAR(publish_date) IS NOT NULL' else f'{equality}{year}'
            st.session_state['restrict_channel'] = 'all' if len(channel) == 7 else '|'.join(channel)
            st.session_state['restrict_key'] = 'all' if keywords == 'No' else 'hogan'
            
    
    if 'panda' in st.session_state:
        i = st.session_state['i']
        
        personality = st.session_state['panda']['channel'][i]
        sample = st.session_state['panda']['id'][i]
        second = st.session_state['panda']['second'][i]
        video_link = st.session_state['panda']['link'][i].split('v=')[-1]
        drive = st.session_state['panda']['drive'][i]
        
        if st.session_state['sound'] == '':
            with st.spinner('Reading Audio File'):
                if local:
                    sound_byte = f'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Audio\\Audio Full\\{personality}\\{sample} {personality}.mp3'
                else:
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
        link = right.button('LINK')
        
        if replay:
            play(sound)
        
        if context:
            play(lead)
            
        if link:
            st.write(f'https://youtu.be/{video_link}?t={second-3}')
        
        upload_form = st.form('upload_both', clear_on_submit=True)
        left, middle, right = upload_form.columns(3)
        
        speaker_upload = left.radio('Speaker', personalities + ['Hogan', 'Guest', 'None'])
        gender_upload = middle.radio('Gender', ['Man', 'Woman', 'None'])
        slce = f'{st.session_state["restrict_channel"]}-{st.session_state["restrict_year"]}-{st.session_state["restrict_key"]}'        
        
        send = upload_form.form_submit_button()
            
        if send:
            new = pd.DataFrame({'channel': [personality], 'id': [sample], 'second': [second],
                                'speaker': [speaker_upload], 'gender': [gender_upload],
                                'slice': [slce.replace("'", "")]})
            
            st.session_state['trained'] = st.session_state['trained'].append(new, ignore_index=True, sort=False)
            
            st.session_state['i'] = i + 1
            st.session_state['sound'] = ''
            
            st.experimental_rerun()
            
        if st.session_state['i'] > 0:
            azure = st.button('AZURE UPLOAD')
            
            if azure:
                with st.spinner('Uploading to Azure'):
                    if username == 'zangorth':
                        upload(st.session_state['trained'], 'ramsey', 'training', username, password)
                        reindex('ramsey', 'training', ['channel', 'id', 'second'], username, password)
                        st.write('Upload Complete')
                        
                    else:
                        st.write('Data from Guest Profiles will not be uploaded')