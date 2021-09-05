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

def collect():
    st.title('The Ramsey Highlights')
    st.header('New Data Collection')
    
    with st.sidebar.expander('Credentials'):
        login = st.form('Login', clear_on_submit=True)
        username = login.text_input('Username:', 'guest_login')
        password = login.text_input('Password:', 'ReadOnly!23', type='password')
        submit = login.form_submit_button()
        
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
                        st.write(f'Broken: {video}')
                    
                st.write('Meta Data')
                st.dataframe(meta_frame)
                st.write('')
                st.write('Audio Code')
                st.dataframe(audio_frame)

def train(local=True):
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
            AND drive IS NOT NULL
            AND channel IN {channel_filter}
            AND {year_filter}
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
            st.session_state['trained'] = trained
            st.session_state['i'] = 0
            st.session_state['sound'] = ''
            
            st.session_state['restrict_year'] = 'all' if year_filter == 'YEAR(publish_date) IS NOT NULL' else f'{equality}{year}'
            st.session_state['restrict_channel'] = 'all' if len(channel) == 7 else '|'.join(channel)
            
    
    if 'panda' in st.session_state:
        i = st.session_state['i']
        
        personality = st.session_state['panda']['channel'][i]
        sample = st.session_state['panda']['id'][i]
        second = st.session_state['panda']['second'][i]
        video_link = st.session_state['panda']['link'][i].split('v=')[-1]
        drive = st.session_state['panda']['drive'][i]
        
        if st.session_state['sound'] == '':
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
        
        if 'speaker' in models and 'gender' in models:
            upload_form = st.form('upload_both', clear_on_submit=True)
            left, middle, right = upload_form.columns(3)
            
            speaker_upload = left.radio('Speaker', personalities + ['Hogan', 'Guest', 'None'])
            speaker_year = st.session_state['restrict_year']
            speaker_channel = st.session_state['restrict_channel']
            
            gender_upload = middle.radio('Gender', ['Man', 'Woman', 'None'])
            gender_year = st.session_state['restrict_year']
            gender_channel = st.session_state['restrict_channel']
            
            send = upload_form.form_submit_button()
            
        elif 'speaker' in models:
            upload_form = st.form('upload_speaker', clear_on_submit=True)
            
            speaker_upload = upload_form.radio('Speaker', personalities + ['Hogan', 'Guest', 'None'])
            speaker_year = st.session_state['restrict_year']
            speaker_channel = st.session_state['restrict_channel']
            
            gender_upload = None
            gender_year = None
            gender_channel = None
            
            send = upload_form.form_submit_button()
            
        elif 'gender' in models:
            upload_form = st.form('upload_gender', clear_on_submit=True)
            
            speaker_upload = None
            speaker_year = None
            speaker_channel = None
            
            gender_upload = upload_form.radio('Gender', ['Man', 'Woman', 'None'])
            gender_year = st.session_state['restrict_year']
            gender_channel = st.session_state['restrict_channel']
            
            send = upload_form.form_submit_button()
            
        if send:
            new = pd.DataFrame({'channel': [personality], 'id': [sample], 'second': [second],
                                'speaker': [speaker_upload], 'gender': [gender_upload],
                                'slice_speaker_year': [speaker_year], 'slice_speaker_channel': [speaker_channel],
                                'slice_gender_year': [gender_year], 'slice_gender_channel': [gender_channel]})
            
            temp = st.session_state['trained'].merge(new, how='outer', on=['channel', 'id', 'second'])
            
            temp['speaker'] = np.where(temp.speaker_x.isnull(), temp.speaker_y, temp.speaker_x)
            temp['gender'] = np.where(temp.gender_x.isnull(), temp.gender_y, temp.gender_x)
            temp['slice_speaker_year'] = np.where(temp.slice_speaker_year_x.isnull(), temp.slice_speaker_year_y, temp.slice_speaker_year_x)
            temp['slice_speaker_channel'] = np.where(temp.slice_speaker_channel_x.isnull(), temp.slice_speaker_channel_y, temp.slice_speaker_channel_x)
            temp['slice_gender_year'] = np.where(temp.slice_gender_year_x.isnull(), temp.slice_gender_year_y, temp.slice_gender_year_x)
            temp['slice_gender_channel'] = np.where(temp.slice_gender_channel_x.isnull(), temp.slice_gender_channel_y, temp.slice_gender_channel_x)
            
            st.session_state['trained'] = temp[['channel', 'id', 'second', 'speaker', 'gender',
                                                'slice_speaker_year', 'slice_speaker_channel',
                                                'slice_gender_year', 'slice_gender_channel']]
            
            st.session_state['i'] = i + 1
            st.session_state['sound'] = ''
            
            st.experimental_rerun()
            
        if st.session_state['i'] > 0:
            azure = st.button('AZURE UPLOAD')
            
            if azure:
                with st.spinner('Uploading to Azure'):
                    if username == 'zangorth':
                        upload(st.session_state['trained'], 'ramsey', 'training', username, password, 'replace')
                        reindex('ramsey', 'training', ['channel', 'id', 'second'], username, password)
                        st.write('Upload Complete')
                        
                    else:
                        st.write('Data from Guest Profiles will not be uploaded')