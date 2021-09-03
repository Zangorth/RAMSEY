from helpers.ramsey import data_collect
from pytube import Playlist
import streamlit as st
import pyodbc as sql
import pandas as pd

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