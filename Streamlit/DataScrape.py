from ramsey.ramsey import Scrape, upload
from pytube import Playlist
import streamlit as st
from time import sleep
import pyodbc as sql
import pandas as pd

###################
# Define Function #
###################
def data_collect(video_link, audio_location, transcript_location):
    with st.spinner('Downloading Audio'):
        new = Scrape(video_link, audio_location, transcript_location)
        metadata = new.metadata()
        new.audio()
        new.transcript()
    
    if type(metadata) == str:
        return []
    
    elif type(metadata) == pd.core.frame.DataFrame:
        with st.spinner('Generating Audio Slices'):
            iterables = new.iterables()
                
        iteration = st.empty()
        i, pb = 0, st.progress(0)
        audio_coding = []
        for sound in iterables:
            audio_coding.append(new.encode_audio(sound))
            
            iteration.text(f'Encoding Audio - Seconds Processed: {i+1}/{len(iterables)}')
            pb.progress((i+1)/len(iterables))
            
            i += 1
                
        iteration.empty()
        pb.empty()
    
        audio_coding = pd.concat(audio_coding)
        
        with st.spinner('Uploading Metadata'):
            metadata['seconds'] = audio_coding['second'].max()
            upload(metadata, 'ramsey', 'metadata')
        
        with st.spinner('Uploading Audio Data'):
            upload(audio_coding, 'ramsey', 'audio')
        
    return [metadata, audio_coding]

#################
# Streamlit App #
#################
st.title('The Ramsey Highlights')
st.header('New Data Collection')


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
                             'Server=ZANGORTH;DATABASE=HomeBase;' +
                             'Trusted_Connection=yes;')
        con = sql.connect(connection_string)
        query = 'SELECT * FROM ramsey.metadata'
        collected = pd.read_sql(query, con)
        con.close()
        
        videos = list(set(videos) - set(collected['link']))
        
    first = True
    meta_frame = None
    iteration_meta = st.empty()
    i, pb_meta = 0, st.progress(0)
    for video in videos:
        iteration_meta.text(f'Processing Video: {i+1}/{len(videos)}')
        pb_meta.progress((i+1)/len(videos))
        i += 1
        
        try:
            out = data_collect(video, audio_location, transcript_location)
        
            if first:
                meta_frame = out[0]
                audio_frame = out[1]
                
                first = False
                
            else:
                meta_frame = meta_frame.append(out[0], ignore_index=True, sort=False)
                audio_frame = audio_frame.append(out[1], ignore_index=True, sort=False)
                
        except:
            with st.spinner('Video Less than One Week Old'):
                sleep(1)
    
    if meta_frame is not None:
        st.write('Meta Data')
        st.dataframe(meta_frame)
        
    else:
        st.write('No New Videos to Upload')