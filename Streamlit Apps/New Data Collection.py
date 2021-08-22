from pytube import Playlist
import streamlit as st
import pyodbc as sql
import pandas as pd
import sys

sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')
import ramsey_helpers as RH

st.set_page_config(layout='wide')
st.title('The Ramsey Highlights')
st.header('New Data Collection')

options = st.sidebar.radio('', ['All Data', 'Link'])

if options == 'All Data':
    audio_location = st.sidebar.text_input('Audio Path:', r'C:\Users\Samuel\Audio\Audio Full')
    transcript_location = st.sidebar.text_input('Transcript Path:', r'C:\Users\Samuel\Audio\Transcript')
    
    run = st.sidebar.button('BEGIN COLLECTION')
    
    if run:
        with st.spinner('Identifying New Videos'):
            videos = Playlist('https://www.youtube.com/watch?v=0JUw1agDjoA&list=UU7eBNeDW1GQf2NJQ6G6gAxw&index=2').video_urls
            videos = list(videos)
            
            con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                              Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                              Trusted_Connection=yes;''')
                              
            query = 'SELECT * FROM RAMSEY.dbo.metadata'
            collected = pd.read_sql(query, con)
            con.close()
            
            videos = set(videos) - set(collected['link'])
        
        if len(videos) <= 0:
            st.write('No New Videos to Upload')
            
        
        else:
            metadata = pd.DataFrame(columns=['id', 'title', 'link', 'publish_date', 'keywords', 'seconds', 'rating', 'view_count'])
            
            first=True
            iteration_meta = st.empty()
            i, pb_meta = 0, st.progress(0)
            for video_link in videos:
                new = RH.NewData(video_link, audio_location, transcript_location)
                metadata = metadata.append(new.scrape(), ignore_index=True, sort=False)
                
                iteration_meta.text(f'Videos Processed: {i+1}/{len(videos)}')
                pb_meta.progress((i+1)/len(videos))
                i += 1
                
                iterables = new.iterables()
                
                iteration_coding = st.empty()
                j, pb_coding = 0, st.progress(0)
                audio_coding = []
                for sound in iterables:
                    audio_coding.append(new.encode_audio(sound))
                    
                    iteration_coding.text(f'Encoding Audio - Seconds Processed: {j+1}/{len(iterables)}')
                    pb_coding.progress((j+1)/len(iterables))
                    j += 1
                    
                iteration_coding.empty()
                pb_coding.empty()
                
                audio_coding = [clip for clip in audio_coding if clip != []]
                audio_coding = pd.DataFrame(audio_coding)
                
                if first:
                    audio_out = new.upload(audio_coding)
                    first = False
                    
                else:
                    temp = new.upload(audio_coding)
                    audio_out = audio_out.append(temp, ignore_index=True, sort=False)
            
            iteration_meta.empty()
            pb_meta.empty()
            
            st.write('Uploaded: Meta Data')
            st.dataframe(metadata, width=5000)
            st.write('')
            st.write('Updated: Audio Coding')
            st.dataframe(audio_out, width=5000)
        

elif options == 'Link':
    video_link = st.sidebar.text_input('Link:')
    audio_location = st.sidebar.text_input('Audio Path:', r'C:\Users\Samuel\Audio\Audio Full')
    transcript_location = st.sidebar.text_input('Transcript Path:', r'C:\Users\Samuel\Audio\Transcript')
    
    run = st.sidebar.button('BEGIN COLLECTION')
    
    if run:
        new = RH.NewData(video_link, audio_location, transcript_location)
        metadata = new.scrape()
        
        if type(metadata) == str:
            st.write(metadata)
        
        elif type(metadata) == pd.core.frame.DataFrame:
            st.write('Uploaded: Meta Data')
            st.dataframe(metadata, width=5000)
            
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
            
            audio_coding = [clip for clip in audio_coding if clip != []]
            audio_coding = pd.DataFrame(audio_coding)
            
            audio_coding = new.upload(audio_coding)
            
            st.write('')
            st.write('Updated: Audio Coding')
            st.dataframe(audio_coding, width=5000)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    