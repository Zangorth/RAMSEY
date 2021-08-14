from matplotlib import pyplot as plt
import seaborn as sea
import pyodbc as sql
import pandas as pd
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')

con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = '''
SELECT predictions.id, [second], 
    CASE WHEN speaker = 'A' THEN 'AO'
        WHEN speaker = 'C' THEN 'Coleman'
        WHEN speaker = 'D' THEN 'Deloney'
        WHEN speaker = 'G' THEN 'Guest'
        WHEN speaker = 'H' THEN 'Hogan'
        WHEN speaker = 'N' THEN 'Silence'
        WHEN speaker = 'R' THEN 'Ramsey'
        WHEN speaker = 'RC' THEN 'Cruze'
        WHEN speaker = 'W' THEN 'Wright'
        END AS speaker,
    DATENAME(dw, publish_date) AS 'dow',
    YEAR(publish_date)*100 + MONTH(publish_date) AS 'YYYYMM',
    publish_date
FROM RAMSEY.dbo.predictions
LEFT JOIN RAMSEY.dbo.metadata
    ON predictions.id = metadata.id
'''

panda = pd.read_sql(query, con)
con.close()

################
# Volume Plots #
################
dates = panda.groupby(['publish_date', 'dow']).size()/60
dates = dates.reset_index()
dates.columns = ['publish_date', 'dow', 'minutes']

dates = dates.sort_values('publish_date')
dates['minute_average'] = dates['minutes'].rolling(window=7).mean()

dates['dow_minute_average'] = dates.groupby('dow')['minutes'].rolling(window=7).mean().reset_index().sort_values('level_1')[['minutes']].reset_index(drop=True)

#palette_options = ['CMRmap_r', 'afmhot_r', 'brg', 'cool', 'cubehelix', 'flag', 'gist_ncar_r', 'hsv',
#                   'icefire', 'inferno', 'magma', 'nipy_spectral', 'BuPu', 'Paired']

#for pal in palette_options:
#    sea.lineplot(x='publish_date', y='dow_minute_average', hue='dow', 
#                 hue_order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
#                 data=dates, palette=pal)
#    plt.title(f'Amount of Content Published by "Ramsey Show Highlights" by Day - {pal}')
#    plt.legend(loc='upper center', ncol=7)
#    plt.ylabel('Minutes Published')
#    plt.xlabel('Date Published')
#    plt.show()
#    plt.close()

sea.set_style('whitegrid')
sea.set(rc={'figure.figsize':(16, 9)})
sea.lineplot(x='publish_date', y='minute_average',
             data=dates, palette='icefire')
plt.title('Amount of Content Published by "Ramsey Show Highlights" by Day\nSeven Day Rolling Average')
plt.ylabel('Minutes Published')
plt.xlabel('Date Published')
plt.savefig(r'Plots\Amount of Content Published.png')
plt.show()
plt.close()


sea.set_style('whitegrid')
sea.set(rc={'figure.figsize':(16, 9)})
sea.lineplot(x='publish_date', y='dow_minute_average', hue='dow', 
             hue_order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
             data=dates, palette='icefire')
plt.title('Amount of Content Published by "Ramsey Show Highlights" by Day\nSeven Day Rolling Average by Day of Week')
plt.legend(loc='upper center', ncol=7)
plt.ylabel('Minutes Published')
plt.xlabel('Date Published')
plt.savefig(r'Plots\Amount of Content Published DOW.png')
plt.show()
plt.close()

#################
# Speaker Plots #
#################
speakers = panda.loc[panda.speaker != 'Silence']
speakers = (speakers.groupby(['publish_date', 'speaker']).size()/60).reset_index()
speakers.columns = ['publish_date', 'speaker', 'minutes']

for date in list(set(speakers.publish_date)):
    for speaker in list(set(speakers.speaker)):
        if len(speakers.loc[(speakers.publish_date == date) & (speakers.speaker == speaker)]) == 0:
            append = pd.DataFrame({'publish_date': date, 'speaker': speaker, 'minutes': [0]})
            speakers = speakers.append(append, ignore_index=True, sort=False)
            

speakers = speakers.sort_values(['speaker', 'publish_date']).reset_index(drop=True)
speakers['speaker_sum_roll'] = speakers.groupby('speaker')['minutes'].rolling(window=7).sum().reset_index().sort_values('level_1')[['minutes']].reset_index(drop=True)

daily_sum = speakers.groupby('publish_date')['minutes'].sum().reset_index().dropna().reset_index(drop=True)
daily_sum['daily_sum_roll'] = daily_sum['minutes']

for i in range(1, 7):
    daily_sum['daily_sum_roll'] += daily_sum['minutes'].shift(i)

speakers = speakers.merge(daily_sum[['publish_date', 'daily_sum_roll']], on='publish_date', how='left')
speakers['speaking_percent'] = speakers['speaker_sum_roll']/speakers['daily_sum_roll']


sea.set(rc={'figure.figsize':(17, 9)})
sea.set_style('whitegrid')
sea.lineplot(x='publish_date', y='speaking_percent', hue='speaker',
             data=speakers, palette='Paired')
plt.title('Percent of Video Time each Host Spends Talking\n F1 Score = 0.9')
plt.ylabel('Percent')
plt.xlabel('Date')
plt.savefig(r'Plots\Speaker Percent.png')
plt.show()
plt.close()


speakers = speakers.loc[speakers.speaker != 'Ramsey']
speakers = speakers.loc[speakers.speaker != 'Guest']
speakers = speakers.loc[pd.to_datetime(speakers.publish_date).dt.year > 2019]

sea.set(rc={'figure.figsize':(17, 9)})
sea.set_style('whitegrid')
sea.lineplot(x='publish_date', y='speaking_percent', hue='speaker',
             data=speakers)
plt.title('Percent of Video Time each Host Spends Talking\n F1 Score = 0.9')
plt.ylabel('Percent')
plt.xlabel('Date')
plt.savefig(r'Plots\Speaker Percent (No Ramsey).png')
plt.show()
plt.close()























