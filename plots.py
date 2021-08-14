from matplotlib import pyplot as plt
import pandas as pd
import pyodbc as sql
import seaborn as sea

sea.set_style('whitegrid')
sea.set(rc={'figure.figsize':(16, 9)})

con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = '''
SELECT predictions.id, [second], speaker,
    DATENAME(dw, publish_date) AS 'dow',
    publish_date
FROM RAMSEY.dbo.predictions
LEFT JOIN RAMSEY.dbo.metadata
    ON predictions.id = metadata.id
'''

panda = pd.read_sql(query, con)
con.close()

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


sea.lineplot(x='publish_date', y='minute_average',
             data=dates, palette='icefire')
plt.title('Amount of Content Published by "Ramsey Show Highlights" by Day\nSeven Day Rolling Average')
plt.ylabel('Minutes Published')
plt.xlabel('Date Published')
plt.show()
plt.close()


sea.lineplot(x='publish_date', y='dow_minute_average', hue='dow', 
             hue_order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
             data=dates, palette='icefire')
plt.title('Amount of Content Published by "Ramsey Show Highlights" by Day\nSeven Day Rolling Average by Day of Week')
plt.legend(loc='upper center', ncol=7)
plt.ylabel('Minutes Published')
plt.xlabel('Date Published')
plt.show()
plt.close()

