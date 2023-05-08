#!/usr/bin/env python
# coding: utf-8

# In[33]:


#Importing required python libraries

get_ipython().system(' pip install google-api-python-client')
import pandas as pd
import seaborn as sns
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from googleapiclient.discovery import build


# In[34]:


#API key was taken out from google-api-client
#Channel_id's were taken out from  youtube channels page

api_key = 'AIzaSyAD4AIW3juOinzGPSvAyVemggtcCVCyS64'


channel_ids = ['UCq-Fj5jknLsUf-MWSy4_brA'
                ,'UCOQNJjhXwvAScuELTT_i7cQ'
                ,'UC55IWqFLDH1Xp7iu1_xknRA'
                ,'UC6-F5tO8uklgE9Zy8IvbdFw'
                , 'UCpEhnqL0y41EpW2TvWAHD7Q'
               , 'UCRm96I5kmb_iGFofE5N691w'
               , 'UC8To9CFsZzvPafxMLzS08iA'
                ,'UCF1JIbMUs6uqoZEY1Haw0GQ']


youtube = build('youtube','v3',developerKey=api_key)


# In[39]:


#Function to get channel statistics


def get_channel_stats(youtube,channel_ids):
        request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=channel_ids)
        response = request.execute()
        return response
        


# In[40]:


chann = []
d = get_channel_stats(youtube,channel_ids)   #this is to select all the channel's data
e = d['items']   #From multiple parameters we only need items since it contains the statistical measures that we require
all = []
for s in range(len(channel_ids)):
    data = dict(
                    channel_name = e[s]['snippet']['title'],
                    subscriber = e[s]['statistics']['subscriberCount'],
                    viewCount = e[s]['statistics']['viewCount'],
                    videoCount = e[s]['statistics']['videoCount']
                )
    chann.append(data)
chann


# In[41]:


chann = pd.DataFrame(chann)   #Converting dictatinary to a dataframe using pandas for better analysis
chann['subscriber']


# In[42]:


#Converting numeric measure to int data type for analysis

chann['videoCount'] = chann['videoCount'].astype('int64') 
chann['subscriber'] = chann['subscriber'].astype('int64')


# In[43]:


#Checking data types of dataframe


chann.dtypes


# In[27]:


#Storing our data to a csv file

chann = chann.to_csv('channels_data')


# In[48]:


#plotting bar graph
#Channel Name vs channel subscribers

import numpy as np
#Function to add labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        

sns.set(rc={'figure.figsize':(15,5)})
ax = sns.barplot(x='channel_name',y='subscriber',data=chann)
x = np.array(chann['channel_name'])
y = np.array(chann['subscriber'])
addlabels(chann['channel_name'],chann['subscriber'])
plt.bar(x,y,color='blue')
plt.title("Subscriber vs channel_name")





# In[49]:


chann


# In[53]:


#analysis for SONY channels


sonyy = chann.copy()
sonyy.set_index('channel_name',inplace=True)


# In[56]:


# Selecting SONY channels from dataframe

sony = sonyy.loc[['Sony LIV','SET India','Sony SAB']]
sony_data = sony.to_csv('sony')  


# In[60]:


sony_channels_stats  =   sonyy.loc[['Sony LIV','SET India','Sony SAB']].describe()
sony_channels_statse = sony_channels_stats.to_csv('sony_channels_stats')


# In[61]:


#Plotting horizontal bar graph for SONY channels video counts

x = ['Sony LIV','SET India','Sony SAB']
x = np.array(x)
y = np.array(sony['videoCount'])




sns.set(rc={'figure.figsize':(15,5)})
plt.xlabel("Total Video Count")
plt.ylabel("Channels")
plt.barh(x, y,color='maroon')
#addlabels(channelss['channel_name'],channelss['videoCount'])
plt.title("Channels vs Total Video Counts for SONY channels")
plt.show()


# In[62]:


#Plotting bar graph for sony channels vs subscribers


x = ['Sony LIV','SET India','Sony SAB']
x = np.array(x)
y = np.array(sony['subscriber'])

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')


sns.set(rc={'figure.figsize':(15,5)})
plt.xlabel("Sony Channels")
plt.ylabel("subscriber count")
plt.bar(x, y,color='blue')
addlabels(x,y)
plt.title("Channels vs Total subscribers for SONY channel")
plt.show()


# # SCRAP, ANALYZE AND VISUALIZE YOUTUBE VIDEOS DETAILS
# 

# In[63]:


chann


# In[68]:


#writing loop to take out playlist id for the channels which will have the details for all videos uploaded by the channel


vid = []
for w in range(0,8):
    data = dict(Channel_name = e[w]['snippet']['title'],playlistid = e[w]['contentDetails']['relatedPlaylists']['uploads'])
    vid.append(data)
vid
    
        
        


# In[71]:


#Getting playlist id from the dataframe and selecting the id for JioCinema youtube channel

vid2 = pd.DataFrame(vid)
vid2
vid3 = vid2['playlistid']
vid3 = list(vid3)
vid3
vid4 = 'UU8To9CFsZzvPafxMLzS08iA'



# In[73]:


#Writing function to get the uploaded videos details for JioCinema


def get_video_ids(youtube, playlist_id):
    
    request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId = playlist_id,
                maxResults = 50)
    response = request.execute()
    
    video_ids = []
    
    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])
        
    next_page_token = response.get('nextPageToken')
    more_pages = True
    
    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                        part='contentDetails',
                        playlistId = playlist_id,
                        maxResults = 50,
                        pageToken = next_page_token)
            response = request.execute()
    
            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])
            next_page_token = response.get('nextPageToken')
        
    return video_ids


# In[74]:


video_ids = get_video_ids(youtube,vid4)


# # Function to get Video Details

# In[122]:


#Writing function to get analysing measures like Title,Publishdate,Views,Likes

all_data = []
def get_video_details(youtube,video_ids):
    for i in range(0,len(video_ids),50):
        request = youtube.videos().list(
        part="snippet,statistics",
        id=','.join(video_ids[i:i+50]))
        response = request.execute()  #this will have only 50 records
    
        for video in response['items']:
            video_stats = dict(
                                Title = video['snippet']['title'],
                                publishdate = video['snippet']['publishedAt'],
                                Views = video['statistics']['viewCount'],
                                Likes = video['statistics']['likeCount'],
                              )   
            all_data.append(video_stats)
    return all_data
 
   


# In[123]:


#getting the data and also converting to csv format to analysis purposes


data_details = get_video_details(youtube,video_ids)
data_details
data_details_jioo = pd.DataFrame(data_details)
tempd = pd.DataFrame(data_details)
jio_datacsv = tempd.to_csv('jio_datacs')


# In[124]:


data_details_jioo = pd.DataFrame(data_details)
data_details_jioo
data_details_jioo['Views'] = data_details_jioo['Views'].astype('int64')
data_details_jioo['Likes'] = data_details_jioo['Likes'].astype('int64')
data_details_jioo['publishdate'] = pd.to_datetime(data_details_jioo['publishdate']).dt.date


# In[140]:


data_details_jioo.dtypes


# In[141]:


data_details_jioo['publishdate'] = pd.to_datetime(data_details_jioo['publishdate'])


# # Top 10 Videos by Views chart ( matplotlib)

# In[142]:


#Top 10 Videos by Views chart

teps = data_details_jioo.sort_values(by='Views',ascending=False).head(10) 


title = list(teps['Title'])
Views = list(teps['Views'])
fig = plt.figure(figsize = (10, 5))
plt.barh(title, Views, color ='maroon')
plt.show()


# # Top 10 Videos by Views (sns plot)

# In[143]:



#Top 10 Videos by Views chart with sns
teps = data_details_jioo.sort_values(by='Views',ascending=False).head(10) 



title = list(teps['Title'])
Views = list(teps['Views'])
fig = plt.figure(figsize = (10, 5))
ax1 = sns.barplot(x='Views',y='Title',data=teps)



# In[147]:


teps = data_details_jioo.sort_values(by='Likes',ascending=False).head(10)  #most liked videos
tepss = tepss.to_csv('new')


# In[149]:


#Top 10 most Liked Videos 


teps = data_details_jioo.sort_values(by='Views',ascending=False).head(10) 


title = list(tepss['Title'])
Liked = list(tepss['Likes'])
fig = plt.figure(figsize = (10, 5))
#plt.barh(title, Liked, color ='orange')
#plt.show()
ax2 = sns.barplot(x='Likes',y='Title',data=tepss)
ax2


# # Monthly Video posting by jioCinema

# In[150]:


temps = data_details_jioo.copy() #copying to test publishdate date conversion


# In[153]:


tempx = tempd.copy()
tempx['publishdate'] = pd.to_datetime(tempx['publishdate'])


# In[154]:


tempx.dtypes


# In[155]:


tempx['publishmonth'] = pd.to_datetime(tempx['publishdate']).dt.strftime('%b')
tempx['year'] = pd.DatetimeIndex(tempx['publishdate']).year


# In[161]:


t = tempx.groupby(['year','publishmonth']).agg({'Title':'count'},as_index=False)
t.sort_values(by=['year','Title'],ascending=False)
g = t.copy()
total_videos_published = g.loc[:,:]
total_videos = total_videos_published.to_csv('total_videos')


# In[51]:


g.loc[([2016,2019,2021,2022],['Sep','Aug','Oct']),:]


# In[57]:


g.loc[([2016,2019,2021,2022,2023],['Sep','Aug','Oct','Nov','Dec','Jan']),:]


# In[162]:


# No. of videos posted in 2016

sixt = g.loc[2016]
a = sixt.index
sixt
b = sixt['Title']
plt.bar(a,b, color ='yellow')
plt.title('videos posted in 2016')
plt.xlabel('months')
plt.ylabel('total videos')


# In[163]:


#total video posting in 2022


r = g.sort_values(by=['year','Title'])
all = r.loc[2022]
a = all.index
b = all['Title']

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        
        
plt.bar(a,b,color='pink')
plt.title('total video posted in year 2022')
plt.xlabel('Months')
plt.ylabel('total videos')
addlabels(a,b)
plt.show()


# In[164]:


#total video posting in 2021


r = g.sort_values(by=['year','Title'])
all = r.loc[2021]
a = all.index
b = all['Title']

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        
        
plt.bar(a,b,color='pink')
plt.title('total video posting in december')
plt.xlabel('Months')
plt.ylabel('total videos')
addlabels(a,b)
plt.show()


# In[165]:


#Videos posted during IPl Season

ipl = g.loc[2023]
ipl.sort_values(by='Title')
month_dict = {'Jan':1,'Feb':2,'Mar':3, 'Apr':4, 'May':5}
ipl.sort_values('publishmonth', key = lambda x : x.apply (lambda x : month_dict[x]))
ipl = ipl.to_csv('ip')


# In[91]:


most_viewedt = data_details_jioo.sort_values(by='Views',ascending=False).head(10)
most_viewed
most_viewed = most_viewedt.to_csv('most')


# In[166]:


data_details_jioo

