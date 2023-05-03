#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install google-api-python-client')
import pandas as pd
import seaborn as sns
from googleapiclient.discovery import build
import matplotlib.pyplot as plt


# In[2]:


from googleapiclient.discovery import build


# In[3]:


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


# In[4]:


##Function to get channel statistics
def get_channel_stats(youtube,channel_ids):
        request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=channel_ids)
        response = request.execute()
        return response
        


# In[5]:


chann = []
d = get_channel_stats(youtube,channel_ids)
#for s in range(len(channel_ids)):
e = d['items']  #To take out statistics for the channels
all = []
for s in range(len(channel_ids)):
    #print(e[s]['statistics'])
    #print(e[s]['snippet']['title'])
    data = dict(
                    channel_name = e[s]['snippet']['title'],
                    subscriber = e[s]['statistics']['subscriberCount'],
                    viewCount = e[s]['statistics']['viewCount'],
                    videoCount = e[s]['statistics']['videoCount']
                )
#    print(data)
    chann.append(data)
chann


# In[6]:


chann = pd.DataFrame(chann)
chann


# In[10]:


chann['subscriber'] = chann['subscriber'].astype('int64')


# In[11]:


filt = (chann['subscriber'] >= 50000000) & (chann['videoCount'] > 100000)


# In[ ]:





# In[12]:


chann.dtypes


# In[13]:


chann


# In[14]:


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




# In[15]:


#plotting bar graph
#channel Name vs Views
#plotting horizontal bar chart
import numpy as np

x = np.array(chann['channel_name'])
y = np.array(chann['videoCount'])

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])

sns.set(rc={'figure.figsize':(20,5)})
plt.xlabel("Total Video Count")
plt.ylabel("Channels")
plt.bar(x, y,color='maroon')
addlabels(chann['channel_name'],chann['videoCount'])
plt.title("Channels vs Total Video Counts")
plt.show()



# In[16]:


chann


# In[17]:


#analysis for SONY channnels
sonyy = chann.copy()
sonyy.set_index('channel_name',inplace=True)
sonyy


# In[18]:


sony = sonyy.loc[['Sony LIV','SET India','Sony SAB']]
sony


# In[19]:


sonyy.loc[['Sony LIV','SET India','Sony SAB']].describe()


# In[20]:


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


# In[21]:


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


# # SCRAP, ANALYZE AND VISUALIZE VIDEO DETAILS
# 

# In[22]:


chann


# In[24]:


e[0]['contentDetails']['relatedPlaylists']['uploads']


# In[25]:


vid = []
for w in range(0,8):
    data = dict(Channel_name = e[w]['snippet']['title'],playlistid = e[w]['contentDetails']['relatedPlaylists']['uploads'])
    vid.append(data)
vid
    
        
        


# In[26]:


vid2 = pd.DataFrame(vid)
vid2
vid3 = vid2['playlistid']
vid3 = list(vid3)
vid3
print(vid3)
vid4 = 'UU8To9CFsZzvPafxMLzS08iA'



# In[27]:


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


# In[28]:


video_ids = get_video_ids(youtube,vid4)


# # Function to get Video Details

# In[29]:


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
                                #Comments = video['statistics']['commentCount']
                              )   
            all_data.append(video_stats)
    return all_data
 
   


# In[63]:


data_details = get_video_details(youtube,video_ids)
data_details
data_details_jio = pd.DataFrame(data_details)
tempd = pd.DataFrame(data_details)


# In[31]:


data_details_jio['Likes'] = data_details_jio['Likes'].astype('int64')


# In[32]:


data_details_jio.dtypes


# In[34]:


data_details_jio


# In[35]:


data_details_jio.groupby('publishdate').Title.agg(['count'])


# In[36]:


data_details_jioo = pd.DataFrame(data_details)
data_details_jioo
data_details_jioo['Views'] = data_details_jioo['Views'].astype('int64')
data_details_jioo['Likes'] = data_details_jioo['Likes'].astype('int64')
data_details_jioo['publishdate'] = pd.to_datetime(data_details_jioo['publishdate']).dt.date


# In[37]:


data_details_jioo.dtypes


# In[38]:


data_details_jioo['publishdate'] = pd.to_datetime(data_details_jioo['publishdate']).dt.date


# # Top 10 Videos by Views chart ( matplotlib)

# In[48]:


#Top 10 Videos by Views chart
teps = data_details_jioo.sort_values(by='Views',ascending=False).head(10) 


title = list(teps['Title'])
Views = list(teps['Views'])
fig = plt.figure(figsize = (10, 5))
plt.barh(title, Views, color ='maroon')
plt.show()


# # Top 10 Videos by Views (sns plot)

# In[51]:


# top 10 videos by Likes b

#Top 10 Videos by Views chart with sns
teps = data_details_jioo.sort_values(by='Views',ascending=False).head(10) 



title = list(teps['Title'])
Views = list(teps['Views'])
fig = plt.figure(figsize = (10, 5))
ax1 = sns.barplot(x='Views',y='Title',data=teps)



# # Monthly Video posting by jioCinema

# In[57]:


temps = data_details_jioo.copy() #copying to test publishdate date conversion


# In[83]:


tempx = tempd.copy()
tempx['publishdate'] = pd.to_datetime(tempx['publishdate']).dt.date


# In[84]:


tempx.dtypes


# In[94]:


tempx['publishmonth'] = pd.to_datetime(tempx['publishdate']).dt.strftime('%b')
tempx['year'] = pd.DatetimeIndex(tempx['publishdate']).year


# In[209]:


t = tempx.groupby(['year','publishmonth']).agg({'Title':'count'},as_index=False)
t.sort_values(by=['year','Title'],ascending=False)
g = t.copy()
g.loc[([2016,2019],'Oct'),:]


# In[211]:


g.loc[([2016,2019,2021],['Sep','Aug','Oct']),:]


# In[214]:


g.loc[2016:2018]


# In[262]:


# No. of videos posted in 2016

sixt = g.loc[2016]
a = sixt.index
sixt
b = sixt['Title']
plt.bar(a,b, color ='yellow')
plt.title('videos posted in 2016')
plt.xlabel('months')
plt.ylabel('total videos')


# In[264]:


#total video posting in 2022


r = g.sort_values(by=['year','Title'])
all = r.loc[2022]
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


# In[265]:


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


# In[ ]:




