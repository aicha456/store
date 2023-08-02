
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
uploaded_file = st.file_uploader("Choose a file")

df = pd.read_csv("https://github.com/aicha456/store/blob/main/Groceries_dataset%20(1).csv")
df['itemDescription']=df['itemDescription'].str.strip()
df.Date = pd.to_datetime(df.Date)
df['Member_number'] = df['Member_number'].astype('str')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df['Date']=df['Date'].astype('datetime64[ns]')
df['Month'] = df.Date.dt.month
df['Month'] = df['Date'].apply(lambda x:x.month)
df['Day of Week'] = df['Date'].apply(lambda time: time.dayofweek)
df['Year'] = df['Date'].apply(lambda t:t.year)
df['Day'] = df['Date'].apply(lambda t : t.day)

limi=list((df['Month'].unique()))

limit=st.sidebar.select_slider('filter by year?',options=[1,2,3,4,5,6,7,8,9,10,11,12],value=(2,6))
if limit:
    df=df[df["Month"].isin(limit)]

else:
    df=df
df
top_25=df.itemDescription.value_counts().sort_values(ascending=False)[0:25]
fig = px.bar(top_25,color=top_25.index, labels={'value':'Quantity Sold','index':'GroceryItems'})
fig.update_layout(showlegend=False, title_text='Top 25 Groceries Sold',title_x=0.5, title={'font':{'size':20}})
st.plotly_chart(fig)

bot_25=df.itemDescription.value_counts().sort_values(ascending=False)[-25:]
fi = px.bar(bot_25,color=bot_25.index, labels={'value':'Quantity Sold','index':'GroceryItems'})
fi.update_layout(showlegend=False, title_text='Bottom 25 Groceries Sold',title_x=0.5, title={'font':{'size':20}})
st.plotly_chart(fi)
top_25c = df.groupby('Member_number').agg(PurchaseQuantity=('itemDescription','count')).sort_values(by='PurchaseQuantity',ascending=False)[0:25].reset_index()

fii = px.bar(top_25c,x='Member_number',y='PurchaseQuantity')
st.plotly_chart(fii)


item_freq = df.groupby(pd.Grouper(key='itemDescription')).size().reset_index(name='count')
fiiig = px.treemap(item_freq, path=['itemDescription'], values='count')
fiiig.update_layout(title_text='Frequency of the Items Sold', title_x=0.5, title_font=dict(size=18))
fiiig.update_traces(textinfo="label+value")


st.plotly_chart(fiiig)

temp3 = df.groupby(['Year','Day'], as_index = False).agg(Sales = ('itemDescription', 'count'))

ig = px.line(temp3, x = 'Day', y = 'Sales', color = 'Year')
ig.update_layout(title_text = 'Sales Per Days of the Month', title_x = 0.5,
                 title = {'font':{'size':20}})

st.plotly_chart(ig)
