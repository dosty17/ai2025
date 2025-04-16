import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already present
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

st.set_page_config(
    page_title="VoC sentiment analysis",
    page_icon="🕸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("VoC: Sentiment Analysis POC")
st.markdown("------------------------------------------------------------------------------------")

filename = st.sidebar.file_uploader("Upload reviews data:", type=("csv", "xlsx"))

if filename is not None:
    data = pd.read_csv(filename)
    data["body"] = data["body"].astype("str")
    data["score"] = data["body"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    data["sentiment"] = np.where(data['score'] >= .5, "Positive", "Negative")
    data = data[['brand', 'body', 'sentiment', 'score', 'date']]
    data['date'] = pd.to_datetime(data['date'])
    data['quarter'] = pd.PeriodIndex(data.date, freq='Q')

    per_dt = data.groupby(['brand', 'sentiment']).size().reset_index()
    per_dt = per_dt.sort_values(['sentiment'], ascending=False)
    per_dt1 = data.groupby(['brand']).size().reset_index()
    per_dt2 = pd.merge(per_dt, per_dt1, how='left', on='brand')
    per_dt2['Sentiment_Percentage'] = per_dt2['0_x'] / per_dt2['0_y']
    per_dt2 = per_dt2[['brand', 'sentiment', 'Sentiment_Percentage']]

    brand_c = data.groupby(['brand']).size().reset_index()
    st.sidebar.write("Reviews count by brand:")
    for i, row in brand_c.iterrows():
        st.sidebar.write(f"{row['brand']} : {row[0]}")

    st.subheader("Phone Reviews Sentiment distribution")
    col3, col4 = st.columns(2)

    with col4:
        data1 = data[data['brand'] == 'Nokia']
        sentiment_count = data1.groupby(['sentiment'])['sentiment'].count()
        sentiment_count = pd.DataFrame({'Sentiments': sentiment_count.index, 'sentiment': sentiment_count.values})
        fig = px.pie(sentiment_count, values='sentiment', names='Sentiments', width=550, height=400)
        fig.update_layout(title_text='Sentiment distribution for Nokia', title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        trend_dt = data[data['brand'] == 'Nokia']
        trend_dt['Review_Month'] = trend_dt['date'].dt.strftime('%m-%Y')
        trend_dt1 = trend_dt.groupby(['Review_Month', 'sentiment']).size().reset_index()
        trend_dt1 = trend_dt1.sort_values(['sentiment'], ascending=False)
        trend_dt1.rename(columns={0: 'Sentiment_Count'}, inplace=True)

        fig2 = px.line(trend_dt1, x="Review_Month", y="Sentiment_Count", color='sentiment', width=600, height=400)
        fig2.update_layout(title_text='Trend analysis of sentiments for Nokia', title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("------------------------------------------------------------------------------------")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(data, x="brand", y="sentiment", histfunc="count", color="sentiment",
                           facet_col="sentiment", labels={"sentiment": "sentiment"},
                           width=550, height=400)
        fig.update_layout(title_text='Distribution by count of sentiment', title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig1 = px.histogram(per_dt2, x="brand", y="Sentiment_Percentage", color="sentiment",
                            facet_col="sentiment", labels={"sentiment": "sentiment"},
                            width=550, height=400)
        fig1.update_layout(yaxis_title="Percentage", title_text='Distribution by percentage of sentiment', title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("------------------------------------------------------------------------------------")
    st.subheader("Word Cloud for reviews Sentiment")

    word_ls = ['phone.', 'phone,', 'will', 'window', 'really', 'andoid', 'tracfone', 'minute', 'best', 'time',
               'amazon', 'need', 'still', 'work', 'phone', 'huawei', 'samsung', 'nokia', 'windows phone', 'great',
               'good', 'use', 'love', 'one', 'amazing', 'still used', 'lumia', 'iphone']
    data['body1'] = data['body'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in word_ls]))
    data['body1'] = data['body1'].str.replace('phone', ' ', regex=False)

    brands = ['Nokia', 'HUAWEI', 'Samsung']
    col_pairs = [st.columns(2), st.columns(2), st.columns(2)]

    for i, brand in enumerate(brands):
        with col_pairs[i][0]:
            df = data[(data["sentiment"] == "Positive") & (data["brand"] == brand) & (data['score'] > .9)]
            words = " ".join(df["body1"])
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(words)
            plt.imshow(wordcloud)
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Positive reviews word cloud for {brand}")
            st.pyplot()

        with col_pairs[i][1]:
            df = data[(data["sentiment"] == "Negative") & (data["brand"] == brand) & (data['score'] <= .2)]
            words = " ".join(df["body1"])
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640, colormap="RdYlGn").generate(words)
            plt.imshow(wordcloud)
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Negative reviews word cloud for {brand}")
            st.pyplot()

    st.markdown("------------------------------------------------------------------------------------")
    st.subheader("Top 5 positive reviews for Nokia:")
    pos = data[(data['brand'] == 'Nokia') & (data['score'] > .9)].sort_values(['score'], ascending=False).reset_index()

    for i in range(min(5, len(pos))):
        st.write(f"{i+1}. Nokia | Positive | Sentiment Score: {pos['score'].iloc[i]} - {pos['body'].iloc[i]}")

    st.markdown("------------------------------------------------------------------------------------")
    st.subheader("Top 5 negative reviews for Nokia:")
    neg = data[(data['brand'] == 'Nokia') & (data['score'] < .1)].sort_values(['score']).reset_index()

    for i in range(min(5, len(neg))):
        st.write(f"{i+1}. Nokia | Negative | Sentiment Score: {neg['score'].iloc[i]} - {neg['body'].iloc[i]}")
