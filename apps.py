import streamlit as st

st.set_page_config(page_title="Senticonomy", layout="wide")

# Define page selection
page = st.sidebar.selectbox("Select a page", ["Data Preprocess and Analysis", "Web Application"])

if page == "Data Preprocess and Analysis":
    st.title("\U0001F4E1 News Data Pipeline: Preprocess, Analyze, Upload")

    import pandas as pd
    import numpy as np
    import time
    import os
    import re
    import string
    import unicodedata
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import plotly.express as px
    import plotly.graph_objects as go
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    from newsapi import NewsApiClient
    import boto3
    from transformers import pipeline

    import nltk
    from nltk.data import find

    def download_nltk_data():
        try:
            find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    download_nltk_data()

    load_dotenv()

    if st.button("\U0001F680 Preprocess and Upload News Data"):
        try:
            st.info("Initializing API and environment variables...")
            newsapi = NewsApiClient(api_key=os.getenv("API_KEY"))
            s3_client = boto3.client('s3', aws_access_key_id=os.getenv('AWS_access_key'), aws_secret_access_key=os.getenv('AWS_secret_key'))

            s3_client.download_file('projectsenticonomy', 'updated_news.csv', 'updated_news.csv')
            master_file = 'updated_news.csv'
            st.success("Downloaded existing news data from AWS S3.")


            if os.path.exists(master_file):
                existing_news = pd.read_csv(master_file)
            else:
                existing_news = pd.DataFrame(columns=['link', 'headline', 'category', 'short_description', 'authors', 'date'])

            query_keywords = {
                'TECH': ['technology', 'tech news', 'gadgets', 'AI'],
                'SPORTS': ['sports', 'football', 'cricket', 'NBA'],
                'ENTERTAINMENT': ['movies', 'celebrity', 'music', 'entertainment'],
                'POLITICS': ['politics', 'government', 'elections'],
                'EDUCATION': ['education', 'students', 'schools', 'university'],
                'ENVIRONMENT': ['climate change', 'environment', 'pollution'],
                'SCIENCE': ['science', 'research', 'NASA', 'discovery'],
                'CRIME': ['crime', 'murder', 'theft', 'arrest'],
                'BUSINESS': ['business', 'finance', 'stocks', 'economy'],
                'TRAVEL': ['travel', 'tourism', 'vacation', 'flights'],
                'STYLE & BEAUTY': ['fashion', 'style', 'makeup', 'beauty']
            }

            news_data = []

            for category, keywords in query_keywords.items():
                for keyword in keywords:
                    try:
                        articles = newsapi.get_everything(q=keyword, language='en', sort_by='publishedAt', page_size=20)
                        for article in articles.get('articles', []):
                            news_data.append({
                                'link': article['url'],
                                'headline': article['title'],
                                'category': category,
                                'short_description': article['description'],
                                'authors': article.get('author', 'Unknown'),
                                'date': article['publishedAt']
                            })
                    except Exception as e:
                        print(f"Error fetching keyword '{keyword}' for {category}: {e}")
                    time.sleep(1.5)

            new_news_df = pd.DataFrame(news_data)
            combined_news = pd.concat([existing_news, new_news_df], ignore_index=True)
            combined_news.drop_duplicates(subset='link', keep='last', inplace=True)
            combined_news['category'] = combined_news['category'].str.upper()
            combined_news.to_csv(master_file, index=False)

            filtered_categories = list(query_keywords.keys())
            filtered_news = combined_news[combined_news['category'].isin(filtered_categories)]
            filtered_file = 'test_data.csv'
            filtered_news.to_csv(filtered_file, index=False)
            st.success("News fetched and saved locally.")

            st.info("Uploading raw data to AWS S3...")
            s3_client = boto3.client('s3', aws_access_key_id=os.getenv('AWS_access_key'), aws_secret_access_key=os.getenv('AWS_secret_key'))
            s3_client.upload_file(filtered_file, 'projectsenticonomy', 'raw_news_data.csv')
            st.success("raw_news_data Uploaded to AWS S3.")

            st.info("🧼 Cleaning data...")

            df = filtered_news.copy()

            # Remove duplicates and nulls in key columns
            df.drop_duplicates(inplace=True)
            df.dropna(subset=['headline', 'short_description'], inplace=True)

            # Replace 'nan' strings with np.nan
            df['category'] = df['category'].replace('nan', np.nan)
            df['authors'] = df['authors'].replace('nan', np.nan)
            df.dropna(subset=['category', 'authors'], inplace=True)

            # Convert 'authors' to string type
            df['authors'] = df['authors'].astype(str)

            # Convert and fix 'date' column
            df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Safely convert to datetime
            df.dropna(subset=['date'], inplace=True)  # Drop rows where date conversion failed
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')  # Format to 'YYYY-MM-DD'

            # Reset index after drops
            df.reset_index(drop=True, inplace=True)

            # Unicode cleaning function
            def clean_unicode(text):
                if not isinstance(text, str):
                    return text
                text = unicodedata.normalize("NFKD", text)
                text = re.sub(r'[\ud800-\udfff]', '', text)  # Remove invalid Unicode
                return text

            df['short_description'] = df['short_description'].apply(clean_unicode)
            
            # Save cleaned data
            # Save cleaned data
            cleaned_filename = "Cleaned_News_DataSet.csv"
            df.to_csv(cleaned_filename, index=False)

            st.info("Uploading cleaned data to AWS S3...")
            s3_client = boto3.client('s3', aws_access_key_id=os.getenv('AWS_access_key'), aws_secret_access_key=os.getenv('AWS_secret_key'))
            s3_client.upload_file(cleaned_filename, 'projectsenticonomy', 'Cleaned_News_DataSet.csv')
            st.success("Uploaded to AWS S3.")

        except Exception as e:
            st.error(f"❌ Error during preprocessing: {e}")


elif page == "Web Application":

    import os
    import pandas as pd
    import re
    import string
    import nltk
    import streamlit as st
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import plotly.graph_objects as go
    import sqlalchemy as sa
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    import boto3
    from transformers import pipeline


    import nltk
    from nltk.data import find

    def download_nltk_data():
        try:
            find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    download_nltk_data()


    # Stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Text preprocessing function
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(cleaned_tokens)

    # Load data and preprocess
    @st.cache_data
    def load_data():
        s3_client = boto3.client('s3', aws_access_key_id=os.getenv('AWS_access_key'), aws_secret_access_key=os.getenv('AWS_secret_key'))
        s3_client.download_file('projectsenticonomy', 'Cleaned_News_DataSet.csv', 'preprocessed.csv')
        df = pd.read_csv("preprocessed.csv", encoding='utf-8')
        df.drop_duplicates(subset='short_description', inplace=True)
        df.dropna(subset=['short_description'], inplace=True)
        df['short_description_clean'] = df['short_description'].astype(str).apply(preprocess_text)
        return df

    df = load_data()

    # Compute clusters
    @st.cache_resource
    def compute_clusters(data):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(data['short_description_clean'])
        k = KMeans(n_clusters=11, random_state=0)
        clusters = k.fit_predict(X)
        return vectorizer, k, clusters

    vectorizer, k, cluster_preds = compute_clusters(df)
    df['cluster'] = cluster_preds

    # Map clusters to categories
    cluster_to_category = df.groupby('cluster')['category'].agg(lambda x: x.value_counts().index[0] if not x.empty else "Unknown").to_dict()

    # Compute sentiment scores
    @st.cache_data
    def compute_sentiment(data):
        analyzer = SentimentIntensityAnalyzer()
        sentiments = data['short_description'].apply(lambda x: analyzer.polarity_scores(str(x)))
        sentiment_df = pd.DataFrame(list(sentiments))
        sentiment_df['sentiment_score'] = sentiment_df['compound']
        data = pd.concat([data.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
        return data

    df = compute_sentiment(df)

    # Derive 'category_cluster' field
    if 'category_cluster' not in df.columns:
        df['category_cluster'] = df['category'].astype(str) + "-" + df['cluster'].astype(str)

    # Store data in session state
    if 'df' not in st.session_state:
        st.session_state['df'] = df

    # Web Interface
    st.markdown("<h1 style='text-align: center;'>\U0001F9E0 Senticonomy Clustering & Sentiment WebApp</h1>", unsafe_allow_html=True)

    # Sidebar for new prediction
    # Sidebar for new prediction
    st.sidebar.header("\U0001F52E Predict New Text")
    user_input = st.sidebar.text_area("Enter a short description to analyze:")
    if user_input:
        cleaned_input = preprocess_text(user_input)
        user_vec = vectorizer.transform([cleaned_input])
        cluster = k.predict(user_vec)[0]
        category = cluster_to_category.get(cluster, "Unknown")
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(user_input)

        st.sidebar.markdown(f"**Cluster:** {cluster}")
        st.sidebar.markdown(f"**Predicted Category:** {category}")
        st.sidebar.markdown("**Sentiment Scores:**")
        st.sidebar.json(sentiment)

    # Upload to AWS RDS
    st.sidebar.subheader("\U0001F4BE Upload to RDS")
    if st.sidebar.button("Upload Data to RDS"):
        try:
            load_dotenv()
            host = os.getenv("RDS_host")
            port = 3306
            username = "project_admin"
            password = os.getenv("RDS_password")
            db = "news_data"

            engine_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{db}'
            engine = sa.create_engine(engine_string)

            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS Cleaned_news_data;"))
                conn.commit()

            df.to_sql('Cleaned_news_data', con=engine, index=False, if_exists='append')
            st.success("Data inserted successfully into MySQL database.")
            st.info(f"{len(df)} rows uploaded at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            st.error(f"Error uploading data to RDS: {e}") 




    # Page navigation
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import pipeline
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    # Sentiment analysis model from Hugging Face
    # sentiment_analyzer = pipeline(
    #     "sentiment-analysis",
    #     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    #     revision="714eb0f"
    # )

    # # Sentiment analysis and return insights
    # def analyze_sentiment(texts):
    #     results = sentiment_analyzer(texts)

    #     # Aggregate sentiment counts and collect confidence scores
    #     sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0}
    #     confidence_scores = {"POSITIVE": [], "NEGATIVE": []}
    #     text_lengths = []

    #     for result, text in zip(results, texts):
    #         sentiment_counts[result['label']] += 1
    #         confidence_scores[result['label']].append(result['score'])
    #         text_lengths.append(len(text))

    #     positive_confidence = np.mean(confidence_scores["POSITIVE"]) if sentiment_counts["POSITIVE"] > 0 else 0
    #     negative_confidence = np.mean(confidence_scores["NEGATIVE"]) if sentiment_counts["NEGATIVE"] > 0 else 0

    #     return sentiment_counts, confidence_scores, positive_confidence, negative_confidence, text_lengths, results

    # Initialize page view state
    if "page_view" not in st.session_state:
        st.session_state.page_view = "Home"

    def set_page(p):
        st.session_state.page_view = p

    # Navigation buttons
    st.markdown("---")
    cols = st.columns(4)
    with cols[0]:
        if st.button("\U0001F3E0 Home"): set_page("Home")
    with cols[1]:
        if st.button("\U0001F4CA Sentiment Scores"): set_page("Sentiment Scores")
    with cols[2]:
        if st.button("\U0001F52C Cluster Analysis"): set_page("Cluster Analysis")
    with cols[3]:
        if st.button("\U0001F4C1 Dataset"): set_page("Dataset")
    st.markdown("---")

    current = st.session_state.page_view

    # Dummy check to load df if available
    if 'df' in st.session_state:
        df = st.session_state['df']

    # Home Page
    if current == "Home":
        st.subheader("\U0001F3E0 Welcome to Senticonomy")
        st.markdown("""
            Welcome to the **Senticonomy Dashboard**, where we blend the science of sentiment with the art of clustering. \U0001F3AF

            \U0001F539 Navigate to **Sentiment Scores** to explore positive/neutral/negative dynamics.  
            \U0001F539 Head over to **Cluster Analysis** to uncover deeper insights.  
            \U0001F539 Transformer model has been integrated for text sentiment analysis.  
            \U0001F539 Check out the **Dataset** for raw data exploration.

            Explore wisely, visualize vividly!
        """)

        st.image("https://media.giphy.com/media/QBd2kLB5qDmysEXre9/giphy.gif", width=400)

        if st.button("Explore Sentiment Scores"):
            st.write("Navigate to the Sentiment Scores section to dive deeper!")

        import streamlit.components.v1 as components
        components.html("""
            <div style="
                position: fixed;
                right: 10px;
                bottom: 10px;
                font-size: 14px;
                color: gray;
                text-align: right;
                z-index: 9999;
            ">
                © 2025 Created by <strong>Santhosh Kumar</strong>
            </div>
        """, height=40)

    # Sentiment Model Page
    # elif current == "Sentiment Model":
    #     st.title("Interactive Sentiment Analysis and Visualization")

    #     user_input = st.text_area("Enter your text for sentiment analysis (separate sentences by a newline):")

    #     if st.button("Analyze Sentiment"):
    #         if user_input:
    #             texts = user_input.split("\n")
    #             sentiment_counts, confidence_scores, positive_confidence, negative_confidence, text_lengths, results = analyze_sentiment(texts)

    #             st.write("Sentiment Analysis Results")
    #             for idx, result in enumerate(results):
    #                 st.write(f"Text {idx + 1}: {texts[idx]}")
    #                 st.write(f"Sentiment: {result['label']} with Confidence: {result['score']:.2f}")

    #             labels = list(sentiment_counts.keys())
    #             sizes = list(sentiment_counts.values())

    #             fig1, ax1 = plt.subplots(figsize=(5, 5))
    #             ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
    #             ax1.set_title("Sentiment Distribution")
    #             st.pyplot(fig1)

    #             confidence_values = [positive_confidence, negative_confidence]

    #             fig2, ax2 = plt.subplots(figsize=(6, 6))
    #             ax2.bar(labels, confidence_values, color=['green', 'red'])
    #             ax2.set_title("Average Confidence Scores for Sentiment")
    #             ax2.set_xlabel("Sentiment")
    #             ax2.set_ylabel("Average Confidence Score")
    #             st.pyplot(fig2)

    #             sentiments_numeric = [1 if result['label'] == 'POSITIVE' else 0 for result in results]

    #             fig3, ax3 = plt.subplots(figsize=(5, 5))
    #             ax3.scatter(text_lengths, sentiments_numeric, color=['green' if sentiment == 1 else 'red' for sentiment in sentiments_numeric])
    #             ax3.set_title("Text Length vs Sentiment")
    #             ax3.set_xlabel("Text Length")
    #             ax3.set_ylabel("Sentiment (1: Positive, 0: Negative)")
    #             st.pyplot(fig3)

    #         else:
    #             st.warning("Please enter some text for analysis.")

    # Sentiment Scores Page
    elif current == "Sentiment Scores":
        st.subheader("📈 Sentiment Score Overview")

        # 1. Diverging Bar Chart for Cluster Sentiment
        sentiment_avg = df.groupby('cluster')['sentiment_score'].mean().reset_index()
        sentiment_avg['sentiment_type'] = sentiment_avg['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
        sentiment_avg['color'] = sentiment_avg['sentiment_type'].map({'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'})
        st.markdown("### Diverging Sentiment Bar Chart by Cluster")
        fig1 = go.Figure(go.Bar(
            x=sentiment_avg['sentiment_score'],
            y=sentiment_avg['cluster'].astype(str),
            orientation='h',
            marker=dict(color=sentiment_avg['color']),
            text=sentiment_avg['sentiment_score'].round(2),
            textposition='auto'))
        fig1.update_layout(title="Sentiment Score by Cluster", xaxis_title="Sentiment Score", yaxis_title="Cluster")
        st.plotly_chart(fig1)

        # 2. Time Series Line Graph with Category Labels
        if {'date', 'category'}.issubset(df.columns):
            st.markdown("### Sentiment Over Time by Category")
            df['date'] = pd.to_datetime(df['date'])
            time_sentiment = df.groupby(['date', 'category'])['sentiment_score'].mean().reset_index()
            fig2 = px.line(time_sentiment, x='date', y='sentiment_score', color='category', markers=True,
                        title='Average Sentiment Over Time by Category')
            st.plotly_chart(fig2)

        # 3. Radar Chart for Average Sentiment per Category
        st.markdown("### Radar Sentiment View by Category")
        radar_df = df.groupby('category')['sentiment_score'].mean().reset_index()
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df['sentiment_score'],
            theta=radar_df['category'],
            fill='toself',
            name='Category Sentiment'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
                                title="Sentiment Radar by Category")
        st.plotly_chart(fig_radar)

        # 4. Heatmap by Category and Cluster
        st.markdown("### Heatmap: Sentiment Score by Category and Cluster")
        heatmap_data = df.groupby(['category', 'cluster'])['sentiment_score'].mean().unstack()
        fig3 = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='RdYlGn')
        st.plotly_chart(fig3)

        # 5. Geographical Sentiment Map (if location exists)
        if {'lat', 'lon'}.issubset(df.columns):
            st.markdown("### Sentiment by Location")
            map_data = df[['lat', 'lon', 'sentiment_score']]
            fig4 = px.scatter_mapbox(map_data, lat="lat", lon="lon", color="sentiment_score",
                                    color_continuous_scale='RdYlGn', zoom=3,
                                    mapbox_style="carto-positron", title="Geographical Sentiment Scores")
            st.plotly_chart(fig4)

        # Optional: Filter and Show Detailed Sentiment
        selected_cluster = st.selectbox("Select Cluster for Sentiment Details", options=sorted(df['cluster'].unique()))
        cluster_data = df[df['cluster'] == selected_cluster]
        st.markdown(f"### Sentiment Distribution for Cluster {selected_cluster}")
        sentiment_dist = cluster_data['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')).value_counts()
        st.bar_chart(sentiment_dist)

        

    # Cluster Analysis Page
    elif current == "Cluster Analysis":
        import plotly.express as px
        import pandas as pd
        import numpy as np
        st.subheader("🧪 Cluster-Based Sentiment Analysis")

        @st.cache_data
        def get_cluster_sentiment_stats(df):
            """Cache the sentiment statistics for clusters to speed up rendering"""
            avg_sentiment = df.groupby('cluster')['sentiment_score'].mean()
            cluster_category_dist = df.groupby(['cluster', 'category']).size().unstack(fill_value=0)
            return avg_sentiment, cluster_category_dist

        @st.cache_data
        def sample_data(df, num_rows=1000):
            """Sample a subset of the dataset for performance testing"""
            return df.sample(n=min(num_rows, len(df)), random_state=42)

        # Cache the statistics
        df_sampled = sample_data(df, num_rows=1000)  # Limit to 1000 rows for performance testing
        avg_sentiment, cluster_category_dist = get_cluster_sentiment_stats(df_sampled)

        # Average Sentiment Score per Cluster
        st.markdown("### Average Sentiment Score per Cluster")
        st.bar_chart(avg_sentiment)

        # Cluster-Category Breakdown
        st.markdown("### Cluster-Category Breakdown")
        st.dataframe(cluster_category_dist)

        # Cluster Sentiment Boxplot
        st.markdown("### Sentiment Boxplot by Cluster")
        fig_boxplot = px.box(df_sampled, x="cluster", y="sentiment_score", points="all", title="Sentiment Score Boxplot by Cluster")
        st.plotly_chart(fig_boxplot)

        # Filtered Data for Specific Cluster
        cluster_filter = st.selectbox("Select a Cluster to View", options=sorted(df['cluster'].unique()))
        filtered_data = df[df['cluster'] == cluster_filter]

        # Limit data shown in the filtered table to improve performance
        filtered_data_sampled = filtered_data.head(100)  # Show only the top 100 entries for performance

        st.markdown(f"Showing {len(filtered_data_sampled)} entries in Cluster {cluster_filter}")
        st.dataframe(filtered_data_sampled[['short_description', 'sentiment_score', 'category']])
        st.download_button("Download Filtered Data", data=filtered_data.to_csv(index=False), file_name=f"Cluster_{cluster_filter}_Data.csv")



    # Dataset Page
    elif current == "Dataset":
        st.subheader("🗃 Full Dataset View")
        st.dataframe(df)
        st.download_button("Download Full Data", data=df.to_csv(index=False), file_name="Senticonomy_Data.csv")


