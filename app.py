
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("netflix shows.csv")
    df.columns = df.columns.str.strip().str.lower()
    df = df[['title', 'listed inside', 'country', 'type', 'release year', 'duration']].dropna(subset=['title', 'listed inside'])
    df['title'] = df['title'].str.strip()
    df['listed inside'] = df['listed inside'].str.strip()
    df = df.drop_duplicates(subset='title')
    df['combined_features'] = df['listed inside'] + ' ' + df['country'].fillna('') + ' ' + df['type'].fillna('')
    return df

@st.cache_data
def create_similarity_matrix(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Recommendation function
def recommend_show(title, df, cosine_sim, indices, num_recommendations=5):
    title = title.strip()
    idx = indices.get(title)
    if idx is None:
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    show_indices = [i[0] for i in sim_scores]
    return df.iloc[show_indices][['title', 'listed inside', 'country', 'release year']]

# Streamlit UI
st.title("🎬 Netflix Show Recommendation System")

df = load_data()
cosine_sim = create_similarity_matrix(df)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

show_input = st.text_input("Enter a show/movie name:")

if st.button("Recommend"):
    if show_input:
        recommendations = recommend_show(show_input, df, cosine_sim, indices)
        if recommendations is not None and not recommendations.empty:
            st.write("### Top Recommendations:")
            st.dataframe(recommendations.reset_index(drop=True))
        else:
            st.error(f"❌ Show '{show_input}' not found in the dataset.")
    else:
        st.warning("Please enter a show/movie name to get recommendations.")
