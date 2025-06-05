import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inject Bootstrap & Netflix style CSS
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

<style>
body, .main {
    background-color: #000000 !important;
    color: white !important;
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
}

h1, .display-4 {
    color: #E50914 !important; /* Netflix red */
    font-weight: 900;
    letter-spacing: 3px;
    margin-bottom: 1.5rem;
}

.form-control {
    background-color: #222 !important;
    color: white !important;
    border: 2px solid #E50914 !important;
    border-radius: 8px !important;
    padding: 12px 15px !important;
    font-size: 1.1rem !important;
}

.form-control:focus {
    border-color: #F40612 !important;
    box-shadow: 0 0 8px #F40612 !important;
    background-color: #222 !important;
    color: white !important;
}

.btn-primary {
    background-color: #E50914 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    padding: 12px 30px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 10px rgba(229,9,20,0.7);
    transition: background-color 0.3s ease;
}

.btn-primary:hover {
    background-color: #B0060F !important;
    box-shadow: 0 6px 14px rgba(176,6,15,0.9);
}

.table {
    background-color: #1E1E1E !important;
    border-radius: 10px;
    color: white !important;
}

.table th {
    background-color: #E50914 !important;
    color: white !important;
    font-weight: 700;
    padding: 12px 20px;
    border: none;
}

.table td {
    padding: 12px 20px;
    border-bottom: 1px solid #333;
}

.container {
    max-width: 900px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

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

def recommend_show(title, df, cosine_sim, indices, num_recommendations=5):
    title = title.strip()
    idx = indices.get(title)
    if idx is None:
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    show_indices = [i[0] for i in sim_scores]
    return df.iloc[show_indices][['title', 'listed inside', 'country', 'release year']]

st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<h1 class="display-4 text-center">Netflix Show Recommendation</h1>', unsafe_allow_html=True)

df = load_data()
cosine_sim = create_similarity_matrix(df)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

show_input = st.text_input("", placeholder="Enter a show or movie name here...", key="show_input", help="Type the name exactly as in the dataset")

if st.button("Recommend", key="recommend_button"):
    if show_input:
        recommendations = recommend_show(show_input, df, cosine_sim, indices)
        if recommendations is not None and not recommendations.empty:
            # Display table with bootstrap classes
            st.markdown(recommendations.to_html(classes='table table-hover table-striped text-white', index=False), unsafe_allow_html=True)
        else:
            st.error(f"Show '{show_input}' not found in the dataset. Try exact name.")
    else:
        st.warning("Please enter a show or movie name.")

st.markdown('</div>', unsafe_allow_html=True)
