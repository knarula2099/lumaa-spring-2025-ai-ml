import pandas as pd
import ast
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def clean_text(text):
    """
    Cleans the input text by handling edge cases and converting it to lowercase.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    try:
        if text.startswith("["):
            return " ".join(ast.literal_eval(text))
        return text.lower().strip()
    except (ValueError, SyntaxError):
        return text.lower().strip()


def preprocess_data(filepath):
    """
    Loads and preprocesses the dataset by cleaning text fields and combining them with weighted features.

    Args:
        filepath (str): The path to the CSV file containing the movie dataset.

    Returns:
        DataFrame: The preprocessed movie dataset.
    """
    movies = pd.read_csv(filepath)

    # Clean text fields
    movies["genres"] = movies["genres"].apply(clean_text)
    movies["keywords"] = movies["keywords"].apply(clean_text)
    movies["overview"] = movies["overview"].fillna("")

    # Combine text fields with text features
    movies["combined_text"] = (
        (movies["genres"] + " ") +
        movies["keywords"] + " " +
        movies["overview"]
    )

    # Drop rows with missing descriptions
    movies = movies.dropna(subset=["combined_text"])
    return movies


def train_tfidf(movies):
    """
    Fits a TF-IDF vectorizer on the combined text of the movies dataset.

    Args:
        movies (DataFrame): The preprocessed movie dataset.

    Returns:
        TfidfVectorizer: The fitted TF-IDF vectorizer.
        sparse matrix: The TF-IDF matrix for the combined text.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2), use_idf=True
    )
    tfidf_matrix = vectorizer.fit_transform(movies["combined_text"])
    return vectorizer, tfidf_matrix


def recommend_movies(query, movies, vectorizer, tfidf_matrix, top_n=5):
    """
    Recommends movies based on the input query using cosine similarity.

    Args:
        query (str): The user's movie description for recommendations.
        movies (DataFrame): The preprocessed movie dataset.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): The TF-IDF matrix for the combined text.
        top_n (int, optional): The number of top recommendations to return. Defaults to 5.

    Returns:
        str: The formatted string of top movie recommendations.
    """
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Apply a threshold to avoid weak matches
    threshold = np.percentile(similarity_scores, 80)  # Top 20% only
    top_indices = similarity_scores.argsort()[::-1]
    filtered_indices = [
        idx for idx in top_indices if similarity_scores[idx] >= threshold]

    if not filtered_indices:
        return "❌ No strong recommendations found. Try a different description."

    # Get top-N recommendations, ensuring uniqueness
    recommended_titles = set()
    final_recommendations = []
    for idx in filtered_indices:
        title = movies.iloc[idx]["title"]
        if title not in recommended_titles:
            final_recommendations.append((idx, similarity_scores[idx]))
            recommended_titles.add(title)
        if len(final_recommendations) >= top_n:
            break

    # Generate output
    output = "\n📽️ **Top Movie Recommendations:**\n"
    for idx, similarity in final_recommendations:
        row = movies.iloc[idx]
        output += (
            f"\n🎬 **Title:** {row['title']}\n"
            f"⭐ **Similarity Score:** {similarity:.2f}\n"
            f"🎭 **Genres:** {row['genres']}\n"
            f"📖 **Description:** {row['overview']}\n"
            f"{'-'*80}\n"
        )
    return output


def main():
    """
    Main function to parse arguments, preprocess data, train the TF-IDF vectorizer, and print movie recommendations.
    """
    parser = argparse.ArgumentParser(
        description="Content-Based Movie Recommendation System")
    parser.add_argument("query", type=str,
                        help="User's movie description for recommendations")
    args = parser.parse_args()

    # Load and preprocess data
    movies = preprocess_data("movies_sample.csv")
    vectorizer, tfidf_matrix = train_tfidf(movies)

    # Get recommendations
    recommendations = recommend_movies(
        args.query, movies, vectorizer, tfidf_matrix)

    # Print results
    print(recommendations)


if __name__ == "__main__":
    main()
