# Content-Based Movie Recommendation System

This repository contains a simple content-based movie recommendation system using **TF-IDF** and **cosine similarity**. Given a short text query describing a movie, the system returns the most relevant movies from the dataset.

## Features

- Processes movie metadata, including genres, keywords, and overviews.
- Uses **TF-IDF vectorization** to convert text into numerical form.
- Computes **cosine similarity** between the user query and each movie.
- Returns the top matching movies based on similarity scores.

## Installation

### Prerequisites

- Python 3.6 or higher

### 1. Clone the Repository

```bash
git clone https://github.com/knarula2099/lumaa-spring-2025-ai-ml.git
cd lumaa-spring-2025-ai-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Recommendation System

```bash
python recommend.py "{Your Query Here}"
```

Modify the query to test different recommendations.

## Dataset

The dataset should contain the following fields:

- `id`: Unique identifier for the movie
- `title`: Movie title
- `overview`: Short description of the movie
- `genres`: Movie genres
- `keywords`: Key themes related to the movie

Dataset file: **movies_sample.csv**.

Dataset source: [Kaggle Movies Dataset](https://www.kaggle.com/datasets/abdallahwagih/movies)

All dataset filtering was done in the `preproccessing.ipynb` Jupyter Notebook.

## How It Works

1. **Data Preprocessing**

   - Reads the dataset from a CSV file.
   - Cleans and processes `genres`, `keywords`, and `overview`.
   - Enhances genre importance by repeating them in the text.

2. **TF-IDF Vectorization**

   - Converts text fields into a numerical representation.
   - Uses **sublinear TF scaling** and **L2 normalization** for better performance.

3. **Similarity Calculation**
   - Computes **cosine similarity** between the user's query and all movies.
   - Sorts results and returns the top matches.

## Video Demo

Video Link in `video_demo.md`

## Salary Expectations

Based on market trends and my experience, my salary expectation is approximately **$8,000** per month.

## Contact

For any questions, reach out at:

- Email: [karannarula@outlook.com](mailto:karannarula@outlook.com)
- LinkedIn: [https://www.linkedin.com/in/knarula03](https://www.linkedin.com/in/knarula03)
