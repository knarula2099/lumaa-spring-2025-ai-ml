# Content-Based Movie Recommendation System  

This repository contains a simple content-based movie recommendation system using **TF-IDF** and **cosine similarity**. Given a short text query describing a movie, the system returns the most relevant movies from the dataset.  

## Features  
- Processes movie metadata, including genres, keywords, and overviews.  
- Uses **TF-IDF vectorization** to convert text into numerical form.  
- Computes **cosine similarity** between the user query and each movie.  
- Returns the top matching movies based on similarity scores.  

## Installation  

### 1. Clone the Repository  
```bash
git clone [YOUR_FORKED_REPO_URL]
cd [PROJECT_DIRECTORY]
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Run the Recommendation System  
```bash
python movie_recommender.py "A sci-fi adventure about space exploration and survival."
```
Modify the query to test different recommendations.  

## Dataset  

The dataset should contain the following fields:  
- `title`: Movie title  
- `overview`: Short description of the movie  
- `genres`: Movie genres  
- `keywords`: Key themes related to the movie  

Dataset file: **movies_sample.csv** (replace with actual file name if different).  

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

## Testing Effectiveness  

To verify the system’s performance, try running:  
```bash
python movie_recommender.py "A sci-fi adventure about space exploration and survival."
```
Expected output should include relevant science-fiction movies.  

If the results are not as expected, adjustments to **genre weighting** or **TF-IDF preprocessing** may be needed.  

## Deliverables  

- **Fork the Public Repository**  
  - This repository must be forked into your GitHub account.  

- **Implement Your Solution**  
  - Load and preprocess the dataset.  
  - Convert text data into vectors using **TF-IDF**.  
  - Compute similarity between the query and movie descriptions.  
  - Return the top matches.  

- **Salary Expectation**  
  - Expected salary per month: **[PLACEHOLDER]**  

- **Short Video Demo**  
  - A brief video demonstration should be provided.  
  - Create a file `demo.md` and paste a link to the screen recording showing:  
    - Running the recommendation code.  
    - A sample query and results.  

- **Submission Deadline**  
  - Submit the forked repository by **Sunday, Feb 23rd, 11:59 PM PST**.  

## Evaluation Criteria  

1. **Functionality**  
   - The code should run without errors.  
   - It should correctly return relevant movie recommendations.  

2. **Code Quality**  
   - Logical and well-structured implementation.  
   - Clear comments where necessary.  

3. **Clarity**  
   - The README should clearly explain setup, execution, and expected results.  

4. **Understanding of Content-Based Recommendation Systems**  
   - Proper use of **TF-IDF** and **cosine similarity** for recommendations.  

## License  
[Specify License Here]  

## Contact  
For any questions, reach out at:  
[Your Email or GitHub Profile]  