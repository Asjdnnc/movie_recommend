# Movie Recommendation System

## Overview

This project implements a content-based movie recommendation system using machine learning techniques. The system recommends movies based on their similarity to a given movie using the **CountVectorizer** and **Cosine Similarity** from scikit-learn.

### Project Features:
- Recommends the top 10 most similar movies based on the input movie's tags.
- Uses **CountVectorizer** to transform movie tags into numerical feature vectors.
- Calculates the cosine similarity between movie vectors to measure their similarity.

---

## Prerequisites

Before running this project, ensure you have the following libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`

## How It Works
### Data Loading:

Load the data from movies.csv and credits.csv into pandas DataFrames and merge them on movie_id to combine movie details and cast information.
```python
movies = pd.read_csv('data/movies.csv')
credits = pd.read_csv('data/credits.csv')
df = movies.merge(credits, on='movie_id')
```

### Data Cleaning:
Select the relevant columns like movie_id,title,overview,genres,keywords,cast and crew for the recommendation system.
```python
df=df[["movie_id","title","overview","genres","keywords","cast","crew"]]
```
### Creating Tags
Combine the overview,genres,keywords,cast and crew to create a tags column that will be used for similarity calculation.
```python
df["tags"] = df["overview"]+df["genres"]+df["keywords"]+df["cast"]+df["crew"]
```
### Vectorization:

CountVectorizer from sklearn.feature_extraction.text is used to transform the text data (movie tags) into numerical vectors.
The CountVectorizer creates a matrix of token counts (i.e., a vector for each movie where each entry represents the frequency of a particular word or token in the movie's tags).
The parameter max_features=5000 ensures that the top 5000 most frequent words are considered, and stop_words="english" removes common English words like "and", "the", etc.
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()
```

### Cosine Similarity:

After vectorizing the movie tags, we calculate the Cosine Similarity between the movie vectors using cosine_similarity from sklearn.metrics.pairwise.
Cosine similarity measures the cosine of the angle between two vectors, with values close to 1 indicating high similarity.
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```

### Movie Recommendation:

The recommend(movie) function takes a movie title as input, calculates its similarity to other movies, and returns the top 10 most similar movies.
The function works by finding the index of the input movie in the DataFrame, then using the cosine similarity matrix to identify the most similar movies.
```python
def recommend(movie):
    movie_index = df[df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    
    for i in movie_list:
        print(df.iloc[i[0]].title)

```

## How to Use

1. Ensure your dataset is in the same directory or modify the path accordingly. The dataset should contain columns like movie_id, title, and tags.
2. Call the recommend(movie) function with the movie title of your choice to get recommendations
```python
recommend("Iron Man")
```

## Conclusion
This project demonstrates a simple and effective way to build a movie recommendation system using CountVectorizer and Cosine Similarity. It provides a foundation for building more complex recommendation systems using other techniques like collaborative filtering or deep learning.
