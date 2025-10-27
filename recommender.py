import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BookRecommender:
    """
    A content-based book recommendation system using TF-IDF and cosine similarity.
    """
    
    def __init__(self, book_data):
        """
        Initialize the recommender with book data.
        
        Args:
            book_data: pandas DataFrame with book information
        """
        self.df = book_data
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        
        # Build the recommendation model
        self._build_model()
    
    def _build_model(self):
        """Build the TF-IDF matrix and compute cosine similarity."""
        print("🔄 Building recommendation model...")
        
        # Initialize TF-IDF Vectorizer
        # TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numbers
        # max_features limits the number of words considered (for performance)
        # stop_words removes common words like 'the', 'and' etc.
        tfidf = TfidfVectorizer(
            max_features=5000,      # Consider top 5000 words
            stop_words='english',   # Remove common English words
            ngram_range=(1, 2)      # Consider single words and word pairs
        )
        
        # Transform book features into TF-IDF matrix
        # Each book becomes a vector of numbers representing its content
        self.tfidf_matrix = tfidf.fit_transform(self.df['combined_features'])
        
        print(f"✅ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Compute cosine similarity matrix
        # Cosine similarity measures how similar two books are based on their vectors
        # Result is a matrix where [i,j] shows similarity between book i and book j
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print(f"✅ Cosine similarity matrix shape: {self.cosine_sim.shape}")
        
        # Create a mapping from book title to index
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        print(f"✅ Index mapping created for {len(self.indices)} books")
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get book recommendations based on a given book title.
        
        Args:
            title (str): Title of the book to get recommendations for
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended book titles with similarity scores
        """
        print(f"🔍 Getting recommendations for: '{title}'")
        
        # Check if book exists in our dataset
        if title not in self.indices:
            # Try fuzzy matching - find similar titles
            try:
                matching_books = self.df[self.df['title'].str.contains(title, case=False, na=False)]
                if len(matching_books) == 0:
                    # Try partial match
                    matching_books = self.df[
                        self.df['title'].apply(
                            lambda x: str(x).lower() if pd.notna(x) else ''
                        ).str.contains(title.lower(), na=False)
                    ]
            except Exception as e:
                print(f"⚠️ Error in title matching: {e}")
                matching_books = pd.DataFrame()
            
            if len(matching_books) == 0:
                return f"❌ Book '{title}' not found in database. Please check the spelling or try a different book."
            else:
                # If multiple matches, use the first one
                title = matching_books.iloc[0]['title']
                print(f"🔍 Using closest match: '{title}'")
        
        # Get the index of the book
        idx = self.indices[title]
        
        # Get similarity scores for all books with this book
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort books based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get indexes of the most similar books (skip the first one as it's the book itself)
        sim_scores = sim_scores[1:num_recommendations+1]
        
        # Get the book indices and similarity scores
        book_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Return the top most similar books with their details
        recommendations = []
        for i, (book_idx, score) in enumerate(zip(book_indices, similarity_scores)):
            book_info = {
                'rank': i + 1,
                'title': self.df.iloc[book_idx]['title'],
                'authors': self.df.iloc[book_idx].get('authors', 'Unknown'),
                'genres': self.df.iloc[book_idx].get('genres', 'Unknown'),
                'description': self.df.iloc[book_idx].get('description', 'No description available.'),
                'similarity_score': round(score, 3)
            }
            recommendations.append(book_info)
        
        return recommendations
    
    def search_books(self, query, max_results=10):
        """
        Search for books by title or author.
        
        Args:
            query (str): Search term
            max_results (int): Maximum number of results to return
            
        Returns:
            list: Matching books
        """
        # Search in titles (case-insensitive)
        title_matches = self.df[
            self.df['title'].str.contains(query, case=False, na=False)
        ]
        
        # Search in authors (case-insensitive)
        author_matches = self.df[
            self.df['authors'].str.contains(query, case=False, na=False)
        ]
        
        # Combine results and remove duplicates
        results = pd.concat([title_matches, author_matches]).drop_duplicates()
        
        if len(results) > max_results:
            results = results.head(max_results)
        
        return results[['title', 'authors', 'genres', 'description']].to_dict('records')
    
    def get_random_books(self, count=5):
        """
        Get random books from the dataset.
        
        Args:
            count (int): Number of random books to return
            
        Returns:
            list: Random books
        """
        random_books = self.df.sample(n=min(count, len(self.df)))
        return random_books[['title', 'authors', 'genres', 'description']].to_dict('records')

# Test the recommender
if __name__ == "__main__":
    from book_data import load_and_prepare_data
    
    print("🧪 Testing Book Recommender...")
    df = load_and_prepare_data()
    if df is not None:
        recommender = BookRecommender(df)
        
        # Show some random books to test with
        print(f"\n🎲 Here are some books in your database to test with:")
        random_books = recommender.get_random_books(3)
        for i, book in enumerate(random_books):
            print(f"   {i+1}. {book['title']} by {book['authors']}")
        
        # Test with the first random book
        if random_books:
            test_book = random_books[0]['title']
            print(f"\n📚 Testing recommendations for: {test_book}")
            recommendations = recommender.get_recommendations(test_book, 5)
            
            if isinstance(recommendations, str):
                print(f"❌ {recommendations}")
            else:
                print("✅ Recommendations found:")
                for rec in recommendations:
                    print(f"   {rec['rank']}. {rec['title']} (Score: {rec['similarity_score']})")
        
        # Test search functionality
        print(f"\n🔍 Testing search functionality...")
        search_results = recommender.search_books("book", 3)
        print(f"Search results for 'book':")
        for i, result in enumerate(search_results):
            print(f"   {i+1}. {result['title']} by {result['authors']}")