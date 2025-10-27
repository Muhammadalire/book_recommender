import pandas as pd
import numpy as np

def load_and_prepare_data():
    """
    Load the book dataset and prepare it for recommendation system.
    Returns: pandas DataFrame with cleaned book data
    """
    
    try:
        # Load the CSV file
        df = pd.read_csv('data/indo_books.csv')
        print(f"✅ Dataset loaded successfully with {len(df)} books")
        
    except FileNotFoundError:
        print("❌ Error: books.csv not found in data/ folder")
        print("Please make sure your CSV file is in the backend/data/ directory")
        return None
    
    # Display actual column names from your CSV
    print(f"📊 Actual columns in your dataset: {list(df.columns)}")
    print(f"📖 First few books:")
    print(df.head(3))
    
    # Map your Indonesian column names to English for easier coding
    column_mapping = {
        'æJudul (Title)': 'title',
        'Penulis (Author)': 'authors', 
        'Genre': 'genres',
        'Rating (dari 5)': 'rating',
        'Summary': 'description'
    }
    
    # Rename columns to English
    df = df.rename(columns=column_mapping)
    print(f"🔄 Renamed columns to: {list(df.columns)}")
    
    # Data Cleaning Steps
    
    # 1. Handle missing values in critical columns
    # Fill missing authors with 'Unknown'
    if 'authors' in df.columns:
        df['authors'] = df['authors'].fillna('Unknown Author')
    
    # Fill missing genres with 'General'
    if 'genres' in df.columns:
        df['genres'] = df['genres'].fillna('General')
    
    # Fill missing titles (if any)
    if 'title' in df.columns:
        df = df.dropna(subset=['title'])  # Remove books without titles
        df['title'] = df['title'].fillna('Unknown Title')
    
    # Fill missing descriptions
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('')
    
    # 2. Create a combined feature for recommendation
    # This combines genres and authors into one text field for better recommendations
    df['combined_features'] = ''
    
    # Combine genres and authors for the recommendation algorithm
    if 'genres' in df.columns and 'authors' in df.columns:
        df['combined_features'] = df['genres'] + ' ' + df['authors']
    elif 'genres' in df.columns:
        df['combined_features'] = df['genres']
    elif 'authors' in df.columns:
        df['combined_features'] = df['authors']
    
    # 3. Add description to features if it exists (makes recommendations more accurate)
    if 'description' in df.columns:
        df['combined_features'] = df['combined_features'] + ' ' + df['description']
    
    print(f"✅ Data preparation complete!")
    print(f"📚 Sample of prepared data:")
    print(df[['title', 'authors', 'genres', 'combined_features']].head(2))
    
    # Show some statistics
    print(f"\n📊 Dataset Summary:")
    print(f"Total books: {len(df)}")
    print(f"Unique genres: {df['genres'].nunique()}")
    print(f"Unique authors: {df['authors'].nunique()}")
    
    return df

# Test the function
if __name__ == "__main__":
    df = load_and_prepare_data()
    if df is not None:
        print("\n🎉 Data module is working correctly!")
        print("You can now run: python3 recommender.py")