from flask import Flask, request, jsonify
from flask_cors import CORS
from book_data import load_and_prepare_data
from recommender import BookRecommender
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for API routes

# Global variables
recommender = None
df = None

def initialize_recommender():
    """Initialize the recommender system with book data."""
    global recommender, df
    
    print("🚀 Initializing Book Recommender System...")
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        print("❌ Failed to load data. Exiting...")
        return False
    
    # Initialize recommender
    recommender = BookRecommender(df)
    print("✅ Recommender system ready!")
    return True

# API Routes

@app.route('/')
def home():
    """Home route - provides API information."""
    return jsonify({
        "message": "Book Recommendation API",
        "endpoints": {
            "/api/search?q=query": "Search for books",
            "/api/recommend": "Get recommendations (POST with JSON)",
            "/api/random": "Get random books"
        },
        "total_books": len(df) if df is not None else 0
    })

@app.route('/api/search')
def search_books():
    """Search for books by title or author."""
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({"error": "Please provide a search query with ?q=book_name"}), 400
    
    if recommender is None:
        return jsonify({"error": "Recommender not initialized"}), 500
    
    try:
        results = recommender.search_books(query, max_results=10)
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get book recommendations for a given book."""
    if recommender is None:
        return jsonify({"error": "Recommender not initialized"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'book_title' not in data:
            return jsonify({"error": "Please provide 'book_title' in JSON body"}), 400
        
        book_title = data['book_title']
        num_recommendations = data.get('num_recommendations', 10)
        
        # Get recommendations
        recommendations = recommender.get_recommendations(book_title, num_recommendations)
        
        # Check if it's an error message
        if isinstance(recommendations, str) and recommendations.startswith("❌"):
            return jsonify({"error": recommendations}), 404
        
        return jsonify({
            "input_book": book_title,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
        
    except Exception as e:
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500

@app.route('/api/random')
def get_random_books():
    """Get random books from the dataset."""
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    try:
        num_books = min(int(request.args.get('count', 5)), 20)  # Max 20 books
        random_books = df.sample(n=num_books)[['title', 'authors', 'genres']]
        return jsonify({
            "books": random_books.to_dict('records')
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get random books: {str(e)}"}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    status = "healthy" if recommender is not None else "unhealthy"
    return jsonify({
        "status": status,
        "total_books": len(df) if df is not None else 0
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Main execution
if __name__ == '__main__':
    print("📚 Starting Book Recommendation Server...")
    
    if initialize_recommender():
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),  # change here
            debug=False,  # disable debug on deploy
            threaded=True
        )
    else:
        print("❌ Failed to start server due to data loading issues")
