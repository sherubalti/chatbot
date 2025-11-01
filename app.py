from flask import Flask, render_template, request, jsonify
import os
import csv
import datetime
import logging
from chatbot import get_chatbot, is_chatbot_ready
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CSV file configuration - Only feedback data
FEEDBACK_FILE = "feedback_data.csv"
FEEDBACK_HEADERS = ["timestamp", "question", "answer", "like", "dislike", "comments"]

def ensure_files_exist():
    """Create feedback CSV file with headers if it doesn't exist"""
    try:
        if not os.path.exists(FEEDBACK_FILE):
            logger.info(f"📁 Creating feedback file: {FEEDBACK_FILE}")
            with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(FEEDBACK_HEADERS)
            logger.info(f"✅ Successfully created {FEEDBACK_FILE} with headers")
        else:
            logger.info(f"✅ Feedback file already exists: {FEEDBACK_FILE}")
    except Exception as e:
        logger.error(f"❌ Error creating feedback file: {e}")
        # You might want to raise the exception or handle it differently
        # depending on your requirements
        raise

def save_feedback_to_csv(question, answer, like=False, dislike=False, comments=""):
    """Save feedback data to CSV file"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, question, answer, int(like), int(dislike), comments])
        logger.info(f"💾 Feedback saved: {like=}, {dislike=}, comments_length={len(comments)}")
    except Exception as e:
        logger.error(f"Error saving feedback to CSV: {e}")

def get_feedback_stats():
    """Get statistics for likes, dislikes, and comments"""
    try:
        likes = 0
        dislikes = 0
        comments_count = 0
        
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    likes += int(row.get('like', 0))
                    dislikes += int(row.get('dislike', 0))
                    if row.get('comments', '').strip():
                        comments_count += 1
        
        return {
            'total_likes': likes,
            'total_dislikes': dislikes,
            'total_comments': comments_count,
            'total_feedback': likes + dislikes + comments_count
        }
    except Exception as e:
        logger.error(f"Error reading feedback stats: {e}")
        return {'total_likes': 0, 'total_dislikes': 0, 'total_comments': 0, 'total_feedback': 0}

def get_feedback_timeline():
    """Get feedback data for timeline chart"""
    try:
        from collections import defaultdict
        timeline_data = defaultdict(lambda: {'likes': 0, 'dislikes': 0, 'comments': 0})
        
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    date = row['timestamp'].split()[0]  # Get only the date part
                    timeline_data[date]['likes'] += int(row.get('like', 0))
                    timeline_data[date]['dislikes'] += int(row.get('dislike', 0))
                    if row.get('comments', '').strip():
                        timeline_data[date]['comments'] += 1
        
        # Convert to list sorted by date
        sorted_dates = sorted(timeline_data.keys())
        return [
            {
                'date': date,
                'likes': timeline_data[date]['likes'],
                'dislikes': timeline_data[date]['dislikes'],
                'comments': timeline_data[date]['comments']
            }
            for date in sorted_dates
        ]
    except Exception as e:
        logger.error(f"Error reading feedback timeline: {e}")
        return []

# Initialize files when module loads
ensure_files_exist()

@app.route('/')
def home():
    chatbot_status = "ready" if is_chatbot_ready() else "initializing"
    return render_template('index.html', chatbot_status=chatbot_status)

@app.route('/status')
def status():
    """Check chatbot status"""
    chatbot = get_chatbot()
    if chatbot and is_chatbot_ready():
        stats = chatbot.get_statistics()
        return jsonify({
            'status': 'ready',
            'message': f'Chatbot ready with {stats["dataset_size"]} Q&A pairs',
            'stats': stats
        })
    else:
        return jsonify({
            'status': 'initializing',
            'message': 'Chatbot is still loading. Please wait...'
        })

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    if not is_chatbot_ready():
        return jsonify({
            'response': {
                'response': 'Chatbot is still initializing. Please try again in a few moments.',
                'response_time': '0ms',
                'confidence': '0.00',
                'cache_hit': False
            }
        })
    
    try:
        user_message = request.json['message']
        chatbot = get_chatbot()
        
        if chatbot is None:
            return jsonify({
                'response': {
                    'response': 'Chatbot not available. Please check server logs.',
                    'response_time': '0ms', 
                    'confidence': '0.00',
                    'cache_hit': False
                }
            })
        
        response_data = chatbot.generate_response(user_message)
        
        # No longer saving chat history to CSV - only feedback data is saved
        
        return jsonify({'response': response_data})
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return jsonify({
            'response': {
                'response': 'Error processing your request. Please try again.',
                'response_time': '0ms',
                'confidence': '0.00',
                'cache_hit': False
            }
        })

@app.route('/feedback', methods=['POST'])
def save_feedback():
    """Save user feedback for a response"""
    try:
        data = request.json
        question = data.get('question', '')
        answer = data.get('answer', '')
        like = data.get('like', False)
        dislike = data.get('dislike', False)
        comments = data.get('comments', '')
        
        save_feedback_to_csv(question, answer, like, dislike, comments)
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback saved successfully',
            'stats': get_feedback_stats()
        })
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/feedback/stats')
def get_feedback_statistics():
    """Get feedback statistics for charts"""
    try:
        return jsonify({
            'stats': get_feedback_stats(),
            'timeline': get_feedback_timeline()
        })
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        return jsonify({'stats': {}, 'timeline': []})

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard to view feedback statistics"""
    return render_template('admin.html')

if __name__ == '__main__':
    # Wait a bit for chatbot to initialize
    logger.info("🚀 Starting Flask server...")
    
    # Check chatbot status
    max_wait = 30  # Maximum wait time in seconds
    wait_time = 0
    
    while not is_chatbot_ready() and wait_time < max_wait:
        logger.info(f"⏳ Waiting for chatbot to initialize... ({wait_time}s)")
        time.sleep(2)
        wait_time += 2
    
    if is_chatbot_ready():
        logger.info("✅ Chatbot is ready! Starting Flask server on http://0.0.0.0:5000")
    else:
        logger.warning("⚠️ Chatbot initialization taking longer than expected. Starting server anyway...")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)