# flask_app.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_message():
    """Process incoming messages from Slack"""
    data = request.json
    message = data.get("text", "")
    
    # Add your custom processing logic here
    processed_result = process_message(message)
    
    return jsonify({"response": processed_result})

def process_message(message):
    """
    Your custom processing logic goes here.
    This is where you put your existing processing code.
    """
    # Example processing - replace with your actual logic
    return f"Processed: {message}"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
