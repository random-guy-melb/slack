# slack_handler.py
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Initialize the app with bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Configure Flask app URL
FLASK_APP_URL = "http://localhost:5000/process"  # Adjust port if needed

@app.message("")
def handle_message(message, say):
    """Handle any messages and forward to Flask app"""
    # Ignore messages from bots to prevent loops
    if message.get("bot_id"):
        return
        
    try:
        # Forward message to Flask app
        response = requests.post(
            FLASK_APP_URL,
            json={"text": message.get("text", "")}
        )
        
        if response.status_code == 200:
            # Send processed result back to Slack
            processed_message = response.json().get("response", "No response from processing")
            say(processed_message)
        else:
            say("Sorry, there was an error processing your message.")
            print(f"Flask app error: Status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        say("Sorry, I couldn't reach the processing service.")
        print(f"Error connecting to Flask app: {e}")

@app.error
def handle_errors(error):
    """Handle any errors that occur"""
    print(f"Slack error: {error}")

if __name__ == "__main__":
    handler = SocketModeHandler(
        app, 
        os.environ.get("SLACK_APP_TOKEN")
    )
    print("⚡️ Slack handler is running!")
    handler.start()
