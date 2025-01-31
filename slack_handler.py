# slack_handler.py
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from dotenv import load_dotenv
import requests
import re

# Load environment variables
load_dotenv()

# Initialize the app with bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Configure Flask app URL
FLASK_APP_URL = "http://localhost:5000/process"

@app.message("")
def handle_message(message, say, client):
    """Handle messages that mention the bot or are sent via DM"""
    # Ignore messages from bots
    if message.get("bot_id"):
        return
        
    text = message.get("text", "")
    channel_type = message.get("channel_type")
    bot_user_id = client.auth_test()["user_id"]
    
    # Check if this is a DM or if the bot was mentioned
    is_dm = channel_type == "im"
    is_mentioned = f"<@{bot_user_id}>" in text
    
    if is_dm or is_mentioned:
        # For mentions, remove the bot's user ID from the text
        if is_mentioned:
            # Remove the bot mention from the text
            text = re.sub(f"<@{bot_user_id}>", "", text).strip()
        
        try:
            # Forward message to Flask app
            response = requests.post(
                FLASK_APP_URL,
                json={"text": text}
            )
            
            if response.status_code == 200:
                processed_message = response.json().get("response", "No response from processing")
                if is_mentioned:
                    # If in a channel, reply in a thread
                    say(processed_message, thread_ts=message.get("ts"))
                else:
                    # If in DM, reply directly
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
