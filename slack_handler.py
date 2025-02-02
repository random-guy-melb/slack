from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from dotenv import load_dotenv
import requests
import tempfile
import plotly.express as px
import pandas as pd
import numpy as np
import re

# Load environment variables
load_dotenv()

# Initialize the app
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Configure Flask app URL
FLASK_APP_URL = "http://localhost:5000/process"


def process_with_flask(text):
    """Helper function to send text to Flask app and get response"""
    try:
        response = requests.post(
            FLASK_APP_URL,
            json={"text": text}
        )
        if response.status_code == 200:
            return response.json().get("response", "No response from processing")
        else:
            print(f"Flask app error: Status {response.status_code}")
            return "Sorry, there was an error processing your message."
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Flask app: {e}")
        return "Sorry, I couldn't reach the processing service."


def generate_plotly_chart(response_text):
    """
    Generate a sample Plotly chart based on the response text.
    """
    df = pd.DataFrame({
        "x": np.arange(10),
        "y": np.random.randint(0, 10, 10)
    })
    fig = px.line(df, x="x", y="y", title="Sample Chart")

    # Save the figure to a temporary PNG file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.write_image(temp_file.name)
    return temp_file.name


def render_chart(text, event, thread_ts=None):
    """
    Upload a chart to Slack. For @mentions, use thread_ts to keep in thread.
    For DMs, thread_ts will be None for normal chat flow.
    """
    chart_path = generate_plotly_chart(text)
    channel_id = event.get("channel")
    try:
        upload_params = {
            "channels": channel_id,
            "file": chart_path,
            "title": "Interactive Plotly Chart",
            "initial_comment": (
                "Here's your Plotly chart (as a static image). "
                "For full interactivity, please visit our web app."
            )
        }
        
        # Only add thread_ts for @mentions
        if thread_ts:
            upload_params["thread_ts"] = thread_ts
            
        app.client.files_upload_v2(**upload_params)
    except Exception as e:
        print(f"Error uploading file: {e}")
    finally:
        os.remove(chart_path)


@app.event("message")
def handle_message(message, say, client, event):
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
            text = re.sub(f"<@{bot_user_id}>", "", text).strip()
            thread_ts = message.get("ts")
            response = process_with_flask(text)
            say(text=response, thread_ts=thread_ts)
            render_chart(text, event, thread_ts)
        else:
            # For DMs, just respond normally without threading
            response = process_with_flask(text)
            say(response)
            render_chart(text, event)


@app.event("app_mention")
def handle_mention(event, say):
    text = event.get("text", "").split(">")[1].strip()
    if not text:
        say("How can I help you?")
        return
    
    response = process_with_flask(text)
    thread_ts = event.get("thread_ts", None) or event.get("ts", None)
    
    # Always use threading for @mentions
    say(text=response, thread_ts=thread_ts)
    render_chart(text, event, thread_ts)


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
