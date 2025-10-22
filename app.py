import streamlit as st
import numpy # Kept for dependency awareness
import random
import pandas as pd
import time
import asyncio
import plotly.express as px
from twikit import Client, Tweet
from twikit.errors import TooManyRequests, NotFound # Import necessary exceptions
from transformers import pipeline

# --- Configuration and Initialization ---

# NOTE: For Streamlit, it's safer to use st.secrets or st.experimental_get_query_params
USERNAME = "howsitze"
PASSWORD = "DOMINATE123@"

@st.cache_resource
def load_roberta_model():
    """Loads the RoBERTa sentiment analysis pipeline."""
    try:
        st.info("Loading RoBERTa model...")
        model_name = "sahri/indonesiasentiment"
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )
        st.success("RoBERTa model loaded successfully!")
        return sentiment_pipeline
    except Exception as e:
        # Catches common errors like ModuleNotFoundError (e.g., from failed tokenizers install)
        st.error(f"FATAL: Failed to load RoBERTa model or its dependencies. Error: {e}")
        return None

@st.cache_resource(ttl=3600)
def init_twikit_client():
    """Initializes and logs into the Twikit client synchronously."""
    client = Client(language='id-ID')
    try:
        # Load saved session cookies if available
        client.load_cookies('cookies.json')
        st.success("Twikit client session loaded successfully.")
    except Exception:
        st.warning("No existing session found or session expired. Logging in...")
        try:
            # Twikit's login method is synchronous/blocking in this context, or we'll wrap it later.
            # Using synchronous login is generally safer for Streamlit initialization.
            # NOTE: Cloudflare/403 errors are common here.
            client.login(
                authInfo=Client.AuthInfo(username=USERNAME, password=PASSWORD)
            )
            client.save_cookies('cookies.json')
            st.success("Twikit client logged in and session saved.")
        except Exception as e:
            st.error(f"FATAL: Twikit Login Failed: {e}. Check credentials or IP/Cloudflare block.")
            return None
    return client

# --- Async Core Function (Wrapped in a single synchronous call) ---

async def async_fetch_tweets_with_limit(query, limit, client):
    """
    ASYNCHRONOUSLY fetches tweets with error and rate limit handling.
    (This is the user-provided function logic with the 'random' import fixed)
    """
    tweets_data = []
    tweet_count = 0
    tweets_result = None
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting async fetch for '{query}'...")

    while tweet_count < limit:
        try:
            if tweets_result is None:
                # First request
                tweets_result = await client.search_tweet(query, 'Latest')
            else:
                # Subsequent pagination
                tweets_result = await tweets_result.next()

            if not tweets_result: # Check if pagination result is empty
                print(f"[{time.strftime('%H:%M:%S')}] End of search results reached.")
                break

            # 1. Data Structuring
            for tweet in tweets_result:
                if tweet_count >= limit:
                    break
                
                # Check for Tweet object and handle data
                if isinstance(tweet, Tweet): 
                    tweets_data.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'user': tweet.user.screen_name,
                        'date': tweet.created_at,
                    })
                    tweet_count += 1
            
            print(f"[{time.strftime('%H:%M:%S')}] Successfully fetched {len(tweets_result)} tweets. Total: {tweet_count}")

            # Politeness delay (using random.randint as requested)
            wait_time = random.randint(5, 20) # Random delay between 5 and 30 seconds
            print(f"[{time.strftime('%H:%M:%S')}] Successful request. Waiting for {wait_time} seconds...")
            await asyncio.sleep(wait_time)


        except TooManyRequests as e:
            # Rate limit handling (based on previous logic)
            wait_time = 900 # Default 15 min
            if hasattr(e, 'headers') and 'x-rate-limit-reset' in e.headers:
                reset_time = int(e.headers['x-rate-limit-reset'])
                wait_time = max(60, reset_time - int(time.time()) + 5) # Minimum 60s
            
            print(f"[{time.strftime('%H:%M:%S')}] Rate limit exceeded. Waiting for {wait_time} seconds.")
            await asyncio.sleep(wait_time)
            continue

        except NotFound:
            print(f"[{time.strftime('%H:%M:%S')}] twikit.errors.NotFound: End of search results reached.")
            break

        except Exception as e:
            error_message = str(e)
            if "ForbiddenStatus: 403" in error_message or "Cloudflare" in error_message:
                print(f"[{time.strftime('%H:%M:%S')}] Forbidden (403): Account/IP blocked. Stopping fetch. Error: {error_message}")
                break
            
            print(f"[{time.strftime('%H:%M:%S')}] An unexpected error occurred: {error_message}")
            await asyncio.sleep(10) 
            break

    return pd.DataFrame(tweets_data)


def fetch_and_analyze_data_sync(query, limit, client, model, tokenizer):
    """
    SYNCHRONOUS wrapper to run the async fetch and the analysis.
    FIXED: Uses robust asyncio loop management to prevent 'Event loop is closed'.
    """
    if not client:
        return pd.DataFrame() 

    st.info(f"Fetching up to {limit} tweets for '{query}'...")
    
    # --- Robust Asyncio Execution FIX ---
    try:
        # Get the current running loop, or create a new one if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # If the loop is closed, create a fresh one
        if loop.is_closed():
            st.warning("Detected closed event loop. Creating a fresh loop for execution.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(async_fetch_tweets_with_limit(query, limit, client))
        else:
            # If the loop is open (or running), use asyncio.run()
            df = asyncio.run(async_fetch_tweets_with_limit(query, limit, client))

    except RuntimeError as e:
        if "cannot run the event loop" in str(e) or "Event loop is closed" in str(e):
            # Final fallback: if standard asyncio.run fails, try run_until_complete on a new loop
            st.error("Asyncio error in the main thread. Attempting forced loop reset.")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                df = loop.run_until_complete(async_fetch_tweets_with_limit(query, limit, client))
            except Exception as inner_e:
                st.error(f"FATAL: Asyncio execution failed even after reset. Error: {inner_e}")
                return pd.DataFrame()
        else:
            raise e


def fetch_and_analyze_data_sync(query, limit, client, sentiment_pipeline):
    """SYNCHRONOUS wrapper to run the async fetch and the analysis."""
    if not client:
        return pd.DataFrame() 

    st.info(f"Fetching up to {limit} tweets for '{query}'...")
    
    # CRITICAL FIX: Use asyncio.run() to execute the async function
    # This must be done here, outside of the global loop.
    try:
        df = asyncio.run(async_fetch_tweets_with_limit(query, limit, client))
    except RuntimeError as e:
        # Catches the error when asyncio is already running (e.g., in a complex environment)
        if "cannot run the event loop" in str(e):
             st.error("Asyncio Runtime Error: The event loop is already running. Please restart Streamlit.")
             return pd.DataFrame()
        raise e
    
    if df.empty:
        st.warning(f"No data found for '{query}'.")
        return df

    # 2. Sentiment Analysis (RoBERTa)
    st.info("Analyzing sentiment with RoBERTa...")
    
    # Apply the sentiment pipeline to the 'text' column
    sentiments = sentiment_pipeline(df['text'].tolist())
    
    # Extract the label and score
    df['sentiment'] = [s['label'] for s in sentiments]
    df['score'] = [s['score'] for s in sentiments]

    return df


# --- Streamlit App Layout ---

def main():
    st.title("🐦 Real-Time RoBERTa Sentiment Analyzer")
    st.markdown("Analyze the sentiment of live data using Twikit, RoBERTa, and Streamlit.")

    # Sidebar for Model and Data Settings
    st.sidebar.header("Configuration")
    query = st.sidebar.text_input("Search Term (e.g., #streamlit)")
    # Setting limit to a reasonable max based on rate limits
    tweet_limit = st.sidebar.slider("Max Tweets to Analyze (per refresh)", 10, 1000, 50) 
    #refresh_rate = st.sidebar.number_input("Refresh Interval (seconds)", 60, 30000, 60)
    
    # Load model and client
    sentiment_pipeline = load_roberta_model()
    twikit_client = init_twikit_client()

    # CRITICAL CHECK: Stop if either failed to load
    if sentiment_pipeline is None:
        st.error("Model loading failed. Please resolve the installation errors (Numpy, tokenizers, PIL).")
        st.stop()
    if twikit_client is None:
        st.error("Twikit client initialization failed. Check credentials or network block (403/Cloudflare).")
        st.stop()
    
    st.sidebar.success("Ready to analyze!")
    
    # Main content area for real-time updates
    placeholder = st.empty()
    
    if st.sidebar.button("Start", type="primary"):
        with placeholder.container():
            st.header(f"Results for '{query}'")
            st.caption(f"Last updated: {pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')}")

            # Call the synchronous wrapper
            df_results = fetch_and_analyze_data_sync(query, tweet_limit, twikit_client, sentiment_pipeline)

            if not df_results.empty:
                
                # 3. Display and Visualization
                sentiment_counts = df_results['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                st.subheader("Sentiment Distribution")
                fig = px.pie(
                sentiment_counts,  # Use the DataFrame that contains the counts
                names='Sentiment', # Column for the labels (e.g., 'Positive', 'Negative')
                values='Count',    # Column for the values (the frequency)
                title='Tweet Sentiment Distribution',
                color='Sentiment'  # Optional: use sentiment names for consistent colors
                )
    
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Latest Analyzed Data (DataFrame)")
                st.dataframe(df_results[['date', 'text', 'sentiment', 'score']], use_container_width=True)

            else:
                st.warning("No data to display. Check your query or Twikit status (rate limit/block).")
        
        # Wait for the specified refresh rate
        #time.sleep(refresh_rate)

if __name__ == "__main__":
    main()