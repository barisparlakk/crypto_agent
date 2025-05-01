import nltk
from nltk.chat.util import Chat, reflections
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from dotenv import load_dotenv

app = Flask(__name__)

#Load environment variables from .env file
load_dotenv()
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')

#Download the necessary NLTK Data
nltk.download('punkt')

#Define pairs of patterns and static responses
static_pairs = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you?', ['I am doing well, thank you!', 'I am fine, how about you?']),
    (r'what is your name?', ['I am an AI chatbot. What is your name?']),
    (r'my name is (.*)', ['Nice to meet you, %1!']),
    (r'quit', ['Goodbye! Have a great day!']),
    (r'top gainers and losers', ['Let me fetch the top gainers and losers for you.']),
    (r'what are the top gainers and losers\??', ['Let me fetch the top gainers and losers for you.']),
    (r'show me the top gainers and losers', ['Let me fetch the top gainers and losers for you.']),
    (r'ohlc of (.*)', ['Let me fetch the OHLC data for %1.']),
    (r'(.*)', ['Sorry, I did not understand that. Can you please rephrase?'])
]

# Create a static chatbot using the defined pairs and reflections
static_chatbot = Chat(static_pairs, reflections)


# Function to get cryptocurrency price from CoinGecko API
def get_crypto_price(crypto_name):
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': crypto_name.lower(),
        'vs_currencies': 'usd'
    }
    headers = {
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if crypto_name.lower() in data:
            price = data[crypto_name.lower()]['usd']
            return f'The price of {crypto_name.capitalize()} in USD is ${price}'
        else:
            return f'Sorry, I could not find the price for {crypto_name}.'
    else:
        return 'Failed to retrieve data from the API.'


# Function to get cryptocurrency market cap from CoinGecko API
def get_crypto_market_cap(crypto_name):
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'ids': crypto_name.lower()
    }
    headers = {
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data:
            market_cap = data[0]['market_cap']
            return f'The market cap of {crypto_name.capitalize()} in USD is ${market_cap:,}'
        else:
            return f'Sorry, I could not find the market cap for {crypto_name}.'
    else:
        return 'Failed to retrieve data from the API.'


# Function to get cryptocurrency trading volume from CoinGecko API
def get_crypto_trading_volume(crypto_name):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_name.lower()}'
    headers = {
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'market_data' in data:
            volume = data['market_data']['total_volume']['usd']
            return f'The trading volume of {crypto_name.capitalize()} in USD is ${volume:,}'
        else:
            return f'Sorry, I could not find the trading volume for {crypto_name}.'
    else:
        return 'Failed to retrieve data from the API.'


# Function to recommend a cryptocurrency based on market cap and trading volume
def recommend_coin():
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 10,
        'page': 1
    }
    headers = {
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data:
            recommended_coin = data[0]['id']
            return f'I recommend considering {recommended_coin.capitalize()} as it has the highest market cap and trading volume.'
        else:
            return 'Sorry, I could not find any recommendations at this moment.'
    else:
        return 'Failed to retrieve data from the API.'


# Function to predict cryptocurrency price using a simple linear regression model
def predict_crypto_price(crypto_name):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_name.lower()}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': 30,  # Use the last 30 days of data
        'interval': 'daily'
    }
    headers = {
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'prices' in data:
            prices = data['prices']
            dates = np.array([price[0] for price in prices]).reshape(-1, 1)  # Dates in milliseconds
            prices = np.array([price[1] for price in prices]).reshape(-1, 1)  # Prices in USD

            # Train a simple linear regression model
            model = LinearRegression()
            model.fit(dates, prices)

            # Predict the price for the next day
            next_day = dates[-1] + (dates[-1] - dates[-2])
            predicted_price = model.predict([next_day])[0][0]

            return f'The predicted price of {crypto_name.capitalize()} for the next day is ${predicted_price:.2f}'
        else:
            return f'Sorry, I could not find enough data to make a prediction for {crypto_name}.'
    else:
        return 'Failed to retrieve data from the API.'


# Function to get top gainers and losers from CoinGecko API
def get_top_gainers_losers():
    url = 'https://pro-api.coingecko.com/api/v3/coins/top_gainers_losers'
    params = {
        'vs_currency': 'usd'  # Mandatory parameter
    }
    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'top_gainers' in data:
            gainers = "\n".join([
                f"{i + 1}. {coin['name']} (+{coin['usd_24h_change']:.2f}%)"
                for i, coin in enumerate(data['top_gainers'])
            ])
        else:
            gainers = "No data available for top gainers."

        if 'top_losers' in data:
            losers = "\n".join([
                f"{i + 1}. {coin['name']} ({coin['usd_24h_change']:.2f}%)"
                for i, coin in enumerate(data['top_losers'])
            ])
        else:
            losers = "No data available for top losers."

        return f"### Top Gainers:\n{gainers}\n\n### Top Losers:\n{losers}"
    elif response.status_code == 401:
        return 'Failed to retrieve data: Unauthorized. Please check your API key.'
    elif response.status_code == 404:
        return 'Failed to retrieve data: Endpoint not found.'
    elif response.status_code == 500:
        return 'Failed to retrieve data: Internal server error at CoinGecko.'
    else:
        return f'Failed to retrieve data: HTTP {response.status_code}.'


# Function to get OHLC range for a cryptocurrency
def get_crypto_ohlc(crypto_name):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_name.lower()}/ohlc'
    params = {
        'vs_currency': 'usd',
        'days': 1  # OHLC data for the last 1 day
    }
    headers = {
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data:
            ohlc = data[-1]  # Most recent OHLC data
            return (f"OHLC for {crypto_name.capitalize()} (USD):\n"
                    f"Open: ${ohlc[1]:.2f}, High: ${ohlc[2]:.2f}, Low: ${ohlc[3]:.2f}, Close: ${ohlc[4]:.2f}")
        else:
            return f'Sorry, I could not find OHLC data for {crypto_name}.'
    else:
        return 'Failed to retrieve data from the API.'


# Function to handle dynamic responses
def handle_dynamic_response(user_input):
    if 'price of' in user_input.lower():
        crypto_name = user_input.lower().split('price of')[-1].strip()
        return get_crypto_price(crypto_name)
    elif 'market cap of' in user_input.lower():
        crypto_name = user_input.lower().split('market cap of')[-1].strip()
        return get_crypto_market_cap(crypto_name)
    elif 'trading volume of' in user_input.lower():
        crypto_name = user_input.lower().split('trading volume of')[-1].strip()
        return get_crypto_trading_volume(crypto_name)
    elif 'recommend a coin' in user_input.lower():
        return recommend_coin()
    elif 'predict price of' in user_input.lower():
        crypto_name = user_input.lower().split('predict price of')[-1].strip()
        return predict_crypto_price(crypto_name)
    elif 'top gainers and losers' in user_input.lower():
        return get_top_gainers_losers()
    elif 'ohlc of' in user_input.lower():
        crypto_name = user_input.lower().split('ohlc of')[-1].strip()
        return get_crypto_ohlc(crypto_name)
    else:
        return static_chatbot.respond(user_input)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = handle_dynamic_response(user_input)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)

