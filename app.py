from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from yfinance import exceptions
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, GRU, Concatenate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import string
import datetime
import requests

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/stock'  
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Hist(db.Model):
    __tablename__ = 'hist'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    stock_name = db.Column(db.String(255),nullable=False)
    four_image = db.Column(db.String(255),nullable=False)
    half_image = db.Column(db.String(255),nullable=False)
    year_image = db.Column(db.String(255),nullable=False)
    tstamp = db.Column(db.String(255), nullable=False)

def generate_random_name(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model(time_step, num_features):
    # LSTM branch
    lstm_branch = Sequential()
    lstm_branch.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, num_features)))
    lstm_branch.add(LSTM(units=50, return_sequences=False))

    # GRU branch
    gru_branch = Sequential()
    gru_branch.add(GRU(units=50, return_sequences=True, input_shape=(time_step, num_features)))
    gru_branch.add(GRU(units=50, return_sequences=False))

    # Concatenate LSTM and GRU outputs
    combined = Concatenate()([lstm_branch.output, gru_branch.output])

    # Output layer
    output_layer = Dense(units=1)(combined)

    # Create model
    model = Model(inputs=[lstm_branch.input, gru_branch.input], outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit([X_train, X_train], y_train, epochs=epochs, batch_size=batch_size, verbose=1)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict([X_test, X_test])
    return predictions

def make_predictions(model, data, scaler, time_step, num_features, future_days):
    prediction_list = data[-time_step:].tolist()
    for _ in range(future_days):
        x_input = np.array(prediction_list[-time_step:]).reshape(1, time_step, num_features)
        prediction = model.predict([x_input, x_input], verbose=0)
        prediction_list.append(prediction[0])
    prediction_list = scaler.inverse_transform(np.array(prediction_list).reshape(-1, 1))
    return prediction_list[-future_days:]

def save_four_predictions_plot(actual_data, future_predictions, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data, label='Actual Stock Prices')
    plt.plot(np.arange(len(actual_data), len(actual_data) + len(future_predictions)), future_predictions, label='Predicted Stock Prices')
    plt.title('Stock Price Prediction for Next 45 Days')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def save_half_predictions_plot(actual_data, future_predictions, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data, label='Actual Stock Prices')
    plt.plot(np.arange(len(actual_data), len(actual_data) + len(future_predictions)), future_predictions, label='Predicted Stock Prices')
    plt.title('Stock Price Prediction for Next Half Year')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def save_year_predictions_plot(actual_data, future_predictions, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data, label='Actual Stock Prices')
    plt.plot(np.arange(len(actual_data), len(actual_data) + len(future_predictions)), future_predictions, label='Predicted Stock Prices')
    plt.title('Stock Price Prediction for Next Year')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def invest_1_lakh(stock_data, future_predictions, investment_amount):
    # Get the latest closing price
    latest_closing_price = stock_data['Close'].iloc[-1]

    # Calculate the number of shares you can buy with 1 lakh
    shares_bought = investment_amount / latest_closing_price

    # Calculate the future value of the investment based on predictions
    future_value = future_predictions[-1] * shares_bought

    return shares_bought, future_value

@app.route('/index')
@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Input validation
        if not username or not email or not password :
            return render_template('signup.html', msg='All fields are required.')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Check for existing username or email
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()

        if existing_user:
            return render_template('signup.html', msg='Username already exists. Please choose a different one.')
        elif existing_email:
            return render_template('signup.html', msg='Email already exists. Please use a different one.')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)

        try:
            db.session.commit()
            print("Database commit successful!")
            return render_template('signup.html', msg='Signup successful. Now you can login.')
        except Exception as e:
            db.session.rollback()
            print("Database commit failed:", str(e))
            return render_template('signup.html', msg='An error occurred during signup.')
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Retrieve user based on the provided email
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            # User exists and password matches
            session['user_id'] = user.id
            session['user_email'] = user.email
            return redirect(url_for('dashboard'))  
        else:
            # User doesn't exist or password is incorrect
            return render_template('login.html', msg='Invalid email or password. Please try again.')
    return render_template('login.html')

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    query = request.form.get('query', '').upper()  # Convert query to uppercase for consistency
    if not query:
        return jsonify([])

    # Read the merged symbols CSV file into a DataFrame
    df = pd.read_csv('dataset/merged_symbols.csv', header=0, dtype=str)

    # Drop rows with missing values
    df = df.dropna()

    # Filter symbols based on the query
    suggestions = df[df['Symbol'].str.contains(query, na=False)]['Symbol'].tolist()

    return jsonify(suggestions[:10])  # Return top 10 suggestions

@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    if session.get('user_id'):
        if request.method == 'POST':
            stock_name = request.form['stock_name']
            symbol = stock_name
            start_date = '2010-01-01'
            end_date = '2024-01-01'
            stock_data = get_stock_data(symbol, start_date, end_date)
            data = stock_data['Close'].values
            scaled_data, scaler = preprocess_data(data)
            # Split data into training and testing sets
            time_step = 100
            X, y = create_dataset(scaled_data, time_step)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Reshape data for LSTM and GRU
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            # Build the combined model
            combined_model = build_model(time_step, 1)
            # Train the combined model
            epochs = 100
            batch_size = 64
            train_model(combined_model, X_train, y_train, epochs, batch_size)
            # Evaluate the model
            predictions = evaluate_model(combined_model, X_test, y_test)
            future_days = 45
            future_predictions = make_predictions(combined_model, scaled_data, scaler, time_step, 1, future_days)

            # Calculate mean absolute error
            mse = mean_squared_error(data[-len(future_predictions):], future_predictions)
            mae = mean_absolute_error(data[-len(future_predictions):], future_predictions)
            random_name = generate_random_name()
            random_four_name = random_name + '.png'
            save_four_predictions_plot(data, future_predictions, os.path.join(app.config['UPLOAD_FOLDER'], random_four_name))

            future_days = 180
            future_predictions = make_predictions(combined_model, scaled_data, scaler, time_step, 1, future_days)

            # Calculate mean absolute error
            mse = mean_squared_error(data[-len(future_predictions):], future_predictions)
            mae = mean_absolute_error(data[-len(future_predictions):], future_predictions)
            random_name = generate_random_name()
            random_half_name = random_name + '.png'
            save_half_predictions_plot(data, future_predictions, os.path.join(app.config['UPLOAD_FOLDER'], random_half_name))

            future_days = 365
            future_predictions = make_predictions(combined_model, scaled_data, scaler, time_step, 1, future_days)

            # Calculate mean absolute error
            mse = mean_squared_error(data[-len(future_predictions):], future_predictions)
            mae = mean_absolute_error(data[-len(future_predictions):], future_predictions)
            random_name = generate_random_name()
            random_year_name = random_name + '.png'
            save_year_predictions_plot(data, future_predictions, os.path.join(app.config['UPLOAD_FOLDER'], random_year_name))
            current_datetime = datetime.datetime.now()
            new_history = Hist(user_id=session['user_id'], stock_name=stock_name, four_image=random_four_name, half_image=random_half_name, year_image=random_year_name, tstamp=current_datetime)
            db.session.add(new_history)
            try:
                db.session.commit()
                print("Database commit successful!")
                return render_template('dashboard.html', random_four_name=random_four_name, random_half_name=random_half_name, random_year_name=random_year_name)
            except Exception as e:
                db.session.rollback()
                print("Database commit failed:", str(e))
                return render_template('dashboard.html')
        return render_template('dashboard.html')
    else:
        return redirect(url_for('index'))
    
@app.route('/invest', methods=['GET','POST'])
def invest():
    if session.get('user_id'):
        if request.method == 'POST':
            stock_name = request.form['stock_name']
            symbol = stock_name
            start_date = '2010-01-01'
            end_date = '2024-01-01'
            stock_data = get_stock_data(symbol, start_date, end_date)
            data = stock_data['Close'].values
            scaled_data, scaler = preprocess_data(data)
            # Split data into training and testing sets
            time_step = 100
            X, y = create_dataset(scaled_data, time_step)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Reshape data for LSTM and GRU
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            # Build the combined model
            combined_model = build_model(time_step, 1)
            # Train the combined model
            epochs = 100
            batch_size = 64
            train_model(combined_model, X_train, y_train, epochs, batch_size)
            future_days = 45
            future_predictions_45 = make_predictions(combined_model, scaled_data, scaler, time_step, 1, future_days)
            investment_amount = 100000  # 1 lakh
            shares_bought_45, future_value_45 = invest_1_lakh(stock_data, future_predictions_45, investment_amount)
            random_name = generate_random_name()
            random_four_name = random_name + '.png'
            save_four_predictions_plot(data, future_predictions_45, os.path.join(app.config['UPLOAD_FOLDER'], random_four_name))
            future_days = 180
            future_predictions_180 = make_predictions(combined_model, scaled_data, scaler, time_step, 1, future_days)
            shares_bought_180, future_value_180 = invest_1_lakh(stock_data, future_predictions_180, investment_amount)
            random_name = generate_random_name()
            random_half_name = random_name + '.png'
            save_half_predictions_plot(data, future_predictions_180, os.path.join(app.config['UPLOAD_FOLDER'], random_half_name))
            future_days = 365
            future_predictions_365 = make_predictions(combined_model, scaled_data, scaler, time_step, 1, future_days)
            shares_bought_365, future_value_365 = invest_1_lakh(stock_data, future_predictions_365, investment_amount)
            random_name = generate_random_name()
            random_year_name = random_name + '.png'
            save_year_predictions_plot(data, future_predictions_365, os.path.join(app.config['UPLOAD_FOLDER'], random_year_name))
            return render_template('invest.html', random_four_name=random_four_name, random_half_name=random_half_name, random_year_name=random_year_name, shares_bought_45=shares_bought_45, future_value_45=future_value_45, shares_bought_180=shares_bought_180, future_value_180=future_value_180, shares_bought_365=shares_bought_365, future_value_365=future_value_365)
        return render_template('invest.html')
    else:
        return redirect(url_for('index'))
    
@app.route('/history', methods=['GET','POST'])
def history():
    if session.get('user_id'):
        all_data = Hist.query.filter_by(user_id=session['user_id']).all()
        return render_template('history.html', all_data=all_data)
    else:
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_email', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)