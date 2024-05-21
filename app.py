import io
from flask import Flask, render_template, request, jsonify
import pandas as pd
from your_model_script import preprocess_data, train_and_evaluate_models

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        df = pd.read_csv(file)
        global uploaded_df
        uploaded_df = df  # Store the dataframe globally for access in other endpoints
        return jsonify({'success': True, 'message': 'File uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/evaluate_classification', methods=['POST'])
def evaluate_classification():
    data = request.get_json()
    algorithm = data['algorithm']
    df = uploaded_df  # Access the globally stored dataframe
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, task='classification')
    model, accuracy = train_and_evaluate_models(X_train, X_test, y_train, y_test, task='classification', algorithm=algorithm)
    return jsonify({'accuracy': accuracy})

@app.route('/evaluate_regression', methods=['POST'])
def evaluate_regression():
    data = request.get_json()
    algorithm = data['algorithm']
    df = uploaded_df  # Access the globally stored dataframe
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, task='regression')
    model, mse = train_and_evaluate_models(X_train, X_test, y_train, y_test, task='regression', algorithm=algorithm)
    return jsonify({'mse': mse})

if __name__ == '__main__':
    app.run(debug=True)
