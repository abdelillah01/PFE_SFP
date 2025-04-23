from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'dataset' not in request.files:
        return jsonify({"error": "No dataset file uploaded"}), 400

    file = request.files['dataset']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # You can choose to save the file or just process it in memory
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Add your prediction code here and return a response with results
    return jsonify({"message": "✅ Prediction completed", "results": "your_results_here"})

if __name__ == '__main__':
    app.run(debug=True)