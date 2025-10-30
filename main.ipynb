{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "dbff98d0-54d0-464a-8760-a024bec31c74",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      " * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Admin\\AppData\\Local\\anaconda3\\New folder\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:3561: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "import json\n",
    "import numpy as np\n",
    "from flask import Flask, request, jsonify\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load model, scaler, and feature columns\n",
    "with open(\"linear_regression_model.pkl\", \"rb\") as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "with open(\"scaler.pkl\", \"rb\") as f:\n",
    "    scaler = pickle.load(f)\n",
    "\n",
    "with open(\"feature_columns.json\", \"r\") as f:\n",
    "    feature_columns = json.load(f)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return \"Manufacturing Equipment Output Prediction API is Running!\"\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    data = request.get_json()\n",
    "    input_data = [data.get(col, 0) for col in feature_columns]\n",
    "    input_scaled = scaler.transform([input_data])\n",
    "    prediction = model.predict(input_scaled)[0]\n",
    "    return jsonify({\"predicted_output\": prediction})\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0390ee5-07ca-4a17-98f6-d5126bc7f3b7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
