from flask import Flask, jsonify, render_template
import requests
from prometheus_client.parser import text_string_to_metric_families
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


app = Flask(__name__)

# Add after METRICS_URL
MONGO_URI = "mongodb+srv://admin:admin@cluster0.ipm8e.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
METRICS_URL = 'http://localhost:5000/metrics'

def get_metrics():
    response = requests.get(METRICS_URL)
    return text_string_to_metric_families(response.text)

def get_metric_value(metric_name, label=None):
    for family in get_metrics():
        for sample in family.samples:
            if sample.name == metric_name:
                if not label or sample.labels == label:
                    return sample.value
    return None

def get_gauge_value(metric_name):
    try:
        response = requests.get(METRICS_URL)
        families = text_string_to_metric_families(response.text)
        for family in families:
            if family.name == metric_name:
                for sample in family.samples:
                    return sample.value
        return 0.0
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return 0.0
    
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/cpu_usage')
def get_cpu_usage():
    return jsonify({'value': get_gauge_value('cpu_usage_percent')})

@app.route('/memory_usage')
def get_memory_usage():
    return jsonify({'value': get_gauge_value('memory_usage_percent')})

@app.route('/mongo_status')
def mongo_status():
    try:
        # The ping command is cheap and does not require auth
        client.admin.command('ping')
        return jsonify({'status': 'active'})
    except ConnectionFailure:
        return jsonify({'status': 'inactive'})
    
if __name__ == '__main__':
    app.run(port=5001)