# src/visualizer/main.py

from flask import Flask, render_template, jsonify, send_from_directory
import os
import json

app = Flask(__name__)

# Path to the latest report and log file
LATEST_REPORT_DIR = "../../data/output/latest"
REPORT_FILE = "report.json"
LOG_FILE = "app_info.log"

@app.route("/")
def index():
    """Serves the main HTML page for the report viewer."""
    return render_template("index.html")

@app.route("/api/report")
def get_report():
    """Serves the latest report.json file."""
    try:
        report_path = os.path.join(LATEST_REPORT_DIR, REPORT_FILE)
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        return jsonify(report_data)
    except FileNotFoundError:
        return jsonify({"error": "report.json not found in the latest output directory."}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in report.json."}), 500

@app.route("/api/log")
def get_log():
    """Serves the latest app_info.log file."""
    try:
        log_path = os.path.join(LATEST_REPORT_DIR, LOG_FILE)
        return send_from_directory(os.path.dirname(log_path), os.path.basename(log_path))
    except FileNotFoundError:
        return jsonify({"error": "app_info.log not found in the latest output directory."}), 404

if __name__ == "__main__":
    app.run(debug=True)
