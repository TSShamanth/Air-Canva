import logging
import threading
from flask import Flask, render_template
from AirCanva import run_opencv
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("Webpage.html")

@app.route("/start-opencv", methods=["POST"])
def start_opencv():
    logging.info("Received request to start OpenCV.")
    try:
        opencv_thread = threading.Thread(target=run_opencv)
        opencv_thread.start()
        logging.info("OpenCV thread started.")
        return "OpenCV started successfully!"
    except Exception as e:
        logging.error(f"Error starting OpenCV: {e}")
        return f"Error starting OpenCV: {e}"

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8080)
