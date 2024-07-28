from flask import Flask
import threading
import logging
from AirCanva import run_opencv  # Make sure this import is correct and points to your AirCanva function

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, filename="server.log")


@app.route('/start-opencv', methods=['POST'])
def start_opencv():
    logging.debug("Received request to start OpenCV")
    opencv_thread = threading.Thread(target=run_opencv)
    opencv_thread.start()
    return "OpenCV started", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)