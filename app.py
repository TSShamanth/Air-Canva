import threading
from flask import Flask, render_template, redirect, url_for

# Importing run_opencv from AirCanva
from AirCanva import run_opencv

app = Flask(__name__)

# Global variable to track if OpenCV is running
opencv_running = False


def start_opencv():
    global opencv_running
    if not opencv_running:
        opencv_running = True
        run_opencv()


@app.route('/')
def home():
    return render_template('Webpage.html', opencv_running=opencv_running)


@app.route('/start_opencv', methods=['POST'])
def start_opencv_route():
    global opencv_running
    if not opencv_running:
        opencv_running = True
        opencv_thread = threading.Thread(target=start_opencv)
        opencv_thread.start()
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
