// Example content script to interact with Flask server
function startOpenCV() {
  fetch('http://127.0.0.1:5000/start-opencv', { method: 'POST' })
    .then(response => response.text())
    .then(data => {
      console.log('Response:', data);
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

// Example event listener for a button click in your extension
document.getElementById('start-opencv-button').addEventListener('click', startOpenCV);
