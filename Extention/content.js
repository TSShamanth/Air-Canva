// Example content script to interact with Flask server
function startOpenCV() {
  fetch('http://localhost:8080/start-opencv', { method: 'POST' })
    .then(response => response.text())
    .then(data => {
      console.log('Response:', data);
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

// Example event listener for a button click in your extension
document.getElementById('startOpenCV').addEventListener('click', startOpenCV);

