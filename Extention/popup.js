document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('startAirCanva').addEventListener('click', () => {
    fetch('http://localhost:5000/start-opencv', {
      method: 'POST'
    })
    .then(response => response.text())
    .then(data => {
      console.log(data); // Handle the response from the server
    })
    .catch(error => {
      console.error('Error:', error);
    });
  });
});
