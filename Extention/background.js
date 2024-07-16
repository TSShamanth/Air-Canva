chrome.runtime.onInstalled.addListener(function() {
  console.log('Extension installed');

  // Example function to send a POST request to start AirCanva
  function startAirCanva() {
    fetch('http://127.0.0.1:5000', {
      method: 'POST'
    })
    .then(response => response.text())
    .then(data => {
      console.log('Response:', data);
      // Optionally handle the response
    })
    .catch(error => {
      console.error('Error:', error);
    });
  }

  // Example: Listen for browser action click event
  chrome.action.onClicked.addListener(function(tab) {
    startAirCanva();
  });
});
