// webpage.js

document.addEventListener('DOMContentLoaded', function() {
  const startOpenCVButton = document.getElementById('startOpenCV');

  startOpenCVButton.addEventListener('click', function() {
    chrome.runtime.sendMessage({ command: "startOpenCV" }, function(response) {
      console.log('Received response:', response);
      // Optionally handle response from background script
    });
  });
});
