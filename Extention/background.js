chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed.");
});

chrome.action.onClicked.addListener((tab) => {
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
