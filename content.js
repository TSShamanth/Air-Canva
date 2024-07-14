// This script will send frames to the Flask server and receive the processed frames
const serverUrl = 'http://localhost:5000/process_frame';

async function processFrame(frame) {
  const formData = new FormData();
  formData.append('frame', frame);

  const response = await fetch(serverUrl, {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  return data.frame;
}

chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
  if (message.action === 'process_frame') {
    const frame = message.frame;
    const processedFrame = await processFrame(frame);
    sendResponse({ frame: processedFrame });
  }
});
