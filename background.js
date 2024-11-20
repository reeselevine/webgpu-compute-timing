chrome.runtime.onInstalled.addListener(() => {
  console.log("WebGPU Timestamp Capture Extension Installed");
});

// Example: Listening for a browser action (like clicking an icon)
chrome.action.onClicked.addListener((tab) => {
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ["content.js"]
  });
});

