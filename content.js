// Inject the `injected.js` script into the page's context
const script = document.createElement("script");
script.src = chrome.runtime.getURL("injected.js");
script.onload = () => script.remove(); // Clean up after loading
(document.head || document.documentElement).appendChild(script);

