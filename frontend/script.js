const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const topnValue = document.getElementById("topnValue"); // span showing value
let topN = parseInt(topnValue.textContent); // default value
const darkModeToggle = document.getElementById("darkModeToggle");

darkModeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
  darkModeToggle.textContent = document.body.classList.contains("dark") ? "☀️" : "🌙";
});

// Increment & decrement buttons
document.getElementById("incrementBtn").addEventListener("click", () => {
  if (topN < 15) {
    topN++;
    topnValue.textContent = topN;
  }
});

document.getElementById("decrementBtn").addEventListener("click", () => {
  if (topN > 3) {
    topN--;
    topnValue.textContent = topN;
  }
});

function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.innerHTML = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const claim = userInput.value.trim();
  if (!claim) return;

  addMessage(claim, "user");
  userInput.value = "";
  sendBtn.disabled = true;
  addMessage("⏳ Processing...", "bot");

  try {
    const res = await fetch("http://127.0.0.1:8000/factcheck", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim: claim, top_n: topN }) // dynamic value
    });

    const data = await res.json();

    const responseText = `
      <b>Claim:</b> ${data.claim}<br>
      <b>Label:</b> ${data.label}<br>
      <b>References:</b> ${data.reference?.join(", ") || "None"}<br>
      <b>Explanation:</b> ${data.explanation}
    `;

    // Replace "Processing..."
    const lastBotMsg = chatBox.querySelector(".bot:last-child");
    lastBotMsg.innerHTML = responseText;
  } catch (err) {
    const lastBotMsg = chatBox.querySelector(".bot:last-child");
    lastBotMsg.innerHTML = "❌ Error contacting server.";
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});