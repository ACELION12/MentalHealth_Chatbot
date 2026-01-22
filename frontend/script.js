// frontend/static/script.js (Final Version)
const API_BASE = (window.API_BASE !== undefined) ? window.API_BASE : "";
const DEBUG = true;

window.dlog = function(...args) { if (DEBUG) console.debug("[aura-debug]", ...args); }

// DOM elements
const chatMessages = document.getElementById("chat-messages");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

// ESCALATION ELEMENTS
const escalationModal = document.getElementById("escalation-modal");
const escalationText = document.getElementById("escalation-text");
const closeEscalationBtn = document.getElementById("close-escalation-btn");

// BOOKING ELEMENTS
const bookingModal = document.getElementById("booking-modal");
const bookingSuggestionText = document.getElementById("booking-suggestion-text");
const confirmBookingBtn = document.getElementById("confirm-booking-btn");
const closeBookingBtn = document.getElementById("close-booking-btn");
const checkSlotsBtn = document.getElementById("check-slots-btn");
const slotDateInput = document.getElementById("slot-date");
const slotsList = document.getElementById("slots-list");

// State Management
let sessionId = localStorage.getItem("aura_session");
if (!sessionId) {
    sessionId = (crypto && crypto.randomUUID) ? crypto.randomUUID() : "sess-" + Date.now();
    localStorage.setItem("aura_session", sessionId);
}
let currentAgentState = "ASSESSING";
let lastUserMessage = ""; // Tracks what YOU said for "God Mode" logic

function renderMessage(text, cls = "agent-message", opts = {}) {
    if (!chatMessages) return;
    const div = document.createElement("div");
    div.className = `message ${cls}`;
    div.textContent = text; // Safer than innerHTML
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

// Typing Indicator
let typingIndicator = null;
function showTyping() {
    if (!chatMessages || typingIndicator) return;
    typingIndicator = document.createElement("div");
    typingIndicator.className = "message agent-message typing";
    typingIndicator.textContent = "Aura is typing…";
    chatMessages.appendChild(typingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
function hideTyping() {
    if (!typingIndicator) return;
    typingIndicator.remove();
    typingIndicator = null;
}

// Welcome Message
if (chatMessages && chatMessages.children.length === 0) {
    renderMessage("Hello! I'm Aura. I'm here to listen. How can I help you today?", "agent-message");
}

// Send Message Logic
async function sendMessageToBackend(message) {
    if (!message) return;
    renderMessage(message, "user-message");
    
    // 1. SAVE USER MESSAGE FOR LOGIC CHECKS
    lastUserMessage = message.toLowerCase();

    if (sendButton) sendButton.disabled = true;
    showTyping();

    const payload = {
        user_message: message,
        session_id: sessionId,
        current_agent_state: currentAgentState
    };

    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        const data = await resp.json();
        handleBackendResponse(data);

    } catch (err) {
        renderMessage("Network error. Please try again.", "agent-message");
    } finally {
        hideTyping();
        if (sendButton) sendButton.disabled = false;
        if (userInput) userInput.focus();
    }
}

// ---------------------------------------------------------
// 🤖 RESPONSE HANDLER (THE LOGIC CORE)
// ---------------------------------------------------------
function handleBackendResponse(response) {
    const agentText = response.agent_response || "...";
    renderMessage(agentText, "agent-message");

    currentAgentState = response.current_agent_state || currentAgentState;
    const ui = (response.ui_state || "CHAT").toUpperCase();

    // 🛠️ SMART CHECK 1: Did the bot imply it opened the scheduler?
    const textLower = agentText.toLowerCase();
    const botImpliesBooking = textLower.includes("scheduler") || 
                              textLower.includes("select a time") ||
                              textLower.includes("book an appointment");

    // 🛠️ SMART CHECK 2 (GOD MODE): Did the USER explicitly ask for it?
    // If so, we force it open, ignoring bot refusals.
    const userWantsBooking = lastUserMessage.includes("book") || 
                             lastUserMessage.includes("schedule") || 
                             lastUserMessage.includes("appointment") ||
                             lastUserMessage.includes("slots") ||
                             lastUserMessage.includes("doctor");

    if (ui === "ESCALATION_MODAL") {
        showEscalationModal(response.modal_text || agentText);
    } 
    else if (ui === "BOOKING_MODAL" || botImpliesBooking || userWantsBooking) { 
        // Force open if User asked, Bot implied, or Backend commanded
        showBookingModal("I've opened the scheduler for you.");
    } 
}

// Basic Event Listeners
if (sendButton) {
    sendButton.addEventListener("click", () => {
        const text = userInput.value.trim();
        userInput.value = "";
        sendMessageToBackend(text);
    });
}
if (userInput) {
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendButton.click();
        }
    });
}

// ---------------------------------------------------------
// 🚨 MODAL FUNCTIONS
// ---------------------------------------------------------
window.showEscalationModal = function(text) {
    if (escalationModal) {
        if (escalationText) escalationText.textContent = text;
        escalationModal.classList.add("open");
    }
}
window.closeEscalationModal = function() {
    if (escalationModal) escalationModal.classList.remove("open");
}
if (closeEscalationBtn) {
    closeEscalationBtn.addEventListener("click", (e) => {
        e.preventDefault();
        window.closeEscalationModal();
        renderMessage("No, I'm safe now.", "user-message");
    });
}

// ---------------------------------------------------------
// 📅 BOOKING LOGIC
// ---------------------------------------------------------
function showBookingModal(text) {
    if (bookingModal) {
        if (bookingSuggestionText) bookingSuggestionText.textContent = text;
        bookingModal.classList.add("open");
    }
}
function closeBookingModal() {
    if (bookingModal) bookingModal.classList.remove("open");
    if (slotsList) slotsList.innerHTML = "";
}

if (closeBookingBtn) closeBookingBtn.addEventListener("click", closeBookingModal);

if (checkSlotsBtn) {
    checkSlotsBtn.addEventListener("click", () => {
        if (slotDateInput && slotDateInput.value) fetchSlotsForDate(slotDateInput.value);
        else alert("Pick a date.");
    });
}

async function fetchSlotsForDate(dateStr) {
    if (!slotsList) return;
    slotsList.innerHTML = "Loading...";
    try {
        const res = await fetch(`${API_BASE}/api/slots?date=${dateStr}`);
        const data = await res.json();
        renderSlots(data.slots || {}, dateStr);
    } catch (e) {
        slotsList.innerHTML = "Error loading slots.";
    }
}

function renderSlots(slotsObj, dateStr) {
    if (!slotsList) return;
    slotsList.innerHTML = "";
    const times = Object.keys(slotsObj).sort();
    
    times.forEach(t => {
        const docs = slotsObj[t];
        const row = document.createElement("div");
        row.className = "slot-item";
        
        const info = document.createElement("div");
        info.innerHTML = `<strong>${t}</strong>`;
        
        const actions = document.createElement("div");
        docs.forEach(doc => {
            const btn = document.createElement("button");
            btn.textContent = `Book ${doc}`;
            btn.addEventListener("click", () => attemptBooking(dateStr, t, doc));
            actions.appendChild(btn);
        });

        row.appendChild(info);
        row.appendChild(actions);
        slotsList.appendChild(row);
    });
}

// ---------------------------------------------------------
// ✨ BOOKING CONFIRMATION & THEME TRIGGER
// ---------------------------------------------------------
async function attemptBooking(date, time, doctor) {
    try {
        const res = await fetch(`${API_BASE}/api/book`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ date, time, doctor, meta: { session: sessionId } })
        });
        
        if (res.ok) {
            const data = await res.json();
            
            renderMessage(
                `✅ Booked ${data.booking.doctor} on ${data.booking.date} at ${data.booking.time}. A reminder has been set in your Google Calendar and sent via Email.`, 
                "agent-message"
            );

            // 🚀 TRIGGER CLARITY MODE (Comfort Transition)
            console.log("🚀 Booking Confirmed! Shifting to Clarity Mode...");
            document.body.classList.add("clarity-mode");
            
            closeBookingModal();
        } 
    } catch (e) {
        renderMessage("Booking failed.", "agent-message");
    }
}

// Emergency Call Logic
window.triggerEmergencyCall = async function() {
    const phoneInput = document.getElementById("patient-phone");
    const phone = phoneInput ? phoneInput.value.trim() : "";
    if(!phone) return alert("Enter phone number");

    try {
        const res = await fetch(`${API_BASE}/api/emergency/call`, {
            method: "POST", headers: {"Content-Type":"application/json"},
            body: JSON.stringify({ phone_number: phone })
        });
        if(res.ok) {
            closeEscalationModal();
            renderMessage("📞 Dialing...", "agent-message");
            
            // 🚀 TRIGGER CLARITY MODE ON EMERGENCY TOO
            document.body.classList.add("clarity-mode"); 
            
            alert("Calling now.");
        }
    } catch(e) { alert("Call failed"); }
};