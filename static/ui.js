import { socket } from './websocket.js';

let chatHistoryDiv = document.getElementById('chatHistory');
let userInputDiv = document.getElementById('userInput');
let llmMessageDiv;
let llmAnswer = ''; // Variable to keep track of the full LLM answer

let onlineTranscription = '';
let offlineTranscription = '';

function create_user_message(offlineTranscription) {
    let messageElement = document.createElement('div');
    messageElement.textContent = offlineTranscription;
    return messageElement;
}

function create_llm_message() {
    let messageElement = document.createElement('div');
    
    // Create the text part of the LLM message (empty initially)
    let textElement = document.createElement('span');
    textElement.classList.add('llm-text');
    messageElement.appendChild(textElement);
    
    // Create a span for each dot and apply the jumping animation with an offset
    let dotsContainer = document.createElement('span');
    dotsContainer.classList.add('jumping-dots');
    
    // Add each dot as a span element
    for (let i = 0; i < 3; i++) {
        let dotElement = document.createElement('span');
        dotElement.textContent = ".";
        dotElement.classList.add('jumping-dot');
        dotElement.style.animationDelay = `${i * 0.2}s`;  // Offset the animation by 0.2s for each dot
        dotsContainer.appendChild(dotElement);
    }
    
    messageElement.appendChild(dotsContainer);
    
    return { messageElement, textElement, dotsContainer };
}

// llm started means user finished speaking
socket.on('llm_started', function(data) {
    if (data.started) {
        let userMessageElement = create_user_message(offlineTranscription);
        chatHistoryDiv.appendChild(userMessageElement);
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;  // Auto-scroll

        offlineTranscription = ''; 
        userInputDiv.textContent = '';
        updateTranscription();

        // Create the LLM message with dots
        const { messageElement, textElement, dotsContainer } = create_llm_message();
        llmMessageDiv = messageElement;

        // Append the LLM message to chat
        chatHistoryDiv.appendChild(llmMessageDiv);
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;  // Auto-scroll

        // Reset LLM answer
        llmAnswer = ''; // Clear LLM answer when a new conversation starts
    }
});

socket.on('llm_stopped', function(data) {
    if (data.stopped) {
        // Clear dots and update the LLM answer
        llmMessageDiv.removeChild(llmMessageDiv.querySelector('.jumping-dots'));
        llmMessageDiv.querySelector('.llm-text').textContent = llmAnswer; 
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;  // Auto-scroll
    }
});

socket.on('online_transcription', function(data) {
    onlineTranscription += data.message;
    updateTranscription();
});

socket.on('offline_transcription', function(data) {
    onlineTranscription = '';
    offlineTranscription += data.message;
    updateTranscription();
});

function updateTranscription() {
    userInputDiv.textContent = offlineTranscription + onlineTranscription;
}

socket.on('llm_answer', function(data) {
    let token = data.message;
    console.log("Received token:", token);

    llmAnswer += token;

    llmMessageDiv.querySelector('.llm-text').textContent = llmAnswer;
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;  // Auto-scroll
});
