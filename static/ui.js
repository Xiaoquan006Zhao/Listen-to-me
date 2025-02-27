import { socket } from './websocket.js';

let chatHistoryDiv = document.getElementById('chatHistory');
let userInputDiv = document.getElementById('userInput');
let llmMessageDiv;
let llmAnswer = ''; // Variable to keep track of the full LLM answer

let onlineTranscription = '';
let offlineTranscription = '';
let idleCounterContainer = document.getElementById('idleCounterContainer');
let maxIdleCounter = 0; // This will be set by the server

let interrupted = false; 
let interruptedable = false; 

function resetInterruptedable() {
    interruptedable = false;
}

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

function flashLLMText() {
    const llmTextElement = llmMessageDiv.querySelector('.llm-text');
    if (llmTextElement) {
        llmTextElement.classList.add('flash');
        // Remove the flash class after the animation duration (500ms)
        setTimeout(() => {
            llmTextElement.classList.remove('flash');
        }, 500);
    }
}



socket.on('listening_to_user', function(data) {
    // Dont interrupt again
    if (data.listening && !interrupted && interruptedable) {
        interrupted = true; 
        flashLLMText();
    }
});


// llm started means user finished speaking
socket.on('llm_started', function(data) {
    if (data.started) {
        interrupted = false;
        interruptedable = true;

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


// Function to create idle counter circles
function createIdleCounterCircles(maxCounter) {
    idleCounterContainer.innerHTML = ''; // Clear existing circles
    for (let i = 0; i < maxCounter; i++) {
        let circle = document.createElement('div');
        circle.classList.add('idle-counter-circle');
        if (i >= maxCounter - 1) {
            circle.classList.add('inactive'); // Initially, all circles are inactive
        }
        idleCounterContainer.appendChild(circle);
    }
    updateIdleCounterCircles(maxCounter);
}

// Update the idle counter circles
function updateIdleCounterCircles(currentCounter) {
    let circles = idleCounterContainer.querySelectorAll('.idle-counter-circle');
    circles.forEach((circle, index) => {
        if (index < currentCounter) {
            circle.classList.remove('inactive');
        } else {
            circle.classList.add('inactive');
        }
    });
}

// -2 for reaction buffer
// Event listener for user_idle_counter_threshold
socket.on('user_idle_counter_threshold', function(data) {
    maxIdleCounter = data.threshold-2;
    createIdleCounterCircles(maxIdleCounter);
});

// Event listener for user_idle_counter
socket.on('user_idle_counter', function(data) {
    if (data.counter !== undefined) {

        updateIdleCounterCircles(data.counter-2);
    }
});

// maxIdleCounter = 8;
// createIdleCounterCircles(maxIdleCounter);

export { resetInterruptedable };