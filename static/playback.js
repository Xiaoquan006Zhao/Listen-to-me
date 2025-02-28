import { socket } from './websocket.js';
import {resetInterruptedable} from './ui.js';

let audioQueue = [];
let isPlaying = false;
let stopPlayback = false;
let currentSource = null;
let stopWhenDone = false;

// Create a persistent AudioContext
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

// Handle "listening_to_user" event to stop playback
socket.on("listening_to_user", (data) => {
    if (data.listening) {
        console.log("Stopping playback due to user listening");
        stopPlayback = true;
        resetInterruptedable();
        audioQueue = []; // Clear the queue
        if (currentSource) {
            currentSource.stop();
            currentSource = null;
        }
    } else {
        console.log("Resuming playback");
        stopPlayback = false;
        stopWhenDone = false;
    }
});

socket.on("all_speech_sent", (data) => {
    if (data.all_sent) {
        stopWhenDone = true;
    } 
});

// Handle incoming audio stream
socket.on("audio_stream", (data) => {
    const { samplerate, samples } = data;
    const { byteArray, sampleRate } = decodeBase64Audio(samples, samplerate);

    audioQueue.push({ byteArray, sampleRate });

    // Start playback if not already playing
    if (!isPlaying) {
        playQueue();
    }
});

// Function to decode Base64 PCM audio
function decodeBase64Audio(base64String, sampleRate) {
    const binaryData = atob(base64String);
    const buffer = new ArrayBuffer(binaryData.length);
    const view = new DataView(buffer);

    for (let i = 0; i < binaryData.length; i++) {
        view.setUint8(i, binaryData.charCodeAt(i));
    }

    const byteArray = new Int16Array(buffer.byteLength / 2);
    for (let i = 0; i < byteArray.length; i++) {
        byteArray[i] = view.getInt16(i * 2, true); // Little-endian
    }

    return { byteArray, sampleRate };
}

// Function to play queued audio
function playQueue() {
    if ((audioQueue.length == 0 && stopWhenDone) || stopPlayback) {
        isPlaying = false;
        resetInterruptedable();
        return;
    }

    isPlaying = true;
    const { byteArray, sampleRate } = audioQueue.shift();

    // Create an audio buffer
    const buffer = audioCtx.createBuffer(1, byteArray.length, sampleRate);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < byteArray.length; i++) {
        channelData[i] = byteArray[i] / 32767.0; // Normalize PCM data
    }

    // Stop previous audio source if needed
    if (currentSource) {
        currentSource.stop();
    }

    // Create and play a new audio source
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.onended = playQueue; // Continue playback when this buffer ends

    currentSource = source;
    source.start();
}
