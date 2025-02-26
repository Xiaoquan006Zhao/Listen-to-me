import { socket } from './websocket.js';

const TARGET_SAMPLE_RATE = 16000;
const SEND_FRAME_SIZE = 9600; // 600ms frame at 16kHz
let audioContext;
let mediaStream;
let workletNode;
let isRecording = false;

let inputBuffer = new Float32Array(0);
let pcmBuffer = new Int16Array(0);

async function startRecording() {
  try {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await audioContext.audioWorklet.addModule('/static/audio-processor.js');
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioContext.createMediaStreamSource(mediaStream);
    
    workletNode = new AudioWorkletNode(audioContext, 'audio-processor');

    workletNode.port.onmessage = ({ data }) => {
        handleAudioData(data);
    };

    source.connect(workletNode);
    workletNode.connect(audioContext.destination);
    
    isRecording = true;
    toggleButtons(true);
  } catch (err) {
    console.error("Recording setup failed:", err);
    alert("Microphone access required for recording");
  }
}

function stopRecording() {
  isRecording = false;
  if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
  if (audioContext) audioContext.close();
  audioContext = null;
  flushBuffers();
  toggleButtons(false);
}

async function handleAudioData(float32Array) {
  if (!isRecording) return;

  // Resample to 16kHz if needed
  if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
    float32Array = await resampleAudio(float32Array, audioContext.sampleRate, TARGET_SAMPLE_RATE);
  }

  // Convert to 16-bit PCM
  const pcmFrame = convertFloat32ToInt16(float32Array);

  // Buffer management
  const combined = new Int16Array(pcmBuffer.length + pcmFrame.length);
  combined.set(pcmBuffer);
  combined.set(pcmFrame, pcmBuffer.length);
  pcmBuffer = combined;

  // Send chunks of appropriate size
  while (pcmBuffer.length >= SEND_FRAME_SIZE) {
    const chunk = pcmBuffer.slice(0, SEND_FRAME_SIZE);
    pcmBuffer = pcmBuffer.slice(SEND_FRAME_SIZE);
    sendAudioChunk(chunk);
  }
}

function resampleAudio(input, originalRate, targetRate) {
  const ratio = targetRate / originalRate;
  const length = Math.round(input.length * ratio);
  const offlineContext = new OfflineAudioContext(1, length, targetRate);
  const buffer = offlineContext.createBuffer(1, input.length, originalRate);
  buffer.copyToChannel(input, 0);
  
  const source = offlineContext.createBufferSource();
  source.buffer = buffer;
  source.connect(offlineContext.destination);
  source.start();
  
  return offlineContext.startRendering().then(resampledBuffer => 
    resampledBuffer.getChannelData(0)
  );
}

function convertFloat32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        int16Array[i] = Math.max(-32768, Math.min(32767, float32Array[i] * 32768));
    }

    return int16Array;
}

function sendAudioChunk(chunk) {
  if (chunk.length > 0) {
    socket.emit('audio_data', chunk.buffer);
  }
}

function flushBuffers() {
  if (pcmBuffer.length > 0) {
    sendAudioChunk(pcmBuffer);
    pcmBuffer = new Int16Array(0);
  }
}

function toggleButtons(recording) {
  document.getElementById("stopRecording").disabled = !recording;
  document.getElementById("startRecording").disabled = recording;
}

export { startRecording, stopRecording };