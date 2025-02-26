import { startRecording, stopRecording } from './recording.js';
import './ui.js';

document.getElementById("startRecording").addEventListener("click", startRecording);
document.getElementById("stopRecording").addEventListener("click", stopRecording);