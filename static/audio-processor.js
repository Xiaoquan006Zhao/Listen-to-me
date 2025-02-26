class AudioProcessor extends AudioWorkletProcessor {
  process(inputs) {
    if (inputs[0] && inputs[0][0]) {
      this.port.postMessage(inputs[0][0]);
    } else {
      console.log('No audio input detected'); // Debugging log
    }
    return true;
  }
}
registerProcessor('audio-processor', AudioProcessor);
