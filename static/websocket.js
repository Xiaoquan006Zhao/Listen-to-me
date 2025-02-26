const socket = io.connect('http://127.0.0.1:8080');

// Server communication handlers
socket.on('audio_ack', data => {
  console.log('Server acknowledgment:', data.message);
});

socket.on('connect_error', error => {
  console.error('WebSocket connection error:', error);
});

export { socket };