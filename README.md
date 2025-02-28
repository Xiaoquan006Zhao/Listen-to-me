# Stop Yapping 
Is a voice assistant that you can interrupt any time. It has near real-time responsiveness on Mac M2-pro. 

It has three major components: Speech-to-Text, LLM, Text-to-Speech.

## Demo

## Speech-to-Text
Currently is powered by FunASR's SenseVoiceSmall model (it has more language options). 
Every 0.6s the model process the current audio chunk and as a responsive cue to indicate that the user input is successfully processed (we denote this as the online process). 
However at current stage there is no smart word boundary detection, the 0.6s hard cut will likely cut the words in pieces causing errors in transcription. 
Thus we adapt two pass system (online and offline) inspired by FunASR. 
After a small pause, the offline model will process the collected during the online process. As a result of more datapoints and contineous audio stream, the offline model is usually more accurate.

### Speaker Verification
FunASR's SenseVoiceSmall offers Speaker Verification which is crucial for detecting user speakin when the assistant is speaking at the same time.

## LLM
The transcribed text is then send to a LLM to generate response. This part is currently powered by Ollama. The response from many LLM tends to be verbose. 
When the interaction is purely text form, humans can just skip a few lines. If the LLM is equiped with audio ability, then it could be annoying to wait for the audio to finish to continue next interaction
This is also the main purpose of this project to be able to interrupt the assistant at any time.

## Text-to-Speech
Currently is powered by kokoro. 
