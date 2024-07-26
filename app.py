from flask import Flask, render_template, request, jsonify
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
from datasets import load_dataset
import torch
import soundfile as sf
import io
import base64
from flask import Flask, render_template, request, jsonify, session, Response
from flask_socketio import SocketIO, emit
import os
import requests
import time
import json


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app)

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize TTS models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

# Initialize STT model
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Global variable to store lecture transcript
lecture_transcript = ""

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/student/<disability>')
def student(disability):
    return render_template('student.html', disability=disability)

@app.route('/teacher')
def teacher():
    return render_template('teacher.html')

@socketio.on('transcript_update')
def handle_transcript_update(data):
    global lecture_transcript
    lecture_transcript += data['text'] + " "
    emit('transcript_update', {'text': lecture_transcript}, broadcast=True)

@app.route('/text_to_speech_stream')
def text_to_speech_stream():
    def generate():
        global lecture_transcript
        last_position = 0
        while True:
            if len(lecture_transcript) > last_position:
                new_text = lecture_transcript[last_position:]
                last_position = len(lecture_transcript)
                
                # Generate speech from new text
                inputs = processor(text=new_text, return_tensors="pt").to(device)
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                speech = speech.cpu().numpy()
                
                # Convert to wav format
                byte_io = io.BytesIO()
                sf.write(byte_io, speech, 16000, format='wav')
                yield byte_io.getvalue()
            
            time.sleep(1)  # Wait for 1 second before checking for new text

    return Response(generate(), mimetype="audio/wav")

@socketio.on('student_question')
def handle_student_question(data):
    emit('new_question', data, broadcast=True)


@app.route('/answer_question', methods=['POST'])
def answer_question():
    question = request.json.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Combine lecture transcript and question for context
    context = f"Lecture transcript: {lecture_transcript}\n\nQuestion: {question}"
    
    # Call Ollama API
    response = requests.post('http://localhost:11434/api/generate', 
                             json={
                                 "model": "llama3.1",
                                 "prompt": context
                             },
                             stream=True)
    
    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'response' in json_response:
                    full_response += json_response['response']
                if 'done' in json_response and json_response['done']:
                    break
        return jsonify({'answer': full_response})
    else:
        return jsonify({'error': 'Failed to get answer from LLM'}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    inputs = processor(text=text, return_tensors="pt").to(device)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Move the speech tensor back to CPU before saving to a file
    speech = speech.cpu()
    
    # Save the speech to a BytesIO object
    buffer = io.BytesIO()
    sf.write(buffer, speech.numpy(), samplerate=16000, format='wav')
    buffer.seek(0)
    
    # Convert to base64
    audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return jsonify({'audio': audio_base64})

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    audio_file.save('temp_audio.wav')
    
    # Perform speech recognition
    transcription = stt_pipe("temp_audio.wav")['text']
    
    return jsonify({'text': transcription})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)