# AI-Disability-Educator: Bridging Learning Gaps for Children with Disabilities

## Project Description
This project is an innovative web application designed to enhance the learning experience for students with disabilities. It provides real-time speech-to-text transcription of lectures, text-to-speech capabilities, and AI-assisted question answering. The project aims to create an inclusive classroom environment where students with visual, hearing, or speaking impairments can participate fully and effectively.

## Bill of Materials (BOM)
- Computer or server to host the application
- Microphone for speech input (for the teacher interface)
- Speakers or headphones for audio output (for students)
- Web browser-enabled devices for students and teachers

## Software Requirements
- Python 3.8+
- Flask
- Flask-SocketIO
- PyTorch
- Transformers (Hugging Face)
- SpeechT5 (for Text-to-Speech)
- Whisper (for Speech-to-Text)
- Ollama (for AI-assisted question answering)
- jQuery
- Socket.IO

## Full Instructions

### Setup and Installation on Linux

1. Clone the repository:
   git clone https://github.com/seanlin97/AI-Disability-Educator.git
   cd AMD-Project-Sean

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  

3. Install the required packages:
   pip install -r requirements.txt

4. Set up Ollama:
   - Follow the installation instructions at Ollama's GitHub repository (https://github.com/ollama/ollama)
   - Ensure the Ollama service is running

5. Start the Flask application:
   python app.py

6. Access the application by opening a web browser and navigating to http://localhost:5000

## Usage

1. On the welcome page, select the appropriate interface (Visual Disability, Hearing Disability, Speaking Disability, or Teacher).

2. For the Teacher Interface:
   - Click "Start Lecture" to begin real time speech-to-text transcription.
   - The transcript will appear in the left box.
   - Student questions will appear in the right box.
   - Click "End Lecture" when finished.

3. For Student Interfaces:
   - The lecture transcript appears in the left box in real time.
   - Type questions in the input area at the bottom.
   - Click "Ask Question" to send the question to the teacher's interface.
   - Click "Ask LLM" to get an AI-generated answer (appears in the right box).
   - For speaking-impaired students, use the "Speak Text" button to have the question read aloud in class.
   - For visually-impaired students, use the "Stop Reading" button to stop reading the response from LLM.

## Acknowledgements
This project would not have been possible without the support and contributions of many individuals and organizations. First and foremost, I would like to express my sincere gratitude to AMD and Hackster.io for organizing the AMD 2023 contest and for providing me with the AMD Ryzen AI PC, which was instrumental in the development of this project. I would also like to thank the open-source community for their incredible work. The project leverages several open-source libraries and models, and I am deeply grateful to the developers and contributors of Flask, PyTorch, Hugging Face Transformers, Whisper, and Ollama. Their dedication and expertise have significantly enhanced the capabilities of this project.

## Notice
This project was developed and tested on Linux using a self-rented AMD GPU with ROCm. Due to the Linux environment, AMD Ryzen™ AI Software, which is compatible only with Windows 11, was not utilized directly. However, the project's functionality benefits from AMD Ryzen™ AI capabilities, including PyTorch and models from Hugging Face. Future integration with AMD Ryzen™ AI Software is feasible.