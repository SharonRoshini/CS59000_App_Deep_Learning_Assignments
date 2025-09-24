import gradio as gr
import torch
from transformers import pipeline

# Load Whisper small pipeline
device = 0 if torch.cuda.is_available() else -1
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    device=device,
)

def transcribe(audio):
    if audio is None:
        return "Please upload or record audio"
    text = asr(audio)["text"]
    return text

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
    title="Whisper Small ASR",
    description="Upload or record audio and get text transcription with OpenAI Whisper Small."
)

if __name__ == "__main__":
    demo.launch()
