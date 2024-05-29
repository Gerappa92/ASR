# Script which allow recording the speaker and after that transform the audio to text.
# Audio to text know also as a ASR (Automatic Speech Recognition) is completed by openai/whisper-tiny model (https://huggingface.co/openai/whisper-tiny)
# Loading the model and tokenizer and using pretrained model is done with HuggingFace interfaces (https://huggingface.co/docs/transformers/quicktour)

# library for recoding sound
# require FFmpeg (https://ffmpeg.org/download.html)
# can be installed with Chocolatey
import sounddevice as sd
# library to save the recoding
import soundfile as sf
# library for tensor manipulations
import torch
# library to mesuare ASR time
import time
# HuggingFace interface
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# The sample rate defines how many samples of the audio are taken per second
# 16000 is the required rate by our model
sample_rate = 16000  
duration = 3  # The duration of the recording in seconds


def record_audio(duration, sample_rate):
    my_recording = sd.rec(int(duration * sample_rate), 
                         samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    return my_recording

# define if use GPU or CPU (https://pytorch.org/get-started/locally/#windows-verification)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny"

# (https://huggingface.co/docs/transformers/v4.41.2/en/model_doc/auto#transformers.AutoModel.from_pretrained)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Infinit loop allows to interativly record and recognize speech
while True:
    try: 
        # Wait for user interaction
        input("Press enter to record")
        print("Recording...")

        # Start recording the speaker
        my_recording = record_audio(duration, sample_rate)

        # This will save the recording in a .wav file
        sf.write('my_recording.wav', my_recording, sample_rate)
        print("Recording finished...")

        # Text recognition
        print("Speach recognition started")

        # Measure the speech recognition
        start_time = time.time()

        result = pipe("my_recording.wav")

        print("You said: " + result["text"])

        elapsed_time = time.time() - start_time
        print(f"Recognized in: {elapsed_time} seconds")

    except KeyboardInterrupt:
        print("\nLoop interrupted by user")
        break