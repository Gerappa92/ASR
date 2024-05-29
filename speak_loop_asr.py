import sounddevice as sd
import soundfile as sf
import torch
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# The sample rate defines how many samples of the audio are taken per second
sample_rate = 16000  
duration = 3  # The duration of the recording in seconds

def record_audio(duration, sample_rate):
    my_recording = sd.rec(int(duration * sample_rate), 
                         samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    return my_recording

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny"

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

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

while True:
    try: 
        input("Press enter to record")
        print("Recording...")
        time.time
        # record
        my_recording = record_audio(duration, sample_rate)
        # This will save the recording in a .wav file
        sf.write('my_recording.wav', my_recording, sample_rate)
        # Text recognition
        start_time = time.time()
        result = pipe("my_recording.wav")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("You said: " + result["text"])
        print(f"Recognized in: {elapsed_time} seconds")

    except KeyboardInterrupt:
        print("\nLoop interrupted by user")
        break