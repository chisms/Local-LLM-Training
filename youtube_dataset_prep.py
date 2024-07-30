import yt_dlp
import os
import ffmpeg
import whisper
import json
import re
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt')

OUTPUT_DIR = 'processed_videos'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_video(url):
    ydl_opts = {'outtmpl': '%(id)s.%(ext)s'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info['id'], info.get('title', 'Unknown')

def extract_audio(video_id):
    # input_file = f"{video_id}.mp4"
    input_file = f"{video_id}.webm"
    output_file = f"{video_id}.wav"
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_file)
    ffmpeg.run(stream)
    return output_file

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

def video_id_from_url(url):
    return yt_dlp.YoutubeDL().extract_info(url, download=False)['id']

def save_full_transcript(transcript, video_id):
    with open(f"{video_id}_full_transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove timestamps and speaker labels
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def segment_text(text):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 1000:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def save_dataset(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def main(url):
    print("Downloading video...")
    video_id, title = download_video(url)
    
    print("Extracting audio...")
    audio_file = extract_audio(video_id)
    
    print("Transcribing audio...")
    transcript = transcribe_audio(audio_file)

    save_full_transcript(transcript, video_id)
    
    print("Cleaning and preprocessing text...")
    cleaned_text = clean_text(transcript)
    
    print("Segmenting text...")
    chunks = segment_text(cleaned_text)
    
    print("Preparing dataset...")
    dataset = [{"text": chunk} for chunk in chunks]
    
    print("Splitting dataset...")
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    print("Saving datasets...")
    save_dataset(train_data, os.path.join(OUTPUT_DIR, f"{video_id}_train.json"))
    save_dataset(val_data, os.path.join(OUTPUT_DIR, f"{video_id}_val.json"))
    save_dataset(test_data, os.path.join(OUTPUT_DIR, f"{video_id}_test.json"))
    
    print("Dataset preparation complete!")

    return train_data

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your video URL
    main(video_url)