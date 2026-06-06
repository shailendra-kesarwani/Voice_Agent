from pydub import AudioSegment
import os

def convert_mp3_chunks_to_wav(input_folder, output_folder, chunk_length_ms=5000):
    """
    Converts MP3 audio chunks in an input folder to WAV audio chunks in an output folder.

    Args:
        input_folder (str): Path to the folder containing MP3 audio chunks.
        output_folder (str): Path to the folder to save WAV audio chunks.
        chunk_length_ms (int): Desired length of each chunk in milliseconds (default: 5000ms = 5 seconds).
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(input_folder, filename)
            
            try:
                audio = AudioSegment.from_mp3(mp3_path)
            except Exception as e:
                print(f"Error loading {mp3_path}: {e}")
                continue

            # Split the audio into chunks
            for i, chunk in enumerate(audio[::chunk_length_ms]):
                chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{i:03d}.wav"
                wav_path = os.path.join(output_folder, chunk_filename)
                
                try:
                    chunk.export(wav_path, format="wav")
                    print(f"Exported: {wav_path}")
                except Exception as e:
                    print(f"Error exporting {wav_path}: {e}")

# Example usage:
input_directory = "mp3_chunks"  # Folder containing your MP3 chunks
output_directory = "wav_chunks"  # Folder to save the converted WAV chunks
convert_mp3_chunks_to_wav(input_directory, output_directory, chunk_length_ms=10000) # 10-second chunks