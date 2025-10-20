non_elephant_wav = r"C:\Users\Mitali brahmankar\Downloads\elephant_sound_project\data\non_elephant_sound_wav"

for filename in os.listdir(non_elephant_wav):
    if filename.endswith(".wav"):
        file_path = os.path.join(non_elephant_wav, filename)
        print(f"Loading {filename}...")
        y, sr = librosa.load(file_path, sr=44100)
        print(f"Loaded {filename} successfully.")
