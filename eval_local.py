import torch
import time
import os
import sys
import yaml
import librosa
from huggingface_hub import hf_hub_download
from collections import defaultdict

# --- Add the project's source code to the Python path ---
# This allows us to import from the 'Easy_Turn' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
# ---

from Easy_Turn.model.whisper_model import WhisperModel
from Easy_Turn.utils.tokenizer import Tokenizer

# ==========================================
# 1. SETTINGS & CONFIGURATION
# ==========================================
# Path to the dataset you downloaded earlier
DATASET_PATH = os.path.join(script_dir, "datasets", "Easy-Turn-Testset")
# Path to the config file within the cloned GitHub repo
CONFIG_PATH = os.path.join(script_dir, "examples", "wenetspeech", "whisper", "conf", "whisper_st_w_context.yaml")
# Hugging Face repo for the raw checkpoint
CHECKPOINT_REPO = "ASLP-lab/Easy-Turn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The 4 target turn-taking tags
TAGS = ['<COMPLETE>', '<INCOMPLETE>', '<BACKCHANNEL>', '<WAIT>']

def extract_tag(text):
    """Finds the Easy-Turn tag in the predicted text."""
    text = text.upper()
    for tag in TAGS:
        if tag in text:
            return tag
    return "<UNKNOWN>"

def main():
    if DEVICE == "cpu":
        print("WARNING: Running on CPU. Latency and Memory will not be accurate for the table.")

    # ==========================================
    # 2. LOAD MODEL MANUALLY
    # ==========================================
    print("Loading model configuration...")
    with open(CONFIG_PATH, 'r') as f:
        configs = yaml.safe_load(f)

    # 1. Build the model architecture from the project's own code
    model_conf = configs['model_conf']
    model = WhisperModel(
        model_conf['whisper_model_name'],
        model_conf['dropout'],
        model_conf['add_context'],
        model_conf['use_conv_context']
    )
    
    # 2. Download the raw PyTorch checkpoint
    print(f"Downloading checkpoint from '{CHECKPOINT_REPO}'...")
    checkpoint_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename="checkpoint.pt")
    
    # 3. Load the weights into the model
    print("Loading weights into model...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    # Create the tokenizer
    tokenizer = Tokenizer()

    # Calculate Params (MB)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    params_mb = param_bytes / (1024 * 1024)
    print(f"Model loaded successfully. Params: {params_mb:.0f} MB")

    # ==========================================
    # 3. LOAD DATASET FROM LOCAL FOLDER
    # ==========================================
    print(f"\nLoading dataset from local path: {DATASET_PATH}...")
    audio_files = [] 
    text_dict = {}   

    print("Scanning subfolders for audio files...")
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".wav"):
                utt_id = file.replace(".wav", "")
                audio_path = os.path.join(root, file)
                txt_path = os.path.join(root, f"{utt_id}.txt")
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text_dict[utt_id] = f.read().strip()
                else:
                    folder_name = os.path.basename(root).upper()
                    text_dict[utt_id] = f"TEXT_NOT_AVAILABLE <{folder_name}>"
                audio_files.append((utt_id, audio_path))

    print(f"Found {len(audio_files)} audio files to process.")
    
    # ==========================================
    # 4. RUN INFERENCE & HARDWARE TRACKING
    # ==========================================
    total_latency_ms = 0
    tag_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\nStarting inference loop...")
    for i, (utt_id, audio_path) in enumerate(audio_files):
        try:
            audio_array, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Skipping {utt_id} due to audio load error: {e}")
            continue
            
        ground_truth_text = text_dict.get(utt_id, "")
        
        # Manually process audio
        audio_tensor = torch.from_numpy(audio_array).to(DEVICE).unsqueeze(0)

        # Measure Latency
        if DEVICE == "cuda": torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            # Use the model's custom decoding method
            hyp = model.decode(audio_tensor, tokenizer)
            
        if DEVICE == "cuda": torch.cuda.synchronize()
        end_time = time.time()

        total_latency_ms += (end_time - start_time) * 1000

        # The output `hyp` is already a text string
        transcription = hyp[0]

        if i < 3:
            print(f"\n--- Sample {i+1} ({utt_id}) ---")
            print(f"Ref: {ground_truth_text}")
            print(f"Hyp: {transcription}")

        # Check Turn-Taking Accuracy
        ref_tag = extract_tag(ground_truth_text)
        hyp_tag = extract_tag(transcription)

        if ref_tag != "<UNKNOWN>":
            tag_stats[ref_tag]['total'] += 1
            if ref_tag == hyp_tag:
                tag_stats[ref_tag]['correct'] += 1

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(audio_files)} files...")

    # ==========================================
    # 5. CALCULATE FINAL METRICS
    # ==========================================
    avg_latency = total_latency_ms / len(audio_files) if len(audio_files) > 0 else 0
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if DEVICE == "cuda" else 0

    def calc_acc(tag_name):
        data = tag_stats.get(tag_name, {'total': 0, 'correct': 0})
        if data['total'] == 0: return "N/A"
        return f"{(data['correct'] / data['total']) * 100:.2f}"

    acc_cp = calc_acc('<COMPLETE>')
    acc_incp = calc_acc('<INCOMPLETE>')
    acc_bc = calc_acc('<BACKCHANNEL>')
    acc_wait = calc_acc('<WAIT>')

    # ==========================================
    # 6. PRINT RESULTS IN TABLE FORMAT
    # ==========================================
    print("\n" + "="*80)
    print("FINAL ABLATION TABLE METRICS")
    print("="*80)
    print(f"Params(MB) : {params_mb:.0f}")
    print(f"Latency(ms): {avg_latency:.0f}")
    print(f"Memory(MB) : {peak_memory_mb:.0f}")
    print("-" * 30)
    print(f"ACC_cp(%)   : {acc_cp}")
    print(f"ACC_incp(%) : {acc_incp}")
    print(f"ACC_bc(%)   : {acc_bc}")
    print(f"ACC_wait(%) : {acc_wait}")
    print("="*80)

if __name__ == "__main__":
    main()