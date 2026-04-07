import torch
import time
import os
import librosa
from huggingface_hub import snapshot_download
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from collections import defaultdict

# ==========================================
# 1. SETTINGS & CONFIGURATION
# ==========================================
MODEL_ID = "ASLP-lab/Easy-Turn"
DATASET_ID = "ASLP-lab/Easy-Turn-Testset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The 4 target turn-taking tags
TAGS =['<COMPLETE>', '<INCOMPLETE>', '<BACKCHANNEL>', '<WAIT>']

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
    # 2. LOAD MODEL (WITH CONFIG WORKAROUND)
    # ==========================================
    print(f"Loading processor and model from {MODEL_ID}...")
    
    # 1. Bypass the broken config.json by borrowing the base Whisper Large V2 config
    base_config = WhisperConfig.from_pretrained("openai/whisper-large-v2")
    
    # 2. Use explicit Whisper classes to safely load the Easy-Turn tokenizer/processor
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    
    # 3. CRITICAL: Update the config's vocab size to match the new tags (like <COMPLETE>) added by Easy-Turn
    base_config.vocab_size = len(processor.tokenizer)
    
    # 4. Load the actual model weights using our patched config
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, config=base_config).to(DEVICE)
    model.eval()

    # Calculate Params (MB)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    params_mb = param_bytes / (1024 * 1024)
    print(f"Model loaded successfully. Params: {params_mb:.0f} MB")

    # ==========================================
    # 3. LOAD DATASET (MANUAL PARSING)
    # ==========================================
    print(f"\nDownloading raw dataset from {DATASET_ID}...")
    dataset_path = snapshot_download(repo_id=DATASET_ID, repo_type="dataset")
    testset_dir = os.path.join(dataset_path, "testset")
    
    if not os.path.exists(testset_dir):
        testset_dir = dataset_path # Fallback if there is no 'testset' subfolder

    audio_files =[] # list of tuples: (utt_id, file_path)
    text_dict = {}   # dict mapping: utt_id -> ground_truth_text

    wav_scp = os.path.join(testset_dir, "wav.scp")
    text_file = os.path.join(testset_dir, "text")
    
    if os.path.exists(wav_scp) and os.path.exists(text_file):
        print("Detected WeNet/Kaldi format (wav.scp + text) - Parsing...")
        with open(wav_scp, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, rel_path = parts[0], parts[1]
                    abs_path = os.path.join(testset_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
                    audio_files.append((utt_id, abs_path))
                    
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    text_dict[parts[0]] = parts[1]
    else:
        print("No wav.scp found, looking for side-by-side .wav and .txt files...")
        for file in os.listdir(testset_dir):
            if file.endswith(".wav"):
                utt_id = file.replace(".wav", "")
                audio_path = os.path.join(testset_dir, file)
                txt_path = os.path.join(testset_dir, f"{utt_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text_dict[utt_id] = f.read().strip()
                else:
                    text_dict[utt_id] = "<UNKNOWN>"
                audio_files.append((utt_id, audio_path))

    print(f"Found {len(audio_files)} audio files to process.")
    if len(audio_files) == 0:
        print(f"ERROR: Could not find audio files in {testset_dir}")
        return

    # ==========================================
    # 4. RUN INFERENCE & HARDWARE TRACKING
    # ==========================================
    total_latency_ms = 0
    tag_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\nStarting inference loop...")
    for i, (utt_id, audio_path) in enumerate(audio_files):
        # 1. Load and resample audio to 16kHz on-the-fly
        try:
            audio_array, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Skipping {utt_id} due to audio load error: {e}")
            continue
            
        ground_truth_text = text_dict.get(utt_id, "")
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").to(DEVICE)

        # 2. Measure Latency
        if DEVICE == "cuda": torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            # max_new_tokens is set to handle the text length + turn tag
            generated_ids = model.generate(inputs["input_features"], max_new_tokens=256)
            
        if DEVICE == "cuda": torch.cuda.synchronize()
        end_time = time.time()

        total_latency_ms += (end_time - start_time) * 1000

        # 3. Decode Text
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        if i < 3:
            print(f"\n--- Sample {i+1} ({utt_id}) ---")
            print(f"Ref: {ground_truth_text}")
            print(f"Hyp: {transcription}")

        # 4. Check Turn-Taking Accuracy
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

    peak_memory_mb = 0
    if DEVICE == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

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