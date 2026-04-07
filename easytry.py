import torch
import time
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset
from collections import defaultdict

# ==========================================
# 1. SETTINGS & CONFIGURATION
# ==========================================
# Change this to the path of your downloaded Easy-Turn model, 
# or the Hugging Face model ID if it's hosted there.
MODEL_ID_OR_PATH = "openai/whisper-large-v2" # <--- UPDATE THIS TO EASY-TURN MODEL
DATASET_ID = "ASLP-lab/Easy-Turn-Testset"
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
        print("WARNING: Running on CPU. Latency and Memory metrics will not be accurate for the table.")

    print(f"Loading processor and model from {MODEL_ID_OR_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID_OR_PATH)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID_OR_PATH).to(DEVICE)
    model.eval()

    # ==========================================
    # 2. CALCULATE PARAMS (MB)
    # ==========================================
    # get_memory_footprint() returns the physical size of the model weights in bytes
    params_mb = model.get_memory_footprint() / (1024 * 1024)
    print(f"Model loaded. Params: {params_mb:.0f} MB")

    # ==========================================
    # 3. LOAD DATASET
    # ==========================================
    print(f"Downloading/Loading dataset from {DATASET_ID}...")
    # 'test' is typically the default split name, check HF page if it differs
    dataset = load_dataset(DATASET_ID, split="test") 
    print(f"Found {len(dataset)} audio files.")

    # ==========================================
    # 4. RUN INFERENCE & HARDWARE TRACKING
    # ==========================================
    total_latency_ms = 0
    tag_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    # Reset PyTorch CUDA memory tracking before we start
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("Starting inference loop...")
    for i, item in enumerate(dataset):
        # 1. Prepare Audio
        audio_array = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        ground_truth_text = item['text'] # Assuming 'text' is the column name for transcript

        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").to(DEVICE)

        # 2. Measure Latency
        # Use torch.cuda.synchronize() to ensure GPU operations finish before stopping the timer
        if DEVICE == "cuda": torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"], max_new_tokens=256)
            
        if DEVICE == "cuda": torch.cuda.synchronize()
        end_time = time.time()

        # Add to total latency
        total_latency_ms += (end_time - start_time) * 1000

        # 3. Decode Text
        # skip_special_tokens=False is CRITICAL so the <COMPLETE> tags aren't deleted by the tokenizer
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # 4. Check Turn-Taking Accuracy
        ref_tag = extract_tag(ground_truth_text)
        hyp_tag = extract_tag(transcription)

        if ref_tag != "<UNKNOWN>":
            tag_stats[ref_tag]['total'] += 1
            if ref_tag == hyp_tag:
                tag_stats[ref_tag]['correct'] += 1

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(dataset)} files...")

    # ==========================================
    # 5. CALCULATE FINAL METRICS
    # ==========================================
    # Average Latency
    avg_latency = total_latency_ms / len(dataset)

    # Peak Memory
    peak_memory_mb = 0
    if DEVICE == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Calculate Accuracies
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