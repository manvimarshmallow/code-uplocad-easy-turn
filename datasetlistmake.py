import os
import json

# Define paths based on your folder structure
dataset_dir = os.path.abspath("datasets/Easy-Turn-Testset")
output_file = os.path.abspath("datasets/data.list")

# Mapping folder names to the exact tags your model expects
tag_map = {
    "backchannel": "<BACKCHANNEL>",
    "complete": "<COMPLETE>",
    "incomplete": "<INCOMPLETE>",
    "wait": "<WAIT>"
}

count = 0
with open(output_file, 'w', encoding='utf-8') as f_out:
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                
                # Determine the tag by checking if folder name is in the path
                tag = "<UNKNOWN>"
                for key_folder, actual_tag in tag_map.items():
                    if f"/{key_folder}/" in wav_path.replace("\\", "/"):
                        tag = actual_tag
                        break
                
                # Create a unique key (the filename without .wav)
                key = file.replace(".wav", "")
                
                # We put the tag in the txt field so eval scripts can check it later
                txt = f"评价文本{tag}" 
                
                # Create WeNet JSON format
                record = {
                    "key": key,
                    "wav": wav_path,
                    "txt": txt
                }
                
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

print(f"Successfully created {output_file} with {count} audio files!")