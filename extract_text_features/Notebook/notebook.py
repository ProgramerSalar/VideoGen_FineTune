import json
import os

# --- CONFIGURATION ---
input_file = "/home/manish/Desktop/projects/videoGen_fineTune/annotation/test_video_annotation.jsonl"   # Your current (broken) file
output_file = "fixed_data.jsonl"   # The new clean file
# ---------------------

print(f"Fixing paths in {input_file}...")

valid_count = 0
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:

    for line in infile:
        try:
            data = json.loads(line)
            
            # 1. Get the source video path
            # Example: "./clip_video/.../clip_0000.mp4"
            video_path = data.get('video', '')
            
            if not video_path:
                print("Skipping line with no video path.")
                continue

            # 2. Create a valid output path based on the VIDEO, not the TEXT
            # Example: "./clip_video/.../clip_0000.pt"
            base, ext = os.path.splitext(video_path)
            new_latent_path = base + ".pt"
            
            # 3. Overwrite the bad fields with this correct path
            data['video_latent'] = new_latent_path
            data['text_latent'] = new_latent_path
            data['text_fea'] = new_latent_path  # setting all just to be safe
            
            # 4. Write to new file
            outfile.write(json.dumps(data) + '\n')
            valid_count += 1
            
        except json.JSONDecodeError:
            continue

print(f"Success! {valid_count} lines fixed.")
print(f"Please point your training script to: {output_file}")