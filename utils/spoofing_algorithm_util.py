
import os
import shutil


# /root/Partial_Spoof_Detection_System/database/ASVspoof2019/LA/spoofing_algorithm/bonafide

# /root/Partial_Spoof_Detection_System/database/ASVspoof2019/LA/spoofing_algorithm/A07

# ====== PATHS ======
metadata_file = "/root/Partial_Spoof_Detection_System/database/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"        # file containing the lines you pasted
source_audio_dir = "/root/Partial_Spoof_Detection_System/database/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac"
output_dir = "/root/Partial_Spoof_Detection_System/database/ASVspoof2019/LA/spoofing_algorithm"

os.makedirs(output_dir, exist_ok=True)

# ====== READ METADATA ======
with open(metadata_file, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()

    # Example line:
    # LA_0039 LA_E_2834763 - A11 spoof
    # LA_0030 LA_E_5849185 - - bonafide

    file_id = parts[1]                 # LA_E_2834763
    algorithm = parts[3]               # A11 or '-'
    label = parts[4]                   # spoof or bonafide

    src_file = os.path.join(source_audio_dir, file_id + ".flac")

    if not os.path.exists(src_file):
        print(f"Missing file: {src_file}")
        continue

    # ====== DETERMINE DESTINATION ======
    if label == "bonafide":
        dst_dir = os.path.join(output_dir, "bonafide")
    else:
        dst_dir = os.path.join(output_dir, algorithm)

    os.makedirs(dst_dir, exist_ok=True)

    dst_file = os.path.join(dst_dir, file_id + ".flac")

    # ====== COPY FILE ======
    shutil.copy2(src_file, dst_file)

print("✅ Copying completed.")

