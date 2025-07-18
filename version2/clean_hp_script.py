import re

INPUT_FILE = 'hp_full_script.txt'
OUTPUT_FILE = 'hp_full_script_clean.txt'

cleaned_lines = []

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        # Remove [tags] and (parenthetical directions)
        line = re.sub(r'\[.*?\]', '', line)
        line = re.sub(r'\(.*?\)', '', line)
        # Remove leading/trailing whitespace
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        cleaned_lines.append(line)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for line in cleaned_lines:
        f.write(line + '\n')

print(f"Cleaned script saved to {OUTPUT_FILE} ({len(cleaned_lines)} lines)") 