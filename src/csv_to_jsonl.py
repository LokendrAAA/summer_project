import pandas as pd
import json

# Load your CSV
df = pd.read_csv("data/emotion-emotion_69k.csv")

# Output list
output = []

for _, row in df.iterrows():
    context = f"Situation: {row['Situation']}\nEmotion: {row['emotion']}\nDialogue: {row['empathetic_dialogues']}"
    # Optional: include 'labels' if needed
    output.append({
        "Context": context.strip(),
        "Response": str(row['labels']) if pd.notna(row['labels']) else ""
    })

# Save as JSONL
with open("empathetic_dialogues_prepared.jsonl", "w", encoding='utf-8') as f:
    for entry in output:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("✅ Conversion complete.")
