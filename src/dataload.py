from datasets import load_dataset

# Load the dataset
dataset = load_dataset("nbertagnolli/counsel-chat")

# The dataset has multiple splits; usually 'train' is the main one
data = dataset['train']

# Convert to pandas DataFrame for easy CSV export
import pandas as pd
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("counsel_chat.csv", index=False, encoding='utf-8')

print("✅ Dataset saved as 'counsel_chat.csv'")
