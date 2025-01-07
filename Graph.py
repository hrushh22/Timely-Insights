import matplotlib.pyplot as plt
import pandas as pd

# Create a dataframe from the table
df = pd.DataFrame({
    'Model': ['BART', 'FLAN-T5', 'GEMMA', 'GPT-3.5', 'Long-LLAMA', 'LLAMA2'],
    'Concat F1_Rouge 1': [0.218, 0.289, 0.236, 0.334, 0.324, 0.471],
    'Concat F1_Rouge 2': [0.082, 0.123, 0.109, 0.161, 0.161, 0.223],
    'Agree F1_Rouge 1': [0.210, 0.238, 0.199, 0.328, 0.278, 0.307],
    'Agree F1_Rouge 2': [0.079, 0.111, 0.136, 0.168, 0.118, 0.161]
})

# Line graph for Concat F1
plt.figure(figsize=(10, 6))
plt.plot(df['Model'], df['Concat F1_Rouge 1'], marker='o', label='Concat F1_Rouge 1')
plt.plot(df['Model'], df['Concat F1_Rouge 2'], marker='o', label='Concat F1_Rouge 2')
plt.xlabel('Model')
plt.ylabel('Concat F1')
plt.title('Concat F1 for Different Models')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Line graph for Agree F1
plt.figure(figsize=(10, 6))
plt.plot(df['Model'], df['Agree F1_Rouge 1'], marker='o', label='Agree F1_Rouge 1')
plt.plot(df['Model'], df['Agree F1_Rouge 2'], marker='o', label='Agree F1_Rouge 2')
plt.xlabel('Model')
plt.ylabel('Agree F1')
plt.title('Agree F1 for Different Models')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()