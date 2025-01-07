import re
import csv
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def split_text_into_chunks(text, chunk_size=300):
    # Split the text into sentences
    sentences = re.split(r'\.|\n', text)

    chunks = []
    current_chunk = ''
    word_count = 0

    for sentence in sentences:
        # Count the number of words in the sentence
        words = sentence.split()
        word_count += len(words)

        # Add the sentence to the current chunk if it doesn't exceed the chunk size
        if word_count <= chunk_size:
            current_chunk += sentence + '. '
        else:
            # If the chunk size is exceeded, start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
            word_count = len(words)

    # Add the last chunk
    if current_chunk and current_chunk != '. ':
        chunks.append(current_chunk.strip())

    return chunks


# Function to summarize text using Flan T5 Large
def summarize_with_flan(text):
    # Tokenize input text
    text = """Summarize this transcript: """ + text + """
        Concise Summary: """
    inputs = tokenizer(text, return_tensors="pt")
    # Generate summary
    outputs = model.generate(**inputs, max_length=300)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response


# Function to recursively summarize chunks of text
def summarize_chunks(chunks):
    summarized_chunks = []
    for chunk in chunks:
        summary = summarize_with_flan(chunk)
        summarized_chunks.append(summary)
    return summarized_chunks


# Function to summarize text with chunking
def summarize_with_chunking(text):
    chunks = split_text_into_chunks(text, chunk_size=300)
    summarized_chunks = summarize_chunks(chunks)
    final_summary = summarize_with_flan(" ".join(summarized_chunks))
    return final_summary

json_file = "MeetingBank.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

for field, field_data in json_data.items():
    for item_id, item_info in field_data.get("itemInfo", {}).items():
        if "flan_summary" not in item_info:
            if "transcripts" in item_info:

                concatenated_text = ""
                for transcript in item_info["transcripts"]:
                    concatenated_text += transcript["text"] + " "
                flan_summary = summarize_with_chunking(concatenated_text)
                item_info["flan_summary"] = flan_summary
                print(flan_summary)
                with open(json_file, 'w') as file:
                    json.dump(json_data, file, indent=4)
                    print(f"Flan summary saved for item {item_id}")
        else:
            print(f"flan summary already exists for {item_id}")
