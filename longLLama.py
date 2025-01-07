import re
import csv
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

torch.cuda.set_device(1)
device = torch.device("cuda:1")

from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("syzymon/long_llama_3b")
model = AutoModelForCausalLM.from_pretrained("syzymon/long_llama_3b", torch_dtype=torch.float32,
    mem_layers=[],
    mem_dtype='bfloat16',
    trust_remote_code=True,
    mem_attention_grouping=(4, 2048))
model = model.to(device)

path = 'LLAMA_MeetingBank.json'

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


# Function to summarize text using LLAMA T5 Large
def summarize_with_LLAMA(text):
    # Tokenize input text
    text = """Summarize this transcript: """ + text + """
        Concise Summary: """
    try:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda:1')
                # outputs = model(input_ids=input_ids)
        generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                num_beams=1,
                last_context_length=1792,
                do_sample=True,
                temperature=1.0,
                )
                # input('enter')
        response = tokenizer.decode(generation_output[0])
    except:
        print("Error")
        response = "Error"
        
    return response

# Function to recursively summarize chunks of text
def summarize_chunks(chunks):
    summarized_chunks = []
    for chunk in chunks:
        summary = summarize_with_LLAMA(chunk)
        summarized_chunks.append(summary)
    return summarized_chunks


# Function to summarize text with chunking
def summarize_with_chunking(text):
    chunks = split_text_into_chunks(text, chunk_size=300)
    summarized_chunks = summarize_chunks(chunks)
    final_summary = summarize_with_LLAMA(" ".join(summarized_chunks))
    return final_summary


json_file = path
json_file2 = 'LLAMA_MeetingBank.json'

with open(json_file, "r") as file:
    json_data = json.load(file)

for field, field_data in json_data.items():
    for item_id, item_info in field_data.get("itemInfo", {}).items():
        if "LLAMA_summary" not in item_info:
            if "transcripts" in item_info:

                concatenated_text = ""
                for transcript in item_info["transcripts"]:
                    concatenated_text += transcript["text"] + " "
                LLAMA_summary = summarize_with_chunking(concatenated_text)
                item_info["LLAMA_summary"] = LLAMA_summary
                print(LLAMA_summary)
                with open(json_file2, 'w') as file:
                    json.dump(json_data, file, indent=4)
                    print(f"LLAMA summary saved for item {item_id}")
        else:
            print(f"LLAMA summary already exists for {item_id}")