import torch
import transformers
import datetime
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Define function to summarize text using BART model
def summarize_with_bart(text):
    try:
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    except:
        summary = "Error"
    return summary

# Define function to convert seconds to HH:MM:SS format
def convert_seconds_to_hh_mm_ss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

# Define chunking function
def split_text_into_chunks(text, chunk_size=300):
    sentences = text.split('.')
    chunks = []
    current_chunk = ''
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count += len(words)

        if word_count <= chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
            word_count = len(words)

    if current_chunk and current_chunk != '. ':
        chunks.append(current_chunk.strip())

    return chunks

# Load BART model for summarization
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device="cuda:0",  # Change this to the appropriate device index
)

# Load JSON data
json_file_path = 'MeetingBank.json'
json_output_path = 'output1_bart.json'

with open(json_file_path, "r") as file:
    json_data = json.load(file)


# Process each item in the JSON data
for field, field_data in tqdm(json_data.items(), desc="Processing items"):
    for item_id, item_info in field_data.get("itemInfo", {}).items():
        try:
            # Check if BART summary already exists
            if "bart_summary" not in item_info:
                if "transcripts" in item_info:
                    concatenated_text = ""
                    for transcript in item_info["transcripts"]:
                        concatenated_text += transcript["text"] + " "

                    # Convert start time to HH:MM:SS format
                    start_time_seconds = float(item_info.get("startTime", 0))
                    start_time = convert_seconds_to_hh_mm_ss(start_time_seconds)

                    # Convert end time to HH:MM:SS format
                    end_time_seconds = float(item_info.get("endTime", 0))
                    end_time = convert_seconds_to_hh_mm_ss(end_time_seconds)

                    # Calculate total duration
                    total_duration_seconds = end_time_seconds - start_time_seconds
                    total_duration_hh_mm_ss = convert_seconds_to_hh_mm_ss(total_duration_seconds)

                    # Split text into chunks
                    chunks = split_text_into_chunks(concatenated_text)

                    # Generate summaries for each chunk
                    bart_summaries = []
                    for chunk in chunks:
                        bart_summary = summarize_with_bart(chunk)
                        bart_summaries.append(bart_summary)

                    # Concatenate summaries of all chunks
                    final_summary = ' '.join(bart_summaries)

                    # Update item_info with BART summary and other details
                    item_info["bart_summary"] = final_summary
                    item_info["start_time"] = start_time
                    item_info["end_time"] = end_time
                    item_info["total_duration_hh_mm_ss"] = total_duration_hh_mm_ss

                    # Save updated JSON data
                    with open(json_output_path, 'w') as file:
                        json.dump(json_data, file, indent=4)
                        print(f"BART summary saved for item {item_id}")
            else:
                print(f"BART summary already exists for {item_id}")
        except Exception as e:
            print(f"Error processing item {item_id}: {str(e)}")

print("All items processed")