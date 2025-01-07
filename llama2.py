import re
import json
import transformers
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

import torch
import time
from transformers import LlamaForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoTokenizer
import json
import datetime
from transformers import pipeline



model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:3",
    max_length=4098,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

def summarize_with_llama2(text):
    template = """
              Given the transcript of meeting conversation, write a concise summary covering all the important events in the meeting.
              MeetingTranscript: {text}
              Summary:
            """
    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    try:
        response = llm_chain.run(text)
    except:
        response = "Error"
        print("Error")
    return response

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

# Load your data from JSON
json_file_path = 'MeetingBank.json'
json_output_path = 'output_llamam2_MeetingBank1.json'

with open(json_file_path, "r") as file:
    json_data = json.load(file)

for field, field_data in json_data.items():
    for item_id, item_info in field_data.get("itemInfo", {}).items():
        if "llama2_summary" not in item_info:
            if "transcripts" in item_info:
                concatenated_text = ""
                for transcript in item_info["transcripts"]:
                    concatenated_text += transcript["text"] + " "

                start_time = item_info.get("start_time", "00:00:00")
                end_time = item_info.get("end_time", "00:00:00")
                total_duration = datetime.datetime.strptime(end_time, "%H:%M:%S") - datetime.datetime.strptime(start_time, "%H:%M:%S")
                # total_duration_hh_mm = total_duration.strftime("%H:%M")

                llama2_summary = summarize_with_llama2(concatenated_text)
                item_info["llama2_summary"] = llama2_summary
                item_info["start_time"] = start_time
                item_info["end_time"] = end_time
                item_info["total_duration_hh_mm"] = str(total_duration)
                print("LLama2 Summary:", llama2_summary)
                with open(json_output_path, 'w') as file:
                    json.dump(json_data, file, indent=4)
                    print(f"LLama2 summary saved for item {item_id}")
        else:
            print(f"LLama2 summary already exists for {item_id}")