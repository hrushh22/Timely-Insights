from evaluate import load
import pandas as pd
import json
from IPython.display import display

# Load the ROUGE metric
rouge = load('rouge')

# Load BART JSON data
bart_json_file = "output1_bart_MeetingBank.json"
with open(bart_json_file, "r") as file:
    bart_json_data = json.load(file)

# Initialize a list to store all the details for BART summaries
bart_details_list = []

# Calculate ROUGE scores and accumulate details for BART summaries
for meeting_data in bart_json_data:
    meeting_id = list(meeting_data.keys())[0]
    item_info = meeting_data[meeting_id]["itemInfo"]

    concatenated_bart_summary = ""
    concatenated_summary = ""
    rouge_scores_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}  # Initialize sum of ROUGE scores
    num_items = len(item_info)  # Number of items

    for item_id, item_info in item_info.items():
        bart_summary = item_info.get("bart_summary", "")
        summary = item_info.get("Summary", "")

        # Calculate ROUGE scores for individual summaries
        rouge_scores = rouge.compute(predictions=[bart_summary], references=[[summary]])

        # Accumulate ROUGE scores
        for rouge_type, score in rouge_scores.items():
            rouge_scores_sum[rouge_type] += score  # ROUGE score is stored as a tuple

        # Concatenate summaries
        concatenated_bart_summary += " " + bart_summary
        concatenated_summary += " " + summary

    # Calculate average ROUGE scores
    avg_rouge_scores = {rouge_type: score / num_items for rouge_type, score in rouge_scores_sum.items()}

    # Calculate ROUGE scores for concatenated summaries
    concatenated_rouge_scores = rouge.compute(predictions=[concatenated_bart_summary],
                                              references=[[concatenated_summary]])

    # Append details to the list for BART summaries
    bart_details_list.append({
        'Meeting ID': meeting_id,
        'Number of Items': num_items,
        'Average ROUGE Scores': avg_rouge_scores,
        'Concatenated ROUGE Scores': concatenated_rouge_scores
    })

# Load llama2 JSON data
llama2_json_file = "output_llamam2_MeetingBank1.json"
with open(llama2_json_file, "r") as file:
    llama2_json_data = json.load(file)

# Initialize a list to store all the details for llama2 summaries
llama2_details_list = []

# Calculate ROUGE scores and accumulate details for llama2 summaries
for meeting_data in llama2_json_data:
    meeting_id = list(meeting_data.keys())[0]
    item_info = meeting_data[meeting_id]["itemInfo"]

    concatenated_llama2_summary = ""
    concatenated_summary = ""
    rouge_scores_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}  # Initialize sum of ROUGE scores
    num_items = len(item_info)  # Number of items

    for item_id, item_info in item_info.items():
        llama2_summary = item_info.get("llama2_summary", "")
        summary = item_info.get("Summary", "")

        # Slice the llama2 summary to extract from "Summary:"
        summary_start_index = llama2_summary.find("Summary:")
        llama2_summary_sliced = llama2_summary[summary_start_index + len("Summary:"):]

        # Calculate ROUGE scores for individual summaries
        rouge_scores = rouge.compute(predictions=[llama2_summary_sliced], references=[[summary]])

        # Accumulate ROUGE scores
        for rouge_type, score in rouge_scores.items():
            rouge_scores_sum[rouge_type] += score  # ROUGE score is stored as a tuple

        # Concatenate summaries
        concatenated_llama2_summary += " " + llama2_summary_sliced
        concatenated_summary += " " + summary

    # Calculate average ROUGE scores
    avg_rouge_scores = {rouge_type: score / num_items for rouge_type, score in rouge_scores_sum.items()}

    # Calculate ROUGE scores for concatenated summaries
    concatenated_rouge_scores = rouge.compute(predictions=[concatenated_llama2_summary],
                                              references=[[concatenated_summary]])

    # Append details to the list for llama2 summaries
    llama2_details_list.append({
        'Meeting ID': meeting_id,
        'Number of Items': num_items,
        'Average ROUGE Scores': avg_rouge_scores,
        'Concatenated ROUGE Scores': concatenated_rouge_scores
    })
    

# Load GPT-3.5 JSON data
gpt_json_file = "output1_GPT.json"  # Update this with the path to your GPT-3.5 JSON file
with open(gpt_json_file, "r") as file:
    gpt_json_data = json.load(file)

# Initialize a list to store all the details for GPT-3.5 summaries
gpt_details_list = []

# Calculate ROUGE scores and accumulate details for GPT-3.5 summaries
for meeting_data in gpt_json_data:
    meeting_id = list(meeting_data.keys())[0]
    item_info = meeting_data[meeting_id]["itemInfo"]

    concatenated_Gpt_summary = ""
    concatenated_summary = ""
    rouge_scores_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}  # Initialize sum of ROUGE scores
    num_items = len(item_info)  # Number of items

    for item_id, item_info in item_info.items():
        Gpt_summary = item_info.get("Gpt-3.5_summary", "")
        summary = item_info.get("Summary", "")

        # Calculate ROUGE scores for individual summaries
        rouge_scores = rouge.compute(predictions=[Gpt_summary], references=[[summary]])

        # Accumulate ROUGE scores
        for rouge_type, score in rouge_scores.items():
            rouge_scores_sum[rouge_type] += score  # ROUGE score is stored as a tuple

        # Concatenate summaries
        concatenated_Gpt_summary += " " + Gpt_summary
        concatenated_summary += " " + summary

    # Calculate average ROUGE scores
    avg_rouge_scores = {rouge_type: score / num_items for rouge_type, score in rouge_scores_sum.items()}

    # Calculate ROUGE scores for concatenated summaries
    concatenated_rouge_scores = rouge.compute(predictions=[concatenated_Gpt_summary],
                                              references=[[concatenated_summary]])

    # Append details to the list for GPT-3.5 summaries
    gpt4_details_list.append({
        'Meeting ID': meeting_id,
        'Number of Items': num_items,
        'Average ROUGE Scores': avg_rouge_scores,
        'Concatenated ROUGE Scores': concatenated_rouge_scores
    })



# Load LLAMA JSON data
LLAMA_json_file = "LLAMA_MeetingBank.json"
with open(LLAMA_json_file, "r") as file:
    LLAMA_json_data = json.load(file)
    
LLAMA_details_list = []

# Calculate ROUGE scores and accumulate details for LLAMA summaries
for meeting_data in LLAMA_json_data:
    meeting_id = list(meeting_data.keys())[0]
    item_info = meeting_data[meeting_id]["itemInfo"]

    concatenated_LLAMA_summary = ""
    concatenated_summary = ""
    rouge_scores_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}  # Initialize sum of ROUGE scores
    num_items = len(item_info)  # Number of items

    for item_id, item_info in item_info.items():
        LLAMA_summary = item_info.get("LLAMA_summary", "")
        summary = item_info.get("Summary", "")

        # Slice the LLAMA summary to extract from "Summary:"
        summary_start_index = LLAMA_summary.find("Summary:")
        LLAMA_summary_sliced = LLAMA_summary[summary_start_index + len("Summary:"):].strip()

        # Calculate ROUGE scores for individual summaries
        rouge_scores = rouge.compute(predictions=[LLAMA_summary_sliced], references=[[summary]])

        # Accumulate ROUGE scores
        for rouge_type, score in rouge_scores.items():
            rouge_scores_sum[rouge_type] += score  # ROUGE score is stored as a tuple

        # Concatenate summaries
        concatenated_LLAMA_summary += " " + LLAMA_summary_sliced
        concatenated_summary += " " + summary

    # Calculate average ROUGE scores
    avg_rouge_scores = {rouge_type: score / num_items for rouge_type, score in rouge_scores_sum.items()}

    # Calculate ROUGE scores for concatenated summaries
    concatenated_rouge_scores = rouge.compute(predictions=[concatenated_LLAMA_summary],
                                              references=[[concatenated_summary]])

    # Append details to the list for LLAMA summaries
    LLAMA_details_list.append({
        'Meeting ID': meeting_id,
        'Number of Items': num_items,
        'Average ROUGE Scores': avg_rouge_scores,
        'Concatenated ROUGE Scores': concatenated_rouge_scores
    })



# Load Gemma JSON data
gemma_json_file = "output1_gemma_MeetingBank.json"
with open(gemma_json_file, "r") as file:
    gemma_json_data = json.load(file)

# Initialize a list to store all the details for Gemma summaries
gemma_details_list = []

# Calculate ROUGE scores and accumulate details for Gemma summaries
for meeting_data in gemma_json_data:
    meeting_id = list(meeting_data.keys())[0]
    item_info = meeting_data[meeting_id]["itemInfo"]

    concatenated_gemma_summary = ""
    concatenated_summary = ""
    rouge_scores_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}  # Initialize sum of ROUGE scores
    num_items = len(item_info)  # Number of items

    for item_id, item_info in item_info.items():
        gemma_summary = item_info.get("gemma_summary", "")
        summary = item_info.get("Summary", "")

        # Calculate ROUGE scores for individual summaries
        rouge_scores = rouge.compute(predictions=[gemma_summary], references=[[summary]])

        # Accumulate ROUGE scores
        for rouge_type, score in rouge_scores.items():
            rouge_scores_sum[rouge_type] += score  # ROUGE score is stored as a tuple

        # Concatenate summaries
        concatenated_gemma_summary += " " + gemma_summary
        concatenated_summary += " " + summary

    # Calculate average ROUGE scores
    avg_rouge_scores = {rouge_type: score / num_items for rouge_type, score in rouge_scores_sum.items()}

    # Calculate ROUGE scores for concatenated summaries
    concatenated_rouge_scores = rouge.compute(predictions=[concatenated_gemma_summary],
                                              references=[[concatenated_summary]])

    # Append details to the list for Gemma summaries
    gemma_details_list.append({
        'Meeting ID': meeting_id,
        'Number of Items': num_items,
        'Average ROUGE Scores': avg_rouge_scores,
        'Concatenated ROUGE Scores': concatenated_rouge_scores
    })
    
# Create a DataFrame from the list of details
gpt4_df = pd.DataFrame(gpt4_details_list)
bart_df = pd.DataFrame(bart_details_list)
llama2_df = pd.DataFrame(llama2_details_list)
LLAMA_df = pd.DataFrame(LLAMA_details_list)
gemma_df = pd.DataFrame(gemma_details_list)

# Combine DataFrames for BART, llama2, and GPT4
combined_df = pd.concat([bart_df, llama2_df, gpt4_df, LLAMA_df, gemma_df], axis=1)

# Add labels for GPT4 ROUGE scores
combined_df.columns = ['BART Meeting ID', 'BART Number of Items', 'BART Average ROUGE Scores', 'BART Concatenated ROUGE Scores',
                       'llama2 Meeting ID', 'llama2 Number of Items', 'llama2 Average ROUGE Scores', 'llama2 Concatenated ROUGE Scores',
                       'GPT4 Meeting ID', 'GPT4 Number of Items', 'GPT4 Average ROUGE Scores', 'GPT4 Concatenated ROUGE Scores',
                       'LLAMA Meeting ID', 'LLAMA Number of Items', 'LLAMA Average ROUGE Scores', 'LLAMA Concatenated ROUGE Scores',
                       'Gemma Meeting ID', 'Gemma Number of Items', 'Gemma Average ROUGE Scores', 'Gemma Concatenated ROUGE Scores']

# Display the combined DataFrame
display(combined_df)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('RougeScore_finalResults1.csv', index=False)