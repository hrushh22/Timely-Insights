import json
from evaluate import load

# Load the ROUGE metric
rouge = load('rouge')

# Load JSON data
json_file = "flan_summaries_rouge_scores.json"
with open(json_file, "r") as file:
    json_data = json.load(file)

# Calculate ROUGE scores and update JSON data
for field, field_data in json_data.items():
    concatenated_flan_summary = ""
    concatenated_summary = ""
    rouge_scores_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}  # Initialize sum of ROUGE scores
    num_items = len(field_data.get("itemInfo", {}))  # Number of items

    for item_id, item_info in field_data.get("itemInfo", {}).items():
        flan_summary = item_info.get("flan_summary", "")
        summary = item_info.get("Summary", "")

        # Calculate ROUGE scores for individual summaries
        rouge_scores = rouge.compute(predictions=[flan_summary], references=[[summary]])
        item_info["rouge_scores"] = rouge_scores
        print("item id: ", item_id, "  rouge score: ", rouge_scores)

        # Accumulate ROUGE scores
        for rouge_type, score in rouge_scores.items():
            rouge_scores_sum[rouge_type] += score  # ROUGE score is stored as a tuple

        # Concatenate summaries
        concatenated_flan_summary += " " + flan_summary
        concatenated_summary += " " + summary

    # Calculate average ROUGE scores
    avg_rouge_scores = {rouge_type: score / num_items for rouge_type, score in rouge_scores_sum.items()}

    # Calculate ROUGE scores for concatenated summaries
    concatenated_rouge_scores = rouge.compute(predictions=[concatenated_flan_summary],
                                              references=[[concatenated_summary]])
    print("meeting id: ", field, "  avg rouge score: ", avg_rouge_scores)
    print("meeting id: ", field, "  concatenated rouge score: ", concatenated_rouge_scores)

    # Store ROUGE scores in the same JSON structure
    field_data["avg_rouge_scores"] = avg_rouge_scores
    field_data["concatenated_rouge_scores"] = concatenated_rouge_scores

# Write updated JSON data back to the file
with open(json_file, "w") as file:
    json.dump(json_data, file, indent=4)