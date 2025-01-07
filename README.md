# Timeline Long Meeting Summarization (TLMS) Research Project

## Overview
This research project investigates the effectiveness of Large Language Models (LLMs) in Timeline Long Meeting Summarization (TLMS). TLMS is an automated process that creates timeline-based summaries of meetings, capturing the progression of events and key discussions. Our study represents one of the first comprehensive evaluations of LLMs' performance in TLMS tasks.

## Project Description
The project focuses on developing and evaluating a chunking-based approach for meeting summarization using various LLMs. We utilize the MeetingBank dataset, which contains meeting data from six major U.S. cities, to assess how effectively different models can capture and condense key events within meeting timelines.

## Dataset
The MeetingBank dataset includes:
- 1,366 meetings from six U.S. cities
- Average meeting duration: 2.6 hours
- Average transcript length: 28,000+ tokens
- Components: Video recordings, transcripts, PDF documents of meeting minutes, agendas, and metadata

## Models Evaluated
We evaluated six different LLMs:
- BART (406M parameters)
- FLAN-T5 (780M parameters)
- GEMMA (7B parameters)
- GPT-3.5 (20B parameters)
- Long-LLaMA (3B parameters)
- LLaMA2 (7B parameters)

## Methodology
Our approach implements Sequential Context Length Chunking, which involves:
1. Aggregating meeting transcripts
2. Dividing transcripts into input-size chunks based on LLM context length
3. First-level summarization of individual chunks
4. Second-level summarization combining chunk summaries for timeline creation

## Evaluation Metrics
We used two main evaluation metrics:
- Concat F1: Evaluates summarization quality across entire meetings
- Agree F1: Assesses summary quality at minute-level granularity

Both metrics utilize ROUGE-1 and ROUGE-2 scores for evaluation.

## Technical Setup
### Hardware Requirements
- Operating System: Ubuntu 18.04.5 LTS
- RAM: 220GB
- GPU: NVIDIA GeForce RTX 3090 (24GB)

### Software Requirements
- Python: 3.10.11
- Hyperparameters:
  - temperature: 0.5
  - top_p: 1
  - top_k: 50

## Key Findings
- LLaMA2 achieved the highest Concat F1 scores
- GPT-3.5 showed strong performance in Agree F1 metrics
- GPT-3.5 demonstrated consistent performance across both meeting-level and minute-level evaluations
- BART and GEMMA showed relatively lower performance
- FLAN-T5 exhibited mediocre results in Concat F1

## Future Work
This research provides a foundation for future studies in TLMS. Potential areas for improvement include:
- Exploring advanced prompting techniques
- Investigating different chunking strategies
- Developing TLMS-specific evaluation metrics
- Implementing more sophisticated timeline generation approaches
