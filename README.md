# LLM-Powered Interview Answer Evaluator with Lip Sync Analysis

This project demonstrates a system for evaluating interview candidate answers using a fine-tuned Large Language Model (LLM) and incorporating lip-sync analysis for video responses.

## Project Overview

The core idea is to automate and enhance the interview evaluation process by leveraging the power of LLMs to provide structured feedback on candidate responses. Additionally, for video interviews, the system aims to provide an objective measure of lip-sync quality, which can be an indicator of video/audio quality or even the authenticity of the response.

## Features

-   **LLM-based Answer Evaluation**: A Mistral-7B-Instruct-v0.1 model is fine-tuned using LoRA (Low-Rank Adaptation) to act as a strict HR evaluator. It scores candidate answers based on: 
    -   Clarity (0-2)
    -   Structure (0-2)
    -   Relevance (0-2)
    -   Quality of Example (0-2)
    -   Professional Communication (0-2)
    The model provides a total score out of 10, along with identified strengths and weaknesses, in a structured JSON format.
-   **Audio Transcription**: Integrates OpenAI Whisper to transcribe spoken answers from video files, converting them into text that can then be fed to the LLM for evaluation.
-   **Lip Sync Analysis**: Attempts to correlate audio energy with estimated mouth openness from video frames using OpenCV (Haar Cascade for face detection) and Librosa for audio processing. This aims to provide a quantitative score for lip-sync quality.

## Technologies Used

-   **Python**: The primary programming language.
-   **Hugging Face Transformers**: For loading and interacting with pre-trained LLMs.
-   **PEFT (Parameter-Efficient Fine-tuning)**: Specifically LoRA, for efficiently fine-tuning the LLM.
-   **BitsAndBytes**: For 4-bit quantization, enabling the use of large models with limited GPU memory.
-   **Datasets**: For handling and processing training data.
-   **OpenAI Whisper**: For robust speech-to-text transcription.
-   **OpenCV**: For video processing and face detection.
-   **Librosa**: For audio analysis.
-   **NumPy**: For numerical operations.

## Setup and Installation

1.  **Clone the repository** (if applicable, or set up in Google Colab).
2.  **Install necessary packages**: The project relies on several Python libraries. You can install them using pip:
    ```bash
    !pip install transformers accelerate bitsandbytes peft datasets openai-whisper opencv-python librosa numpy
    !apt install -y ffmpeg # For audio/video processing
    ```
3.  **Prepare your data**: The LLM fine-tuning requires a dataset in JSONL format (e.g., `deneme1.jsonl`). This dataset should contain examples of questions and evaluated answers to train the evaluator model.

## Usage

### 1. LLM Fine-tuning

The `oaZkFYweTb-P`, `l3ZQjVe_d3BE`, `_ObEdCcyi75u`, `uCEiZga0iT-6`, `5eWppMX5i2xP`, and `NzWNlmWmjbyP` cells handle the model loading, 4-bit quantization, LoRA configuration, dataset preparation, and training of the Mistral model.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ... (model and tokenizer loading, bnb_config, lora_config, dataset loading and tokenization as in the notebook)

trainer.train()
```

### 2. Evaluating Text Answers

The `evaluate_answer` function uses the fine-tuned LLM to assess a given question and candidate answer.

```python
# Define the evaluation function (as in WepOHpmfH9Rp)
def evaluate_answer(question, answer):
    # ... (function implementation)

question = "Tell me about a time you had to learn something quickly."
answer = """
One situation where I had to learn something very quickly was during my internship, when I was asked to fine-tune a large language model for a domain-specific task.
, I had never worked hands-on with parameter-efficient fine-tuning methods like LoRA before. The timeline was tight, and I needed to deliver a working prototype within a short period.
To adapt quickly, I first broke the problem into smaller parts.his experience strengthened my ability to learn under pressure, structure complex topics efficiently, and translate theory into practical implementation quickly."""

print(evaluate_answer(question, answer))
```

### 3. Evaluating Video Answers (Transcription and Lip Sync)

First, transcribe the video using OpenAI Whisper, then use the transcription for LLM evaluation. Additionally, the lip-sync analysis attempts to provide a score for the video itself.

```python
import whisper

# Transcribe the video (as in oYpiJWgPPOWk)
whisper_model = whisper.load_model("base")
result = whisper_model.transcribe("/content/LLM_Cypher.mp4")
transcript = result["text"]
print(transcript)

# Evaluate the transcribed answer (as in 0-xPHD3ZPdL2)
question = "Tell me about a time you had to learn something quickly."
answer = transcript
print(evaluate_answer(question, answer))

# Perform Lip Sync Analysis (as in 1WXdi2oNgJvL)
import cv2
import numpy as np
import librosa

VIDEO_PATH = "/content/LLM_Cypher.mp4"
FPS = 10
DURATION = 5

# ... (rest of the lip sync analysis code using Haar Cascade, Librosa, etc.)

print("Face Presence Ratio:", face_presence_ratio)
print("Lip Sync Correlation:", correlation)
print("Lip Sync Score:", lip_sync_score)
```
