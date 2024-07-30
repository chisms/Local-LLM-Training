# YouTube Video LLM Training with GPT-2

This project allows you to fine-tune a GPT-2 language model on the content of YouTube videos. It includes scripts for preparing the dataset from a YouTube video, training the model, and running inference.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- ffmpeg (for audio extraction)

## Setup

1. Clone this repository or download the scripts to your local machine.

2. Create a virtual environment:
   ```
   python3 -m venv youtube_llm_env
   ```

3. Activate the virtual environment:
   - On Unix or MacOS:
     ```
     source youtube_llm_env/bin/activate
     ```
   - On Windows:
     ```
     youtube_llm_env\Scripts\activate
     ```

4. Install the required packages:
   ```
   pip install yt-dlp ffmpeg-python openai-whisper nltk scikit-learn transformers torch datasets
   ```

5. Download the NLTK punkt tokenizer:
   ```
   python -c "import nltk; nltk.download('punkt')"
   ```

## Preparing the Dataset

1. Open `youtube_dataset_prep.py` and replace the `video_url` with the URL of the YouTube video you want to use:

   ```python
   video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID_HERE"
   ```

2. Run the script:
   ```
   python youtube_dataset_prep.py
   ```

   This script will:
   - Download the video
   - Extract the audio
   - Transcribe the audio using Whisper
   - Clean and preprocess the text
   - Segment the text into chunks
   - Split the data into training, validation, and test sets
   - Save these datasets as JSON files

   The process may take some time, especially for longer videos.

## Training the Model

1. Ensure the file names in `train_llm.py` match your dataset files (e.g., "video_id_train.json").

2. The script uses the GPT-2 model. Key points about the training process:
   - It fine-tunes a pre-trained GPT-2 model on your specific data.
   - The script is set to run for 3 epochs by default. An epoch is one complete pass through the entire training dataset.
   - You can adjust the number of epochs in the `TrainingArguments` if needed.

3. Run the training script:
   ```
   python train_llm.py
   ```

   This will fine-tune the model and save it in a directory named `my_fine_tuned_model`.

## Running Inference

1. Ensure that the `model_path` in `inference.py` points to the correct directory where your model was saved (usually `"./my_fine_tuned_model"`).

2. Run the inference script:
   ```
   python inference.py
   ```

3. Enter prompts when prompted to generate text based on your fine-tuned model.

## Understanding the Process

- Video Processing: The `youtube_dataset_prep.py` script handles downloading, transcribing, and preprocessing the video content.
- Fine-tuning: This process takes a pre-trained GPT-2 model and further trains it on your specific dataset. It doesn't create a new model from scratch but adapts an existing one to your content.
- Training Epochs: The model goes through the entire dataset multiple times (3 by default). You can adjust this in the `TrainingArguments` of `train_llm.py`.
- Model Output: The fine-tuned model will generate text that's influenced by both its original training data and your YouTube video content.

## Customization

- To adjust the number of training epochs, modify the `num_train_epochs` parameter in the `TrainingArguments` in `train_llm.py`.
- To use a different variant of GPT-2, change the model name in `from_pretrained()` (e.g., "gpt2-medium", "gpt2-large").
- To change how text is generated, modify the parameters in the `generate_text` function in `inference.py`.
- To process a different video, change the `video_url` in `youtube_dataset_prep.py`.

## Troubleshooting

- If you encounter errors related to missing modules, ensure all required packages are installed and your virtual environment is activated.
- If the model isn't generating relevant content, consider:
  - Checking the quality and relevance of your training data
  - Increasing the number of training epochs
  - Adjusting hyperparameters like learning rate or batch size
- If the model isn't found during inference, verify that the `model_path` in `inference.py` matches the directory where your model was saved after training.
- If you encounter issues with video download or processing, ensure you have the latest version of yt-dlp and ffmpeg installed.

## Notes

- Training language models can be computationally intensive. Ensure you have adequate computational resources.
- Be mindful of YouTube's terms of service when downloading videos.
- The quality of your model will depend on the quality and quantity of your training data.

## Future Expansions

While this project uses GPT-2, it's possible to adapt it to use other models like BERT, T5, or even LLaMA (with proper access and modifications). Each model has its strengths and may require different approaches to fine-tuning and inference.