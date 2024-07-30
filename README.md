# YouTube Video LLM Training with GPT-2

This project allows you to fine-tune a GPT-2 language model on the content of YouTube videos, PDFs, or JSON files. It includes scripts for preparing the dataset, training the model, and running inference, as well as a Flask web application for easy interaction.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- ffmpeg (for audio extraction)
- CUDA-capable GPU (optional, but recommended for faster training)

## Setup

1. Clone this repository or download the scripts to your local machine.

2. Create a virtual environment:
python3 -m venv youtube_llm_env
Copy
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
pip install yt-dlp ffmpeg-python openai-whisper nltk scikit-learn transformers torch datasets flask werkzeug PyPDF2
Copy
5. Download the NLTK punkt tokenizer:
python -c "import nltk; nltk.download('punkt')"
Copy
6. If you have a CUDA-capable GPU, ensure you have the appropriate CUDA toolkit and cuDNN installed. The PyTorch installation should automatically detect and use your GPU.

## Running the Web Application

1. Start the Flask application:
python app.py
Copy
2. Open a web browser and navigate to `http://localhost:5000`.

3. Use the web interface to upload YouTube URLs, PDF files, or JSON files for processing and training.

## Preparing the Dataset

The `process_files.py` script handles dataset preparation for different file types:

- YouTube videos: Transcribes the video and processes the text.
- PDF files: Extracts text from the PDF and processes it.
- JSON files: Loads text data from a JSON file.

## Training the Model

The `train_llm.py` script handles the model training:

- It uses the GPT-2 model for fine-tuning.
- Training hyperparameters can be adjusted in the `TrainingArguments` in `train_llm.py`.
- The script automatically detects and uses a CUDA-capable GPU if available.

## Running Inference

The `py_inference.py` script allows you to interact with the trained model:

1. Ensure that the `model_path` in the script points to your trained model directory.
2. Run the script:
python py_inference.py
Copy3. Enter prompts when prompted to generate text based on your fine-tuned model.

## Project Structure

- `app.py`: Flask web application for file upload and processing.
- `youtube_dataset_prep.py`: Handles YouTube video downloading and transcription.
- `process_files.py`: Processes different file types (YouTube, PDF, JSON) for training.
- `train_llm.py`: Trains the GPT-2 model on the processed data.
- `py_inference.py`: Allows interaction with the trained model.
- `index.html`: HTML template for the web interface.

## Customization

- Adjust training parameters in `train_llm.py` (e.g., number of epochs, learning rate).
- Modify the `generate_text` function in `py_inference.py` to change text generation parameters.
- Update the Flask app in `app.py` to add new features or change the web interface.

## Troubleshooting

- If you encounter CUDA-related errors, ensure your CUDA toolkit and cuDNN are correctly installed and compatible with your PyTorch version.
- For issues with specific file types, check the relevant processing function in `process_files.py`.
- If the model isn't generating relevant content, consider increasing the number of training epochs or adjusting other hyperparameters.

## Notes

- Training language models can be computationally intensive. A CUDA-capable GPU is recommended for faster training.
- Be mindful of YouTube's terms of service when downloading videos.
- The quality of your model will depend on the quality and quantity of your training data.

## Future Expansions

- Support for additional file types and data sources.
- Integration with more advanced language models.
- Improved web interface with real-time training progress updates.
