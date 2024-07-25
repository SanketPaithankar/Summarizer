# Text Summarization with Qwen2 0.5B Model

This Python script utilizes the Qwen2 0.5B model from Hugging Face to summarize text. You can provide the text either through a file or directly as a command-line argument.

## Requirements

- Python 3.7 or higher
- `click` library
- `torch` library
- `transformers` library

You can install the necessary libraries using pip:

```bash
pip install click torch transformers
```
Setting Up a New Environment
To ensure a clean setup, it is recommended to create a new Python virtual environment. Follow these steps:

Create a Virtual Environment:

```bash
python -m venv myenv
```
Replace myenv with your preferred environment name.

Activate the Virtual Environment:

On Windows:

```bash
myenv\Scripts\activate
```
With the virtual environment activated, install the dependencies:

```bash
pip install click torch transformers
```
Usage

You can use the script via the command line. There are two ways to provide the text to summarize:

From a Text File:

```bash
python script.py --text-file path/to/yourfile.txt
```
This will read the content of yourfile.txt, summarize it, and save the summary to yourfile_summary.txt.

Directly as an Argument:

```bash
python script.py "Your text here"
```
This will summarize the provided text and save the summary to yourtext_summary.txt.

Options
-t or --text-file: Path to the text file to summarize.
text: Directly provide the text to summarize as an argument.
Example
To summarize text from a file named example.txt:

```bash
python text_summarizer.py -t example.txt
```


To summarize text directly:
``` bash
python text_summarizer.py "The quick brown fox jumps over the lazy dog."
```
This will generate a summary of the provided text and save it to yourtext_summary.txt.
