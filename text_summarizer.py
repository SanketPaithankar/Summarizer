import click
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"  # Adjust to use available device
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def summarize_text(text):
    prompt = "Summarize the following text:\n\n" + text
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([formatted_text], return_tensors="pt", padding=True, truncation=True)

    generated_ids = model.generate(
        model_inputs['input_ids'],
        attention_mask=model_inputs['attention_mask'],
        max_new_tokens=256
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def save_summary(text, summary, filename):
    with open(filename, 'w') as file:
        file.write(f"Original Text:\n{text}\n\nSummary:\n{summary}")

@click.command()
@click.option('-t', '--text-file', type=click.Path(exists=True), help='Path to the text file to summarize')
@click.argument('text', required=False)
def main(text_file, text):
    """
    Summarize text using Qwen2 0.5B model.
    
    You can provide the text to summarize either via a text file or directly as an argument.
    """
    if text_file:
        with open(text_file, 'r') as file:
            text = file.read()
        summary_filename = text_file.replace(".txt", "_summary.txt")
    elif text:
        summary_filename = "yourtext_summary.txt"
    else:
        click.echo('Error: No text provided. Please provide text via a file or as an argument.')
        return

    summary = summarize_text(text)
    save_summary(text, summary, summary_filename)
    click.echo(f'Summary saved to {summary_filename}')

if __name__ == '__main__':
    main()
