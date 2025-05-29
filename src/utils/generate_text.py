import os
from transformers import pipeline

# Replace 'your_token' with the active Hugging Face token
token = "nlp-token"


# Define the file path and read the first 500 words
file_path = './data/raw/pg_essays/a_fundraising_survival_guide.txt'
with open(file_path, 'r') as file:
    text = file.read()

# Extract the first 500 words
first_500_words = ' '.join(text.split()[:500])

# Define the prompt based on the extracted text
# Define the prompt based on the extracted text
prompt = (
    f"Below is an excerpt from an essay written by Paul Graham. "
    f"Please continue the essay in the same style, tone, and diction as the original text.\n\n"
    f"Excerpt:\n{first_500_words}\n\n"
    f"Instructions:\n"
    f"- Write exactly 500 words to continue the essay.\n"
    f"- Maintain the same writing style, tone, and vocabulary as the excerpt.\n"
    f"- The continuation does not need to conclude the essay; it can end mid-thought or mid-sentence."
)

# Load the Llama-4 model hosted on Hugging Face with authentication
text_generator = pipeline('text-generation', model='ibm-granite/granite-3.3-2b-instruct', use_auth_token=token)

# Generate the next 500 words
output = text_generator(prompt, max_length=1000, num_return_sequences=1)

# Extract and print the generated text
generated_text = output[0]['generated_text']
print("Generated Text:")
print(generated_text)