# Define the file path and read the first 500 words
file_path = 'data/raw/pg_essays/after_credentials.txt'
with open(file_path, 'r') as file:
    text = file.read()

# calculate the length of the text
text_length = len(text.split())

# print the length of the text
print(f"Length of the text: {text_length} words")

# Extract the first 500 words
first_500_words = ' '.join(text.split()[:500])

# extract second 500 words
second_500_words = ' '.join(text.split()[500:1000])

print("First 500 words of the text:")
print(first_500_words)
print()

print("Second 500 words of the text:")
print(second_500_words)
