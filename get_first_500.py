# Define the file path and read the first 500 words
file_path = './data/raw/pg_essays/a_fundraising_survival_guide.txt'
with open('data/raw/pg_essays/what_to_do.txt', 'r') as file:
    text = file.read()

# Extract the first 500 words
first_500_words = ' '.join(text.split()[:500])

print("First 500 words of the text:")
print(first_500_words)