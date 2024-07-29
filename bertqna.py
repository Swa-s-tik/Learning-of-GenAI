from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Define the document and question
document = "The quick brown fox jumps over the lazy dog."
question = "What jumps over the lazy dog?"

# Tokenize the inputs
inputs = tokenizer(question, document, return_tensors="pt")

# Get the model's outputs
with torch.no_grad():
    outputs = model(**inputs)

# Get the start and end logits
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Find the positions of the start and end tokens with the highest scores
answer_start = torch.argmax(start_logits)
answer_end = torch.argmax(end_logits) + 1  # +1 to include the end token in the answer

# Convert the token ids back to text
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))

# Print the answer
print("Answer:", answer)
