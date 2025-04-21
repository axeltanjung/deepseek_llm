# Import libraries
import torch
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepseek
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from deepseek import Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load your dataset
dataset = load_dataset("your_dataset_name")

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Fine-tune the model using DeepSeek
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()

# Generate text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Perplexity: {eval_results['eval_loss']}")

# Generate embeddings using SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(dataset['train']['text'], convert_to_tensor=True)

# Create a FAISS index and add embeddings
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings.cpu().numpy())

# Example query
query = "example query text"
query_embedding = embedder.encode([query], convert_to_tensor=True)
D, I = index.search(query_embedding.cpu().numpy(), k=5)  # k is the number of nearest neighbors

# Print the results
print("Nearest neighbors:", I)
print("Distances:", D)