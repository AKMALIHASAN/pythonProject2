import torch
from transformers import AutoTokenizer, AutoModel

# Load multilingual BERT model and tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#### test
# Function to preprocess input text
def preprocess(text):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return tokens


# Function to compute similarity between two articles
def similarity(article1, article2):
    # Preprocess articles
    tokens1 = preprocess(article1)
    tokens2 = preprocess(article2)

    # Compute embeddings for article 1
    with torch.no_grad():
        output1 = model(**tokens1)
        embeddings1 = output1.last_hidden_state.mean(dim=1)

    # Compute embeddings for article 2
    with torch.no_grad():
        output2 = model(**tokens2)
        embeddings2 = output2.last_hidden_state.mean(dim=1)

    # Compute cosine similarity between embeddings
    cos = torch.nn.CosineSimilarity(dim=1)
    sim = cos(embeddings1, embeddings2)

    return sim.item()


# Example usage
article1 = "In a major breakthrough, researchers have discovered a new treatment for cancer."
article2 = "Scientists have made a significant discovery in the fight against cancer."
similarity_score = similarity(article1, article2)
print(similarity_score)
