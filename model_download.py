from huggingface_hub import snapshot_download

# Downloads the full repository to a local directory
# snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="./my_local_model")

from sentence_transformers import SentenceTransformer, util

# This will download and load the model
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('./my_local_model')
# Example usage
sentences = ["This is an example sentence", "Each sentence is converted"]
# embeddings = model.encode(sentences)
# 2. Define the sentences you want to compare
sentences = [
    "What actions can farmers take to improve soil health based on the microbial composition they identify through analysis?",
    "Upon analyzing microbial composition through various testing methods, farmers can take actionable steps to improve soil health. If the analysis indicates a lack of beneficial microbes, farmers might consider adding organic matter or cover crops that support microbial diversity. This can include incorporating compost or using crop rotation strategies that naturally encourage beneficial microbial populations. Additionally, monitoring soil pH and nutrient levels can help farmers create an optimal environment for desired microbes, thus enhancing soil health over time. Implementing these strategies not only supports plant growth but also fosters a more balanced soil ecosystem.",
]
# 3. Compute embeddings
# convert_to_tensor=True allows for faster calculation using PyTorch
embeddings = model.encode(sentences, convert_to_tensor=True)

# 4. Calculate the cosine similarity score
# This returns a matrix; [0, 1] gets the score between the first and second sentence
score = util.cos_sim(embeddings[0], embeddings[1])

print(f"Similarity Score: {score.item():.4f}")
