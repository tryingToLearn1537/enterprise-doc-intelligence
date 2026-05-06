from sentence_transformers import SentenceTransformer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL

print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Embedding model loaded!")


def get_embedding(text: str) -> list:
    """
    Takes one piece of text and returns its vector (list of 384 numbers).
    This is the 'fingerprint' of that text's meaning.
    """
    embedding = model.encode(text)
    return embedding.tolist()  # Convert numpy array to plain Python list


def get_embeddings_batch(texts: list) -> list:
    """
    Takes a LIST of texts and returns a list of vectors.
    Much faster than calling get_embedding() one by one.
    Used when storing all chunks from a PDF.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


# --- TEST ---
if __name__ == "__main__":

    # Test 1: Single embedding
    test_text = "The company reported strong revenue growth in Q3."
    vector = get_embedding(test_text)
    print(f"\n✅ Single embedding works!")
    print(f"   Text: '{test_text}'")
    print(f"   Vector size: {len(vector)} numbers")
    print(f"   First 5 numbers: {vector[:5]}")

    # Test 2: Similarity check
    # Similar sentences should have vectors close to each other
    from sentence_transformers import util
    import torch

    sentence1 = "What is the company's annual revenue?"
    sentence2 = "How much money did the company make this year?"
    sentence3 = "What is the capital of France?"

    vec1 = model.encode(sentence1, convert_to_tensor=True)
    vec2 = model.encode(sentence2, convert_to_tensor=True)
    vec3 = model.encode(sentence3, convert_to_tensor=True)

    sim_12 = util.cos_sim(vec1, vec2).item()
    sim_13 = util.cos_sim(vec1, vec3).item()

    print(f"\n✅ Similarity test:")
    print(f"   '{sentence1}'")
    print(f"   vs '{sentence2}'")
    print(f"   Similarity: {sim_12:.2f} (should be HIGH, close to 1.0)")
    print(f"\n   '{sentence1}'")
    print(f"   vs '{sentence3}'")
    print(f"   Similarity: {sim_13:.2f} (should be LOW, close to 0.0)")

    print(f"\n🎉 embeddings.py is working correctly!")