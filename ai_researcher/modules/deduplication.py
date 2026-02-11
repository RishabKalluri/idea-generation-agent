"""
Deduplication Module using embedding-based similarity.

Uses sentence transformers to detect and remove duplicate/similar ideas
based on semantic similarity of their text representations.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# Load SeedIdea class directly to avoid anthropic dependency
import importlib.util

def _load_seed_idea_class():
    """Load SeedIdea class without triggering anthropic import."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    idea_gen_path = os.path.join(current_dir, "idea_generation.py")
    
    if not os.path.exists(idea_gen_path):
        return None
    
    spec = importlib.util.spec_from_file_location("idea_generation_direct", idea_gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SeedIdea

SeedIdea = _load_seed_idea_class()


# ============================================================================
# EMBEDDING MODEL
# ============================================================================

# Lazy loading of the embedding model to avoid import-time overhead
_EMBEDDING_MODEL = None

def get_embedding_model():
    """Load the embedding model (lazy loading)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("[Deduplication] Loading embedding model: all-MiniLM-L6-v2")
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("[Deduplication] Model loaded successfully")
    return _EMBEDDING_MODEL


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def embed_ideas(ideas: List, use_full_text: bool = True) -> np.ndarray:
    """
    Embed all ideas using their text representation.
    
    Args:
        ideas: List of SeedIdea objects
        use_full_text: If True, use raw_text; if False, use title + problem + method
    
    Returns:
        Array of shape (num_ideas, embedding_dim)
    """
    model = get_embedding_model()
    
    # Extract text for embedding
    if use_full_text:
        texts = []
        for idea in ideas:
            if hasattr(idea, 'raw_text') and idea.raw_text:
                texts.append(idea.raw_text)
            else:
                # Fallback to concatenated fields
                texts.append(idea.to_string() if hasattr(idea, 'to_string') else str(idea))
    else:
        # Use key fields only for faster/more focused comparison
        texts = []
        for idea in ideas:
            text = f"{idea.title}\n{idea.problem}\n{idea.proposed_method}"
            texts.append(text)
    
    print(f"[Deduplication] Embedding {len(texts)} ideas...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings


def compute_embedding(text: str) -> np.ndarray:
    """Compute embedding for a single text."""
    model = get_embedding_model()
    return model.encode([text], convert_to_numpy=True)[0]


# ============================================================================
# SIMILARITY FUNCTIONS
# ============================================================================

def compute_pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix.
    
    Args:
        embeddings: Array of shape (num_ideas, embedding_dim)
    
    Returns:
        Similarity matrix of shape (num_ideas, num_ideas)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms
    
    # Compute cosine similarity
    similarities = np.dot(normalized, normalized.T)
    
    return similarities


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    return np.dot(emb1, emb2) / (norm1 * norm2)


# ============================================================================
# DEDUPLICATION FUNCTIONS
# ============================================================================

def deduplicate_ideas(
    ideas: List,
    threshold: float = 0.8,
    show_progress: bool = True
) -> List:
    """
    Remove duplicate ideas based on embedding similarity.
    
    For each idea, check if it has similarity > threshold with any 
    previously kept idea. If so, discard it.
    
    Args:
        ideas: List of SeedIdea objects in generation order
        threshold: Similarity threshold (paper uses 0.8)
        show_progress: Whether to show progress bar
    
    Returns:
        List of deduplicated ideas (approximately 5% of input based on paper)
    """
    if len(ideas) == 0:
        return []
    
    if len(ideas) == 1:
        return ideas.copy()
    
    print(f"\n[Deduplication] Starting deduplication of {len(ideas)} ideas")
    print(f"  Similarity threshold: {threshold}")
    
    # Compute all embeddings at once (more efficient)
    embeddings = embed_ideas(ideas)
    
    # Track kept ideas and their embeddings
    kept_ideas = [ideas[0]]
    kept_embeddings = [embeddings[0]]
    kept_indices = [0]
    
    # Track duplicates for analysis
    duplicate_count = 0
    duplicate_pairs = []  # (duplicate_idx, original_idx, similarity)
    
    # Iterate through remaining ideas
    iterator = range(1, len(ideas))
    if show_progress:
        iterator = tqdm(iterator, desc="Deduplicating", initial=1, total=len(ideas))
    
    for i in iterator:
        current_embedding = embeddings[i]
        
        # Check similarity with all kept ideas
        is_duplicate = False
        max_similarity = 0.0
        most_similar_idx = -1
        
        for j, kept_emb in enumerate(kept_embeddings):
            similarity = cosine_similarity(current_embedding, kept_emb)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_idx = kept_indices[j]
            
            if similarity > threshold:
                is_duplicate = True
                break  # Found a duplicate, no need to check others
        
        if is_duplicate:
            duplicate_count += 1
            duplicate_pairs.append((i, most_similar_idx, max_similarity))
        else:
            kept_ideas.append(ideas[i])
            kept_embeddings.append(current_embedding)
            kept_indices.append(i)
    
    # Print summary
    keep_rate = len(kept_ideas) / len(ideas) * 100
    print(f"\n[Deduplication] Complete!")
    print(f"  Original: {len(ideas)} ideas")
    print(f"  Kept: {len(kept_ideas)} ideas ({keep_rate:.1f}%)")
    print(f"  Removed: {duplicate_count} duplicates ({100-keep_rate:.1f}%)")
    
    return kept_ideas


def deduplicate_to_target(
    ideas: List,
    target_count: int = None,
    target_percent: float = 0.05,
    min_threshold: float = 0.5,
    max_threshold: float = 0.9,
    show_progress: bool = True
) -> List:
    """
    Deduplicate ideas to reach approximately a target count or percentage.
    
    Uses binary search to find the right threshold that yields the target.
    
    Args:
        ideas: List of SeedIdea objects
        target_count: Target number of ideas (overrides target_percent if set)
        target_percent: Target percentage to keep (default 5%)
        min_threshold: Minimum similarity threshold to try
        max_threshold: Maximum similarity threshold to try
        show_progress: Whether to show progress
    
    Returns:
        List of deduplicated ideas close to the target count
    """
    if len(ideas) == 0:
        return []
    
    # Determine target count
    if target_count is None:
        target_count = max(1, int(len(ideas) * target_percent))
    
    print(f"\n[Deduplication] Targeting ~{target_count} ideas ({target_count/len(ideas)*100:.1f}%)")
    
    # Compute embeddings once
    embeddings = embed_ideas(ideas)
    
    def count_with_threshold(threshold: float) -> int:
        """Count how many ideas survive with given threshold."""
        kept_embeddings = [embeddings[0]]
        count = 1
        
        for i in range(1, len(ideas)):
            is_duplicate = False
            for kept_emb in kept_embeddings:
                if cosine_similarity(embeddings[i], kept_emb) > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_embeddings.append(embeddings[i])
                count += 1
        return count
    
    # Binary search for right threshold
    low, high = min_threshold, max_threshold
    best_threshold = (low + high) / 2
    best_count = count_with_threshold(best_threshold)
    
    for _ in range(10):  # Max 10 iterations
        mid = (low + high) / 2
        count = count_with_threshold(mid)
        
        if abs(count - target_count) < abs(best_count - target_count):
            best_threshold = mid
            best_count = count
        
        if count > target_count:
            # Too many kept, need lower threshold (more aggressive)
            high = mid
        else:
            # Too few kept, need higher threshold (less aggressive)
            low = mid
        
        if abs(count - target_count) <= max(1, target_count * 0.1):
            break  # Close enough (within 10%)
    
    print(f"  Found threshold: {best_threshold:.3f} → ~{best_count} ideas")
    
    # Run actual deduplication with best threshold
    return deduplicate_ideas(ideas, threshold=best_threshold, show_progress=show_progress)


def deduplicate_ideas_with_info(
    ideas: List,
    threshold: float = 0.8
) -> Tuple[List, Dict]:
    """
    Deduplicate ideas and return detailed information.
    
    Returns:
        Tuple of (kept_ideas, info_dict)
        
        info_dict contains:
        - kept_indices: indices of kept ideas in original list
        - duplicate_pairs: list of (dup_idx, original_idx, similarity)
        - similarities: max similarity for each idea to previously kept
    """
    if len(ideas) == 0:
        return [], {"kept_indices": [], "duplicate_pairs": [], "similarities": []}
    
    print(f"\n[Deduplication] Starting deduplication of {len(ideas)} ideas")
    embeddings = embed_ideas(ideas)
    
    kept_ideas = [ideas[0]]
    kept_embeddings = [embeddings[0]]
    kept_indices = [0]
    
    duplicate_pairs = []
    similarities = [0.0]  # First idea has no comparison
    
    for i in tqdm(range(1, len(ideas)), desc="Deduplicating"):
        current_embedding = embeddings[i]
        
        # Find maximum similarity with kept ideas
        max_similarity = 0.0
        most_similar_idx = -1
        
        for j, kept_emb in enumerate(kept_embeddings):
            similarity = cosine_similarity(current_embedding, kept_emb)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_idx = kept_indices[j]
        
        similarities.append(max_similarity)
        
        if max_similarity > threshold:
            duplicate_pairs.append((i, most_similar_idx, max_similarity))
        else:
            kept_ideas.append(ideas[i])
            kept_embeddings.append(current_embedding)
            kept_indices.append(i)
    
    info = {
        "kept_indices": kept_indices,
        "duplicate_pairs": duplicate_pairs,
        "similarities": similarities
    }
    
    return kept_ideas, info


# ============================================================================
# DIVERSITY ANALYSIS
# ============================================================================

def analyze_diversity(
    ideas: List,
    batch_size: int = 100,
    threshold: float = 0.8
) -> Dict:
    """
    Analyze diversity metrics as ideas are generated.
    Reproduces Figure 4 from the paper.
    
    Args:
        ideas: List of SeedIdea objects
        batch_size: Size of batches for computing metrics
        threshold: Similarity threshold for duplicate detection
    
    Returns:
        Dict with:
        - batch_non_duplicate_rates: % non-duplicates in each batch
        - cumulative_unique_counts: running count of unique ideas
        - overall_duplicate_rate: total duplicate rate
    """
    if len(ideas) == 0:
        return {
            "batch_non_duplicate_rates": [],
            "cumulative_unique_counts": [],
            "overall_duplicate_rate": 0.0
        }
    
    print(f"\n[Diversity Analysis] Analyzing {len(ideas)} ideas in batches of {batch_size}")
    
    # Compute all embeddings
    embeddings = embed_ideas(ideas)
    
    # Track metrics
    batch_non_duplicate_rates = []
    cumulative_unique_counts = []
    
    # Track kept ideas across all batches
    all_kept_embeddings = []
    all_kept_indices = []
    
    num_batches = (len(ideas) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(ideas))
        batch_embeddings = embeddings[start_idx:end_idx]
        
        # Count non-duplicates in this batch
        batch_non_duplicates = 0
        
        for i, emb in enumerate(batch_embeddings):
            global_idx = start_idx + i
            
            # Check against all previously kept ideas
            is_duplicate = False
            for kept_emb in all_kept_embeddings:
                if cosine_similarity(emb, kept_emb) > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                batch_non_duplicates += 1
                all_kept_embeddings.append(emb)
                all_kept_indices.append(global_idx)
        
        # Compute metrics for this batch
        batch_size_actual = end_idx - start_idx
        non_dup_rate = batch_non_duplicates / batch_size_actual * 100
        batch_non_duplicate_rates.append(non_dup_rate)
        cumulative_unique_counts.append(len(all_kept_embeddings))
        
        print(f"  Batch {batch_idx + 1}/{num_batches}: "
              f"{batch_non_duplicates}/{batch_size_actual} non-duplicates ({non_dup_rate:.1f}%), "
              f"cumulative unique: {len(all_kept_embeddings)}")
    
    overall_duplicate_rate = (1 - len(all_kept_embeddings) / len(ideas)) * 100
    
    result = {
        "batch_non_duplicate_rates": batch_non_duplicate_rates,
        "cumulative_unique_counts": cumulative_unique_counts,
        "overall_duplicate_rate": overall_duplicate_rate,
        "total_unique": len(all_kept_embeddings),
        "total_ideas": len(ideas)
    }
    
    print(f"\n[Diversity Analysis] Summary:")
    print(f"  Total ideas: {len(ideas)}")
    print(f"  Unique ideas: {len(all_kept_embeddings)} ({100-overall_duplicate_rate:.1f}%)")
    print(f"  Duplicate rate: {overall_duplicate_rate:.1f}%")
    
    return result


def find_most_similar_pairs(
    ideas: List,
    top_k: int = 10
) -> List[Tuple[int, int, float]]:
    """
    Find the top-k most similar pairs of ideas.
    
    Useful for debugging and understanding what's being flagged as duplicates.
    
    Returns:
        List of (idx1, idx2, similarity) tuples, sorted by similarity descending
    """
    if len(ideas) < 2:
        return []
    
    print(f"[Finding Similar Pairs] Computing similarities for {len(ideas)} ideas...")
    embeddings = embed_ideas(ideas)
    similarities = compute_pairwise_similarities(embeddings)
    
    # Find top-k pairs (excluding self-similarity)
    pairs = []
    for i in range(len(ideas)):
        for j in range(i + 1, len(ideas)):
            pairs.append((i, j, similarities[i, j]))
    
    # Sort by similarity descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    return pairs[:top_k]


def cluster_similar_ideas(
    ideas: List,
    threshold: float = 0.8
) -> List[List[int]]:
    """
    Group similar ideas into clusters.
    
    Ideas with similarity > threshold are placed in the same cluster.
    Uses a simple greedy clustering approach.
    
    Returns:
        List of clusters, where each cluster is a list of idea indices
    """
    if len(ideas) == 0:
        return []
    
    print(f"[Clustering] Clustering {len(ideas)} ideas (threshold={threshold})...")
    embeddings = embed_ideas(ideas)
    
    clusters = []
    assigned = [False] * len(ideas)
    
    for i in range(len(ideas)):
        if assigned[i]:
            continue
        
        # Start a new cluster with this idea
        cluster = [i]
        assigned[i] = True
        
        # Find all similar ideas
        for j in range(i + 1, len(ideas)):
            if assigned[j]:
                continue
            
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity > threshold:
                cluster.append(j)
                assigned[j] = True
        
        clusters.append(cluster)
    
    print(f"[Clustering] Found {len(clusters)} clusters")
    print(f"  Largest cluster: {max(len(c) for c in clusters)} ideas")
    print(f"  Single-idea clusters: {sum(1 for c in clusters if len(c) == 1)}")
    
    return clusters


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_deduplication_report(
    ideas: List,
    kept_ideas: List,
    info: Dict,
    filepath: str
):
    """Save a detailed deduplication report to file."""
    with open(filepath, 'w') as f:
        f.write("DEDUPLICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Original ideas: {len(ideas)}\n")
        f.write(f"Kept ideas: {len(kept_ideas)}\n")
        f.write(f"Removed: {len(ideas) - len(kept_ideas)}\n")
        f.write(f"Keep rate: {len(kept_ideas)/len(ideas)*100:.1f}%\n\n")
        
        f.write("KEPT IDEAS:\n")
        f.write("-" * 40 + "\n")
        for idx in info.get("kept_indices", []):
            f.write(f"  [{idx}] {ideas[idx].title}\n")
        
        f.write("\nDUPLICATE PAIRS:\n")
        f.write("-" * 40 + "\n")
        for dup_idx, orig_idx, sim in info.get("duplicate_pairs", []):
            f.write(f"  [{dup_idx}] similar to [{orig_idx}] (sim={sim:.3f})\n")
            f.write(f"    Duplicate: {ideas[dup_idx].title}\n")
            f.write(f"    Original:  {ideas[orig_idx].title}\n\n")
    
    print(f"[Deduplication] Report saved to {filepath}")
