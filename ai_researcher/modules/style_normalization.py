"""
Style Normalization Module.

Normalizes the writing style of research proposals to match a template format.
This removes stylistic cues that could reveal human vs AI origin, ensuring
fair comparison during evaluation.
"""

import os
import sys
import re
from typing import List, Optional, Tuple
from tqdm import tqdm

# Load FullProposal class directly to avoid anthropic dependency
import importlib.util

def _load_full_proposal():
    """Load FullProposal class without triggering anthropic import."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    idea_gen_path = os.path.join(current_dir, "idea_generation.py")
    
    spec = importlib.util.spec_from_file_location("idea_gen_direct", idea_gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FullProposal, module.SeedIdea

FullProposal, SeedIdea = _load_full_proposal()


# ============================================================================
# PROMPTS
# ============================================================================

STYLE_NORMALIZATION_PROMPT = """You are a writing assistant specialized in editing academic writing. I will give you a research idea and a template. Your task is to edit the idea to follow the template's format.

Research idea to edit:
{idea_text}

Template to match:
{template_text}

Make sure that you only edit the wording and formatting, including things like punctuation, capitalization, linebreaks, and bullet points. Also make sure to edit any informal wording and phrasing to use vocabulary that sounds like the template's writing style. No other changes are allowed beyond these.

Formatting rules:
- The main sections should be indexed clearly without indentation at the beginning
- The title section does not need indexing; other sections (problem statement, motivation, proposed method, step-by-step experiment plan, test case examples, and fallback plan) should be indexed 1 to 6
- Each section can have sub-bullets for sub-sections if applicable
- Leave an empty line after each section
- Use tab as indentation with appropriate nested indentation for sub-bullets
- All bullets should have a clear hierarchy
- Only leave empty lines between sections and remove any extra line breaks
- For the fallback plan, condense bullet points into one coherent paragraph
- For line breaks, avoid Raw String Literals or Double Backslashes

Citation rules:
- Keep author citations like "(Si et al., 2023)" or "(An et al., 2024)"
- Remove numbered citations like "[1]" or "[3,4,5]" and rephrase if needed

Content preservation rules:
- Do not change any content of the idea
- Preserve the exact meaning of the original idea
- Do not change, remove, or add any other details
- Do not drop any sections (including test case examples)
- Do not rename any models, datasets, or methods
- Do not drop clarification or examples in brackets
- Keep all clarification and examples mentioned in all sections

Model updates:
- If any version of Claude is mentioned, change it to Claude-3.5
- If any version of LLaMA is mentioned, change it to LLaMA-3
- Do not make any other model changes

For the proposed method section, avoid big changes. If it comes as a coherent paragraph, don't break it into bullet points. If it's already in bullet points, keep it that way.

Now directly generate the edited idea to match the format of the template."""


# Simplified prompt for quick normalization
QUICK_NORMALIZATION_PROMPT = """Reformat this research proposal to follow academic writing standards.

Proposal:
{idea_text}

Rules:
1. Use numbered sections (1-6): Problem Statement, Motivation, Proposed Method, Experiment Plan, Test Cases, Fallback Plan
2. Title has no number
3. Use consistent formatting (tabs for indentation, empty lines between sections)
4. Keep author citations like "(Smith et al., 2023)", remove numbered citations like "[1]"
5. Update model names: Claude → Claude-3.5, LLaMA → LLaMA-3
6. Preserve all content exactly, only change formatting and style

Output the reformatted proposal:"""


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def format_proposal_for_normalization(proposal: FullProposal) -> str:
    """Format a FullProposal object as text for normalization."""
    sections = []
    
    sections.append(f"Title: {proposal.title}")
    sections.append("")
    
    if proposal.problem_statement:
        sections.append(f"1. Problem Statement: {proposal.problem_statement}")
        sections.append("")
    
    if proposal.motivation:
        sections.append(f"2. Motivation: {proposal.motivation}")
        sections.append("")
    
    if proposal.proposed_method:
        sections.append(f"3. Proposed Method: {proposal.proposed_method}")
        sections.append("")
    
    if proposal.experiment_plan:
        sections.append(f"4. Step-by-Step Experiment Plan: {proposal.experiment_plan}")
        sections.append("")
    
    if proposal.test_case_examples:
        sections.append(f"5. Test Case Examples: {proposal.test_case_examples}")
        sections.append("")
    
    if proposal.fallback_plan:
        sections.append(f"6. Fallback Plan: {proposal.fallback_plan}")
    
    return "\n".join(sections)


def parse_normalized_text(normalized_text: str, original_proposal: FullProposal) -> FullProposal:
    """
    Parse normalized text back into a FullProposal object.
    Falls back to original content if parsing fails for a section.
    """
    # Extract title
    title_match = re.search(r'Title:\s*(.+?)(?:\n\n|\n1\.)', normalized_text, re.DOTALL)
    title = title_match.group(1).strip() if title_match else original_proposal.title
    
    # Extract sections using flexible patterns
    def extract_section(pattern: str, fallback: str) -> str:
        match = re.search(pattern, normalized_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Clean up any leading colons or whitespace
            content = re.sub(r'^[:\s]+', '', content)
            return content if content else fallback
        return fallback
    
    # Section patterns (flexible to handle various formats)
    problem_statement = extract_section(
        r'1\.\s*Problem Statement[:\s]*(.+?)(?:\n\n|\n2\.|\Z)',
        original_proposal.problem_statement
    )
    
    motivation = extract_section(
        r'2\.\s*Motivation[:\s]*(.+?)(?:\n\n|\n3\.|\Z)',
        original_proposal.motivation
    )
    
    proposed_method = extract_section(
        r'3\.\s*Proposed Method[:\s]*(.+?)(?:\n\n|\n4\.|\Z)',
        original_proposal.proposed_method
    )
    
    experiment_plan = extract_section(
        r'4\.\s*(?:Step-by-Step\s+)?Experiment Plan[:\s]*(.+?)(?:\n\n|\n5\.|\Z)',
        original_proposal.experiment_plan
    )
    
    test_case_examples = extract_section(
        r'5\.\s*Test Case Examples?[:\s]*(.+?)(?:\n\n|\n6\.|\Z)',
        original_proposal.test_case_examples
    )
    
    fallback_plan = extract_section(
        r'6\.\s*Fallback Plan[:\s]*(.+?)(?:\n\n|\Z)',
        original_proposal.fallback_plan
    )
    
    return FullProposal(
        title=title,
        problem_statement=problem_statement,
        motivation=motivation,
        proposed_method=proposed_method,
        experiment_plan=experiment_plan,
        test_case_examples=test_case_examples,
        fallback_plan=fallback_plan,
        raw_text=normalized_text,
        seed_idea=original_proposal.seed_idea
    )


def normalize_style(
    proposal: FullProposal,
    template: str,
    client,
    model_name: str,
    max_retries: int = 2
) -> FullProposal:
    """
    Normalize the style of a proposal to match a template.
    This removes stylistic cues that could reveal human vs AI origin.
    
    Args:
        proposal: The proposal to normalize
        template: Template text to match style
        client: OpenAI client
        model_name: Model to use
        max_retries: Number of retries on failure
    
    Returns:
        Normalized FullProposal
    """
    idea_text = format_proposal_for_normalization(proposal)
    
    prompt = STYLE_NORMALIZATION_PROMPT.format(
        idea_text=idea_text,
        template_text=template
    )
    
    for attempt in range(max_retries + 1):
        try:
            response = _call_llm(client, model_name, prompt, max_tokens=4000)
            normalized = parse_normalized_text(response, proposal)
            
            # Validate that we didn't lose content
            if _validate_normalization(proposal, normalized):
                return normalized
            else:
                print(f"  Warning: Normalization validation failed, attempt {attempt + 1}")
                
        except Exception as e:
            print(f"  Error normalizing proposal: {e}")
            if attempt == max_retries:
                return proposal  # Return original on failure
    
    return proposal


def normalize_style_quick(
    proposal: FullProposal,
    client,
    model_name: str
) -> FullProposal:
    """
    Quick normalization without a template.
    Uses a simplified prompt for basic formatting.
    """
    idea_text = format_proposal_for_normalization(proposal)
    
    prompt = QUICK_NORMALIZATION_PROMPT.format(idea_text=idea_text)
    
    try:
        response = _call_llm(client, model_name, prompt, max_tokens=4000)
        return parse_normalized_text(response, proposal)
    except Exception as e:
        print(f"  Error in quick normalization: {e}")
        return proposal


def normalize_all_proposals(
    proposals: List[FullProposal],
    template_path: str,
    client,
    model_name: str,
    show_progress: bool = True
) -> List[FullProposal]:
    """
    Normalize all proposals to the same style.
    
    Args:
        proposals: List of proposals to normalize
        template_path: Path to template file
        client: OpenAI client
        model_name: Model to use
        show_progress: Show progress bar
    
    Returns:
        List of normalized FullProposal objects
    """
    with open(template_path, 'r') as f:
        template = f.read()
    
    print(f"\n[Style Normalization] Normalizing {len(proposals)} proposals")
    print(f"  Template: {template_path}")
    
    normalized = []
    iterator = proposals
    if show_progress:
        iterator = tqdm(proposals, desc="Normalizing styles")
    
    for proposal in iterator:
        normalized_proposal = normalize_style(proposal, template, client, model_name)
        normalized.append(normalized_proposal)
    
    print(f"[Style Normalization] Complete!")
    return normalized


def normalize_all_quick(
    proposals: List[FullProposal],
    client,
    model_name: str,
    show_progress: bool = True
) -> List[FullProposal]:
    """
    Quick normalization for all proposals without template.
    """
    print(f"\n[Quick Normalization] Normalizing {len(proposals)} proposals")
    
    normalized = []
    iterator = proposals
    if show_progress:
        iterator = tqdm(proposals, desc="Quick normalizing")
    
    for proposal in iterator:
        normalized_proposal = normalize_style_quick(proposal, client, model_name)
        normalized.append(normalized_proposal)
    
    return normalized


# ============================================================================
# VALIDATION
# ============================================================================

def _validate_normalization(original: FullProposal, normalized: FullProposal) -> bool:
    """
    Validate that normalization preserved key content.
    
    Returns:
        True if normalization is valid, False otherwise
    """
    # Check that title is present and similar
    if not normalized.title or len(normalized.title) < 5:
        return False
    
    # Check that major sections are present
    if not normalized.problem_statement:
        return False
    if not normalized.proposed_method:
        return False
    
    # Check that content wasn't drastically shortened
    original_len = len(format_proposal_for_normalization(original))
    normalized_len = len(format_proposal_for_normalization(normalized))
    
    # Allow some reduction but not more than 50%
    if normalized_len < original_len * 0.5:
        return False
    
    return True


def check_style_consistency(proposals: List[FullProposal]) -> dict:
    """
    Check style consistency across proposals.
    
    Returns:
        Dict with consistency metrics
    """
    metrics = {
        "total_proposals": len(proposals),
        "has_numbered_sections": 0,
        "has_title": 0,
        "has_all_sections": 0,
        "avg_length": 0,
        "section_counts": {
            "problem_statement": 0,
            "motivation": 0,
            "proposed_method": 0,
            "experiment_plan": 0,
            "test_case_examples": 0,
            "fallback_plan": 0
        }
    }
    
    total_length = 0
    
    for proposal in proposals:
        text = format_proposal_for_normalization(proposal)
        total_length += len(text)
        
        if proposal.title:
            metrics["has_title"] += 1
        
        # Check numbered sections
        if re.search(r'1\.\s*Problem', text):
            metrics["has_numbered_sections"] += 1
        
        # Count sections
        if proposal.problem_statement:
            metrics["section_counts"]["problem_statement"] += 1
        if proposal.motivation:
            metrics["section_counts"]["motivation"] += 1
        if proposal.proposed_method:
            metrics["section_counts"]["proposed_method"] += 1
        if proposal.experiment_plan:
            metrics["section_counts"]["experiment_plan"] += 1
        if proposal.test_case_examples:
            metrics["section_counts"]["test_case_examples"] += 1
        if proposal.fallback_plan:
            metrics["section_counts"]["fallback_plan"] += 1
        
        # Check if all sections present
        if all([
            proposal.problem_statement,
            proposal.motivation,
            proposal.proposed_method,
            proposal.experiment_plan,
            proposal.test_case_examples,
            proposal.fallback_plan
        ]):
            metrics["has_all_sections"] += 1
    
    metrics["avg_length"] = total_length / len(proposals) if proposals else 0
    
    return metrics


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_citations(text: str) -> str:
    """
    Preprocess citations according to style rules.
    - Keep author citations like "(Si et al., 2023)"
    - Remove numbered citations like "[1]" or "[3,4,5]"
    """
    # Remove numbered citations
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    
    # Clean up any double spaces from removal
    text = re.sub(r'  +', ' ', text)
    
    return text


def update_model_names(text: str) -> str:
    """
    Update model names according to style rules.
    - Claude → Claude-3.5
    - LLaMA → LLaMA-3
    """
    # Update Claude versions
    text = re.sub(r'Claude(?:-\d+(?:\.\d+)?)?(?!-3\.5)', 'Claude-3.5', text, flags=re.IGNORECASE)
    
    # Update LLaMA versions
    text = re.sub(r'LLaMA(?:-\d+)?(?!-3)', 'LLaMA-3', text, flags=re.IGNORECASE)
    text = re.sub(r'Llama(?:-\d+)?(?!-3)', 'LLaMA-3', text, flags=re.IGNORECASE)
    
    return text


def basic_style_cleanup(text: str) -> str:
    """
    Apply basic style cleanup rules without LLM.
    """
    # Preprocess citations
    text = preprocess_citations(text)
    
    # Update model names
    text = update_model_names(text)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' +', ' ', text)  # Single spaces
    
    # Normalize bullet points
    text = re.sub(r'^[-•●]\s*', '- ', text, flags=re.MULTILINE)
    
    return text.strip()


# ============================================================================
# TEMPLATE UTILITIES
# ============================================================================

def load_template(template_path: str) -> str:
    """Load a style template from file."""
    with open(template_path, 'r') as f:
        return f.read()


def get_default_template() -> str:
    """Return the default style template."""
    return """Title: [Descriptive title of the research idea]

1. Problem Statement
	The problem this research addresses is [problem description]. Current approaches [current limitations]. This matters because [importance].

2. Motivation
	The key insight is [key insight]. This is motivated by [motivation]. Unlike existing work, [differentiation].

3. Proposed Method
	We propose [method name/description]. The approach works as follows:
	- Step 1: [description]
	- Step 2: [description]
	- Step 3: [description]

4. Step-by-Step Experiment Plan
	We will evaluate our approach through:
	- Datasets: [dataset names and descriptions]
	- Baselines: [baseline methods]
	- Metrics: [evaluation metrics]
	- Analysis: [planned analyses]

5. Test Case Examples
	Example 1: [Concrete example showing how the method works]
	Example 2: [Another example demonstrating different aspects]

6. Fallback Plan
	If the main approach does not work as expected, we will [fallback strategies]. We can also [alternative approaches] to ensure meaningful results."""


def create_template_from_proposal(proposal: FullProposal) -> str:
    """Create a template from an exemplary proposal."""
    return format_proposal_for_normalization(proposal)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _call_llm(client, model_name: str, prompt: str, max_tokens: int = 4000) -> str:
    """Call OpenAI LLM and return response text."""
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def normalize_batch(
    proposals: List[FullProposal],
    template: str,
    client,
    model_name: str,
    batch_size: int = 5
) -> List[FullProposal]:
    """
    Normalize proposals in batches for better throughput.
    """
    normalized = []
    
    for i in range(0, len(proposals), batch_size):
        batch = proposals[i:i + batch_size]
        
        for proposal in batch:
            try:
                normalized_proposal = normalize_style(
                    proposal, template, client, model_name
                )
                normalized.append(normalized_proposal)
            except Exception as e:
                print(f"Error normalizing: {e}")
                normalized.append(proposal)
    
    return normalized


def save_normalized_proposals(proposals: List[FullProposal], output_path: str):
    """Save normalized proposals to a file."""
    with open(output_path, 'w') as f:
        for i, proposal in enumerate(proposals, 1):
            f.write(f"{'='*60}\n")
            f.write(f"PROPOSAL {i}\n")
            f.write(f"{'='*60}\n\n")
            f.write(format_proposal_for_normalization(proposal))
            f.write("\n\n")
    
    print(f"[Style Normalization] Saved {len(proposals)} proposals to {output_path}")
