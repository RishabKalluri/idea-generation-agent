"""
Paper-to-Proposal Converter Module.

Takes a scraped arXiv paper (with extracted PDF content) and uses an LLM
to rewrite it into the FullProposal outline format used by the idea
generation agent.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Union

from .arxiv_scraper import ArxivPaper


# ============================================================================
# DATA CLASS (mirrors ai_researcher FullProposal)
# ============================================================================

@dataclass
class Proposal:
    """A research proposal in the standard outline format."""
    title: str
    problem_statement: str
    motivation: str
    proposed_method: str
    experiment_plan: str
    test_case_examples: str
    fallback_plan: str

    # Provenance
    source_arxiv_id: str = ""
    source_openreview_id: str = ""
    source_title: str = ""
    source_authors: List[str] = None
    source_published: str = ""

    # Acceptance status (from OpenReview)
    accepted: Optional[bool] = None  # True=accepted, False=rejected, None=unknown

    def __post_init__(self):
        if self.source_authors is None:
            self.source_authors = []

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Proposal":
        # Backward-compatible: ignore unknown keys, fill missing ones
        valid_fields = {
            'title', 'problem_statement', 'motivation', 'proposed_method',
            'experiment_plan', 'test_case_examples', 'fallback_plan',
            'source_arxiv_id', 'source_openreview_id', 'source_title',
            'source_authors', 'source_published', 'accepted',
        }
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return Proposal(**filtered)

    def to_string(self) -> str:
        return f"""1. Title: {self.title}

2. Problem Statement: {self.problem_statement}

3. Motivation: {self.motivation}

4. Proposed Method: {self.proposed_method}

5. Step-by-Step Experiment Plan: {self.experiment_plan}

6. Test Case Examples: {self.test_case_examples}

7. Fallback Plan: {self.fallback_plan}"""


# ============================================================================
# CONVERSION PROMPT
# ============================================================================

CONVERT_PROMPT = """You are an expert AI researcher. Given the content of a published research paper, rewrite it as a research **proposal** — as if it were written *before* the research was conducted, by someone who had the idea but hadn't executed it yet.

Use the exact format below. Write in future tense where appropriate ("we will...", "we propose to..."). The proposal should be detailed enough that another researcher could execute the idea.

FORMAT:
1. Title: [A concise, descriptive title for the proposed research]

2. Problem Statement: [2-4 sentences. What problem does this address? Why is it important?]

3. Motivation: [3-5 sentences. What are the limitations of existing methods? What is the key insight that makes the proposed approach promising?]

4. Proposed Method: [5-10 sentences. Describe the method step by step. Be specific about the approach, architecture, or algorithm.]

5. Step-by-Step Experiment Plan: [Detailed plan with specific datasets, baselines, metrics, and ablations. Include actual prompts or implementation details where relevant.]

6. Test Case Examples: [1-2 concrete examples showing how the method would work on specific inputs. Show the expected input, the steps, and the expected output.]

7. Fallback Plan: [2-3 backup plans if the primary approach doesn't work. What variations would you try? What would a pivot look like?]

---

PAPER TITLE: {title}

AUTHORS: {authors}

PAPER CONTENT:
{content}

---

Now write the proposal. Output ONLY the 7 sections in the format above, nothing else."""


# ============================================================================
# CONVERSION FUNCTION
# ============================================================================

def _call_llm(client, model_name: str, prompt: str, max_tokens: int = 4096) -> str:
    """Call the LLM and return response text."""
    response = client.chat.completions.create(
        model=model_name,
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content if content is not None else ""


def convert_paper_to_proposal(
    paper: ArxivPaper,
    client,
    model_name: str,
) -> Optional[Proposal]:
    """
    Convert a single arXiv paper into a Proposal.

    Args:
        paper: ArxivPaper with content_md populated
        client: OpenAI client
        model_name: Model name for LLM

    Returns:
        Proposal object, or None on failure
    """
    # Use PDF content if available, otherwise fall back to abstract
    content = paper.content_md if paper.content_md else paper.abstract
    if not content or len(content.strip()) < 100:
        return None

    authors_str = ", ".join(paper.authors[:5])
    if len(paper.authors) > 5:
        authors_str += " et al."

    prompt = CONVERT_PROMPT.format(
        title=paper.title,
        authors=authors_str,
        content=content,
    )

    response = _call_llm(client, model_name, prompt, max_tokens=4096)
    return _parse_proposal(response, paper)


def convert_batch(
    papers: List[ArxivPaper],
    client,
    model_name: str,
) -> List[Proposal]:
    """
    Convert a batch of papers to proposals.

    Args:
        papers: List of ArxivPaper objects
        client: OpenAI client
        model_name: Model name

    Returns:
        List of successfully converted Proposals
    """
    proposals = []

    for i, paper in enumerate(papers):
        if (i + 1) % 5 == 0:
            print(f"  Converting {i+1}/{len(papers)}...")

        try:
            proposal = convert_paper_to_proposal(paper, client, model_name)
            if proposal:
                proposals.append(proposal)
        except Exception as e:
            print(f"    Warning: Failed to convert '{paper.title[:40]}...': {e}")

    return proposals


# ============================================================================
# OPENREVIEW CONVERSION
# ============================================================================

def convert_openreview_paper_to_proposal(
    paper,  # OpenReviewPaper
    client,
    model_name: str,
) -> Optional[Proposal]:
    """
    Convert a single OpenReview paper into a Proposal.

    Same logic as convert_paper_to_proposal but sources from OpenReviewPaper
    and attaches acceptance status.
    """
    content = paper.content_md if paper.content_md else paper.abstract
    if not content or len(content.strip()) < 100:
        return None

    authors_str = ", ".join(paper.authors[:5])
    if len(paper.authors) > 5:
        authors_str += " et al."

    prompt = CONVERT_PROMPT.format(
        title=paper.title,
        authors=authors_str,
        content=content,
    )

    response = _call_llm(client, model_name, prompt, max_tokens=4096)

    # Create a minimal ArxivPaper-like object for the parser
    class _PaperProxy:
        pass

    proxy = _PaperProxy()
    proxy.arxiv_id = ""
    proxy.title = paper.title
    proxy.authors = paper.authors
    proxy.published = paper.published

    proposal = _parse_proposal(response, proxy)
    if proposal:
        proposal.source_openreview_id = paper.openreview_id
        proposal.accepted = paper.accepted
    return proposal


def convert_openreview_batch(
    papers,  # List[OpenReviewPaper]
    client,
    model_name: str,
) -> List[Proposal]:
    """
    Convert a batch of OpenReview papers to proposals.
    """
    proposals = []

    for i, paper in enumerate(papers):
        if (i + 1) % 5 == 0:
            print(f"  Converting {i+1}/{len(papers)}...")

        try:
            proposal = convert_openreview_paper_to_proposal(paper, client, model_name)
            if proposal:
                proposals.append(proposal)
        except Exception as e:
            print(f"    Warning: Failed to convert '{paper.title[:40]}...': {e}")

    return proposals


# ============================================================================
# PARSING
# ============================================================================

def _parse_proposal(text: str, paper: ArxivPaper) -> Optional[Proposal]:
    """Parse LLM output into a Proposal dataclass."""
    import re

    sections = {
        "title": "",
        "problem_statement": "",
        "motivation": "",
        "proposed_method": "",
        "experiment_plan": "",
        "test_case_examples": "",
        "fallback_plan": "",
    }

    # Pattern: "1. Title:" or "2. Problem Statement:" etc.
    patterns = [
        (r"1\.\s*Title:\s*(.*?)(?=\n\s*2\.\s*Problem)", "title"),
        (r"2\.\s*Problem Statement:\s*(.*?)(?=\n\s*3\.\s*Motivation)", "problem_statement"),
        (r"3\.\s*Motivation:\s*(.*?)(?=\n\s*4\.\s*Proposed Method)", "motivation"),
        (r"4\.\s*Proposed Method:\s*(.*?)(?=\n\s*5\.\s*(?:Step-by-Step\s*)?Experiment)", "proposed_method"),
        (r"5\.\s*(?:Step-by-Step\s*)?Experiment Plan:\s*(.*?)(?=\n\s*6\.\s*Test Case)", "experiment_plan"),
        (r"6\.\s*Test Case Examples?:\s*(.*?)(?=\n\s*7\.\s*Fallback)", "test_case_examples"),
        (r"7\.\s*Fallback Plan:\s*(.*?)$", "fallback_plan"),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()

    # Require at least title and proposed method
    if not sections["title"] or not sections["proposed_method"]:
        return None

    return Proposal(
        title=sections["title"],
        problem_statement=sections["problem_statement"],
        motivation=sections["motivation"],
        proposed_method=sections["proposed_method"],
        experiment_plan=sections["experiment_plan"],
        test_case_examples=sections["test_case_examples"],
        fallback_plan=sections["fallback_plan"],
        source_arxiv_id=paper.arxiv_id,
        source_title=paper.title,
        source_authors=paper.authors,
        source_published=paper.published,
    )
