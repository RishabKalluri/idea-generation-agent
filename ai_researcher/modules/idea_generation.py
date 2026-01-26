"""
Idea Generation Module.

Handles seed idea generation and full proposal generation using LLMs.
Uses direct file loading to avoid anthropic dependency.
"""

import os
import re
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm

# Load Paper class directly to avoid anthropic import
import importlib.util

def _load_paper_class():
    """Load Paper class without triggering anthropic import."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    ss_path = os.path.join(parent_dir, "utils", "semantic_scholar.py")
    
    if not os.path.exists(ss_path):
        # Fallback: define a simple Paper type
        return None
    
    spec = importlib.util.spec_from_file_location("semantic_scholar_direct", ss_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Paper

Paper = _load_paper_class()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SeedIdea:
    """Dataclass representing a seed research idea."""
    title: str
    problem: str
    existing_methods: str
    motivation: str
    proposed_method: str
    experiment_plan: str
    raw_text: str = ""
    rag_used: bool = False
    
    def to_string(self) -> str:
        """Convert to formatted string."""
        return f"""Title: {self.title}

Problem: {self.problem}

Existing Methods: {self.existing_methods}

Motivation: {self.motivation}

Proposed Method: {self.proposed_method}

Experiment Plan: {self.experiment_plan}"""


# ============================================================================
# PROMPTS AND TEMPLATES
# ============================================================================

SEED_IDEA_FORMAT = """Title: [Concise title for the idea]

Problem: [1-2 sentences describing the problem]

Existing Methods: [1-2 sentences on current approaches and their limitations]

Motivation: [2-3 sentences on why the proposed approach should work better]

Proposed Method: [3-5 sentences describing the key steps of the method]

Experiment Plan: [2-3 sentences on how to evaluate the method]"""


SEED_IDEA_PROMPT = """You are an expert NLP researcher tasked with generating novel research ideas.

Research Topic: {topic}

{rag_section}

Here are some example ideas in the format you should follow:

{demo_examples}

Previously generated ideas (avoid duplicating these):
{previous_titles}

Generate a novel, creative research idea on this topic. The idea should:
1. Address a real problem in the field
2. Be different from existing methods and the previously generated ideas
3. Be feasible to implement with API access to LLMs (no extensive GPU training)
4. Have a clear evaluation plan

Respond with your idea in this exact format:
{seed_idea_format}"""


RAG_SECTION_TEMPLATE = """Here are some relevant papers to inspire your idea (but don't just copy them):

{papers}"""


# ============================================================================
# DEMO EXAMPLE LOADING
# ============================================================================

def get_demo_examples_dir() -> str:
    """Get the path to the demo examples directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, "data", "demo_examples")


def load_demo_examples(num_examples: int = 6) -> List[str]:
    """
    Load demonstration examples from files.
    
    Args:
        num_examples: Number of examples to load
        
    Returns:
        List of example strings
    """
    demo_dir = get_demo_examples_dir()
    examples = []
    
    # Try to load from files
    if os.path.exists(demo_dir):
        example_files = sorted([
            f for f in os.listdir(demo_dir) 
            if f.endswith('.txt') and f.startswith('example_')
        ])
        
        for filename in example_files[:num_examples]:
            filepath = os.path.join(demo_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    examples.append(f.read().strip())
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    # If no files found, use built-in examples
    if not examples:
        examples = _get_builtin_examples()[:num_examples]
    
    return examples


def _get_builtin_examples() -> List[str]:
    """Return built-in demonstration examples based on real papers."""
    return [
        # Example 1: Based on Chain-of-Verification (Dhuliawala et al., 2023)
        """Title: Chain-of-Verification for Reducing LLM Hallucinations

Problem: Large language models often generate plausible-sounding but factually incorrect information, especially for knowledge-intensive tasks.

Existing Methods: Current approaches include retrieval augmentation and fine-tuning on curated data, but these require external knowledge bases or expensive training.

Motivation: By having the model generate verification questions about its own output and then answer them independently, we can catch inconsistencies without external resources. This self-verification approach leverages the model's own knowledge more effectively.

Proposed Method: First, generate an initial response to a query. Then, automatically generate verification questions targeting factual claims in the response. Answer each verification question independently (without seeing the original response). Finally, compare answers to identify and correct inconsistencies.

Experiment Plan: Evaluate on fact-verification benchmarks like FEVER and knowledge-intensive QA datasets. Compare against baseline LLMs and retrieval-augmented methods. Measure both factual accuracy and hallucination rate.""",

        # Example 2: Based on Self-Refine (Madaan et al., 2023)
        """Title: Iterative Self-Refinement for Text Generation

Problem: LLM outputs often contain errors or suboptimal content that could be improved with revision, but current approaches generate text in a single pass.

Existing Methods: Existing methods rely on human feedback or fine-tuning for improvement, which is expensive and doesn't generalize across tasks.

Motivation: Humans naturally iterate on their writing through drafting and revision. By prompting LLMs to critique their own outputs and then refine based on that critique, we can improve quality without additional training.

Proposed Method: Generate an initial output for a given task. Prompt the model to provide specific feedback on the output's weaknesses. Use the feedback to generate an improved version. Repeat the critique-refine cycle until quality converges or a maximum number of iterations is reached.

Experiment Plan: Test on diverse generation tasks including code generation, math reasoning, and open-ended writing. Compare single-pass generation vs. iterative refinement. Measure improvement across iterations and computational cost trade-offs.""",

        # Example 3: Based on Constitutional AI concepts
        """Title: Principle-Guided Response Generation

Problem: LLMs may generate harmful, biased, or unhelpful content without explicit guardrails during generation.

Existing Methods: RLHF and content filtering work but require extensive human labeling or may block legitimate content.

Motivation: By explicitly conditioning generation on a set of principles (like helpfulness, harmlessness, and honesty), the model can self-regulate during generation rather than relying solely on post-hoc filtering.

Proposed Method: Define a set of guiding principles as natural language instructions. During generation, periodically pause to evaluate whether the current output aligns with each principle. If violations are detected, revise the problematic section. Continue generation with the corrected context.

Experiment Plan: Evaluate on safety benchmarks and helpfulness metrics. Compare against baseline models and RLHF-trained models. Measure both safety improvements and potential impacts on helpfulness.""",

        # Example 4: Based on Tree-of-Thoughts concepts
        """Title: Parallel Exploration of Reasoning Paths

Problem: Complex reasoning tasks require exploring multiple solution approaches, but standard autoregressive generation commits to a single path.

Existing Methods: Chain-of-thought prompting helps but still follows a linear reasoning path that may lead to dead ends.

Motivation: Human problem-solving often involves considering multiple approaches simultaneously and backtracking when one fails. Allowing LLMs to explore multiple reasoning branches in parallel can find better solutions.

Proposed Method: Given a complex problem, generate multiple initial reasoning steps representing different approaches. Expand each branch by generating subsequent reasoning steps. Evaluate branches using the model's own assessment of progress and correctness. Prune unpromising branches and expand promising ones. Return the best complete solution found.

Experiment Plan: Test on mathematical reasoning, planning tasks, and puzzle-solving benchmarks. Compare against chain-of-thought and self-consistency methods. Analyze the trade-off between exploration breadth and computational cost.""",

        # Example 5: Based on ReAct concepts  
        """Title: Interleaved Reasoning and Action for Task Completion

Problem: LLMs struggle with tasks requiring interaction with external tools or environments because they generate complete responses without intermediate feedback.

Existing Methods: Tool-augmented LLMs exist but typically separate reasoning from action, leading to suboptimal tool use.

Motivation: By interleaving reasoning traces with actions and observations, the model can adapt its strategy based on intermediate results, similar to how humans think while doing.

Proposed Method: For each step, generate a thought explaining the current reasoning. Based on the thought, generate an action to take (e.g., search, calculate, lookup). Execute the action and observe the result. Incorporate the observation into the next reasoning step. Continue until the task is complete.

Experiment Plan: Evaluate on multi-step reasoning tasks requiring tool use, such as question answering with search and mathematical problem solving. Compare against pure reasoning and pure action-based approaches. Measure task completion rate and efficiency.""",

        # Example 6: Based on retrieval augmentation concepts
        """Title: Adaptive Retrieval Augmentation Based on Query Complexity

Problem: Retrieval-augmented generation retrieves documents for all queries, but some queries don't need external knowledge while others need extensive retrieval.

Existing Methods: Current RAG systems use fixed retrieval strategies regardless of query complexity, wasting computation or missing relevant information.

Motivation: By first assessing query complexity and knowledge requirements, the system can adaptively decide how much retrieval is needed, optimizing both accuracy and efficiency.

Proposed Method: Classify incoming queries into complexity levels based on knowledge requirements. For simple queries, generate directly without retrieval. For medium complexity, retrieve a small set of documents. For complex queries, perform iterative retrieval with query reformulation. Aggregate retrieved information and generate the final response.

Experiment Plan: Test on QA datasets spanning different complexity levels. Compare against fixed retrieval strategies. Measure accuracy, latency, and computational cost across query types."""
    ]


# ============================================================================
# PAPER FORMATTING FOR RAG
# ============================================================================

def format_papers_for_rag(papers: List, k: int = 10) -> str:
    """
    Randomly select k papers and format their titles/abstracts for RAG context.
    
    Args:
        papers: List of Paper objects
        k: Number of papers to include
        
    Returns:
        Formatted string with paper titles and abstracts
    """
    if not papers:
        return ""
    
    # Randomly select papers
    selected = random.sample(papers, min(k, len(papers)))
    
    formatted_papers = []
    for i, paper in enumerate(selected, 1):
        paper_str = f"[Paper {i}]\nTitle: {paper.title}"
        if paper.abstract:
            # Truncate long abstracts
            abstract = paper.abstract[:500]
            if len(paper.abstract) > 500:
                abstract += "..."
            paper_str += f"\nAbstract: {abstract}"
        formatted_papers.append(paper_str)
    
    return "\n\n".join(formatted_papers)


# ============================================================================
# SEED IDEA PARSING
# ============================================================================

def parse_seed_idea(response: str) -> Optional[SeedIdea]:
    """
    Parse LLM response into SeedIdea dataclass.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        SeedIdea object or None if parsing fails
    """
    try:
        # Extract each section using regex
        title_match = re.search(r'Title:\s*(.+?)(?=\n\n|\nProblem:)', response, re.DOTALL)
        problem_match = re.search(r'Problem:\s*(.+?)(?=\n\n|\nExisting Methods:)', response, re.DOTALL)
        existing_match = re.search(r'Existing Methods:\s*(.+?)(?=\n\n|\nMotivation:)', response, re.DOTALL)
        motivation_match = re.search(r'Motivation:\s*(.+?)(?=\n\n|\nProposed Method:)', response, re.DOTALL)
        method_match = re.search(r'Proposed Method:\s*(.+?)(?=\n\n|\nExperiment Plan:)', response, re.DOTALL)
        experiment_match = re.search(r'Experiment Plan:\s*(.+?)(?=\n\n|$)', response, re.DOTALL)
        
        # Check if all required fields are present
        if not all([title_match, problem_match, existing_match, motivation_match, method_match, experiment_match]):
            # Try alternative parsing
            return _parse_seed_idea_fallback(response)
        
        return SeedIdea(
            title=title_match.group(1).strip(),
            problem=problem_match.group(1).strip(),
            existing_methods=existing_match.group(1).strip(),
            motivation=motivation_match.group(1).strip(),
            proposed_method=method_match.group(1).strip(),
            experiment_plan=experiment_match.group(1).strip(),
            raw_text=response
        )
    except Exception as e:
        print(f"Warning: Could not parse seed idea: {e}")
        return None


def _parse_seed_idea_fallback(response: str) -> Optional[SeedIdea]:
    """Fallback parser for seed ideas with slightly different formatting."""
    try:
        lines = response.strip().split('\n')
        sections = {}
        current_section = None
        current_content = []
        
        section_markers = ['title:', 'problem:', 'existing methods:', 'motivation:', 
                         'proposed method:', 'experiment plan:']
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this is a section header
            matched_section = None
            for marker in section_markers:
                if line_lower.startswith(marker):
                    matched_section = marker.replace(':', '')
                    break
            
            if matched_section:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = matched_section
                # Get content after the marker
                content_start = line.lower().find(':') + 1
                first_line_content = line[content_start:].strip()
                current_content = [first_line_content] if first_line_content else []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Build SeedIdea
        if 'title' in sections:
            return SeedIdea(
                title=sections.get('title', ''),
                problem=sections.get('problem', ''),
                existing_methods=sections.get('existing methods', ''),
                motivation=sections.get('motivation', ''),
                proposed_method=sections.get('proposed method', ''),
                experiment_plan=sections.get('experiment plan', ''),
                raw_text=response
            )
        
        return None
    except Exception:
        return None


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_seed_ideas(
    topic: str,
    papers: List,
    client,
    model_name: str,
    num_ideas: int = 4000,
    rag_rate: float = 0.5,
    num_demo_examples: int = 6,
    papers_per_rag: int = 10,
    show_progress: bool = True
) -> List[SeedIdea]:
    """
    Generate seed ideas for a research topic.
    
    Args:
        topic: Research topic description
        papers: List of Paper objects for RAG context
        client: OpenAI client
        model_name: Model to use (e.g., "gpt-4")
        num_ideas: Number of ideas to generate
        rag_rate: Fraction of ideas to use RAG for (0.0 to 1.0)
        num_demo_examples: Number of demonstration examples to include
        papers_per_rag: Number of papers to include in RAG context
        show_progress: Whether to show progress bar
        
    Returns:
        List of SeedIdea objects
    """
    print(f"\n[Seed Idea Generation] Starting generation")
    print(f"  Topic: {topic}")
    print(f"  Target: {num_ideas} ideas")
    print(f"  RAG rate: {rag_rate*100:.0f}%")
    print(f"  Model: {model_name}")
    
    # Load demo examples
    demo_examples = load_demo_examples(num_demo_examples)
    demo_examples_str = "\n\n---\n\n".join([f"Example {i+1}:\n{ex}" for i, ex in enumerate(demo_examples)])
    
    print(f"  Demo examples: {len(demo_examples)}")
    print(f"  Papers for RAG: {len(papers) if papers else 0}")
    
    # Track generated ideas
    generated_ideas = []
    previous_titles = []
    failed_attempts = 0
    max_failed = min(100, num_ideas // 10)  # Allow some failures
    
    # Progress bar
    iterator = tqdm(range(num_ideas), desc="Generating ideas", disable=not show_progress)
    
    for i in iterator:
        # Decide whether to use RAG
        use_rag = papers and (random.random() < rag_rate)
        
        # Build RAG section
        if use_rag:
            papers_str = format_papers_for_rag(papers, k=papers_per_rag)
            rag_section = RAG_SECTION_TEMPLATE.format(papers=papers_str)
        else:
            rag_section = ""
        
        # Build previous titles (show last 20 to avoid prompt being too long)
        if previous_titles:
            recent_titles = previous_titles[-20:]
            previous_titles_str = "\n".join([f"- {t}" for t in recent_titles])
        else:
            previous_titles_str = "(None yet)"
        
        # Build prompt
        prompt = SEED_IDEA_PROMPT.format(
            topic=topic,
            rag_section=rag_section,
            demo_examples=demo_examples_str,
            previous_titles=previous_titles_str,
            seed_idea_format=SEED_IDEA_FORMAT
        )
        
        try:
            # Call LLM
            response = _call_llm_openai(client, model_name, prompt, max_tokens=1024)
            
            # Parse response
            idea = parse_seed_idea(response)
            
            if idea:
                idea.rag_used = use_rag
                generated_ideas.append(idea)
                previous_titles.append(idea.title)
                
                # Update progress bar
                iterator.set_postfix({
                    "generated": len(generated_ideas),
                    "rag": "✓" if use_rag else "✗"
                })
            else:
                failed_attempts += 1
                if failed_attempts >= max_failed:
                    print(f"\nWarning: Too many parsing failures ({failed_attempts})")
                    
        except Exception as e:
            failed_attempts += 1
            if failed_attempts % 10 == 0:
                print(f"\nWarning: Error generating idea {i+1}: {e}")
    
    print(f"\n[Seed Idea Generation] Complete!")
    print(f"  Generated: {len(generated_ideas)} ideas")
    print(f"  Failed: {failed_attempts} attempts")
    
    rag_count = sum(1 for idea in generated_ideas if idea.rag_used)
    print(f"  With RAG: {rag_count} ({rag_count*100/len(generated_ideas):.1f}%)")
    
    return generated_ideas


def _call_llm_openai(client, model_name: str, prompt: str, max_tokens: int = 1024) -> str:
    """Call OpenAI LLM and return response text."""
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ============================================================================
# FULL PROPOSAL GENERATION (Placeholder for future implementation)
# ============================================================================

def generate_full_proposal(seed_idea: SeedIdea, rag_context: Optional[str], 
                          client, model_name: str) -> str:
    """
    Generate full research proposal from seed idea.
    
    Args:
        seed_idea: Initial seed idea
        rag_context: Optional RAG context from papers
        client: OpenAI client
        model_name: Model to use
        
    Returns:
        Full research proposal text
    """
    # TODO: Implement full proposal generation
    # This will expand the seed idea into a full proposal with:
    # - Detailed methodology
    # - Related work analysis
    # - Specific experiments
    # - Expected results
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_ideas_to_file(ideas: List[SeedIdea], filepath: str):
    """Save generated ideas to a text file."""
    with open(filepath, 'w') as f:
        for i, idea in enumerate(ideas, 1):
            f.write(f"{'='*80}\n")
            f.write(f"IDEA {i}\n")
            f.write(f"{'='*80}\n\n")
            f.write(idea.to_string())
            f.write(f"\n\n[RAG used: {idea.rag_used}]\n\n")
    print(f"Saved {len(ideas)} ideas to {filepath}")


def load_ideas_from_file(filepath: str) -> List[SeedIdea]:
    """Load ideas from a text file."""
    ideas = []
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by idea separator
    idea_blocks = re.split(r'={80}\nIDEA \d+\n={80}', content)
    
    for block in idea_blocks:
        if block.strip():
            idea = parse_seed_idea(block)
            if idea:
                ideas.append(idea)
    
    return ideas
