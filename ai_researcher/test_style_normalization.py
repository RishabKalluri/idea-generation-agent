#!/usr/bin/env python3
"""Test script for the style normalization module."""

import os
import sys

def test_module_loads():
    """Test that the module can be loaded without anthropic."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        'style_norm', 
        'modules/style_normalization.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    print("✓ Module loaded successfully")
    print(f"  normalize_style: {module.normalize_style}")
    print(f"  normalize_all_proposals: {module.normalize_all_proposals}")
    print(f"  format_proposal_for_normalization: {module.format_proposal_for_normalization}")
    print(f"  parse_normalized_text: {module.parse_normalized_text}")
    print(f"  FullProposal: {module.FullProposal}")
    
    return module


def test_format_proposal(module):
    """Test proposal formatting."""
    FullProposal = module.FullProposal
    
    proposal = FullProposal(
        title="Self-Correcting Language Models",
        problem_statement="LLMs often generate errors without self-detection",
        motivation="Self-verification could significantly improve reliability",
        proposed_method="Add a verification step after each generation phase",
        experiment_plan="Test on QA benchmarks and math reasoning tasks",
        test_case_examples="Example: Q: What is 2+2? A: 5. Verify: incorrect, should be 4.",
        fallback_plan="If full verification fails, use simpler heuristic checks"
    )
    
    formatted = module.format_proposal_for_normalization(proposal)
    
    print(f"\n✓ Formatted proposal:")
    print("-" * 40)
    print(formatted[:500])
    print("-" * 40)
    
    # Check that all sections are present
    assert "Title:" in formatted
    assert "1. Problem Statement" in formatted
    assert "2. Motivation" in formatted
    assert "3. Proposed Method" in formatted
    assert "4. Step-by-Step Experiment Plan" in formatted
    assert "5. Test Case Examples" in formatted
    assert "6. Fallback Plan" in formatted
    
    print("✓ All sections present in formatted output")
    
    return proposal


def test_parse_normalized(module):
    """Test parsing normalized text back to FullProposal."""
    FullProposal = module.FullProposal
    
    # Create a normalized text string
    normalized_text = """Title: Self-Correcting Language Models for Improved Reliability

1. Problem Statement: Large language models frequently generate factual errors and logical inconsistencies without any self-detection mechanism.

2. Motivation: By incorporating self-verification capabilities, we can significantly improve the reliability and trustworthiness of LLM outputs.

3. Proposed Method: We propose adding a dedicated verification step after each generation phase that checks for consistency and factual accuracy.

4. Step-by-Step Experiment Plan: We will evaluate our approach on:
	- QA benchmarks (SQuAD, TriviaQA)
	- Math reasoning tasks (GSM8K, MATH)
	- Logical reasoning (LogiQA)

5. Test Case Examples: 
	Example 1: Q: What is 2+2? Initial A: 5. Verification: 2+2=4, not 5. Corrected A: 4.
	Example 2: Q: Who wrote Hamlet? Initial A: Charles Dickens. Verification: Shakespeare wrote Hamlet. Corrected A: Shakespeare.

6. Fallback Plan: If the full verification mechanism proves too computationally expensive, we will implement simpler heuristic-based checks focusing on numerical consistency and named entity verification."""

    # Create original proposal for fallback
    original = FullProposal(
        title="Original Title",
        problem_statement="Original problem",
        motivation="Original motivation",
        proposed_method="Original method",
        experiment_plan="Original experiment",
        test_case_examples="Original examples",
        fallback_plan="Original fallback"
    )
    
    parsed = module.parse_normalized_text(normalized_text, original)
    
    print("\n✓ Parsed normalized text:")
    print(f"  Title: {parsed.title[:50]}...")
    print(f"  Problem: {parsed.problem_statement[:50]}...")
    print(f"  Method: {parsed.proposed_method[:50]}...")
    
    assert "Self-Correcting" in parsed.title
    assert "language models" in parsed.problem_statement.lower()
    assert "verification" in parsed.proposed_method.lower()
    
    print("✓ Parsing preserves content correctly")


def test_citation_preprocessing(module):
    """Test citation preprocessing."""
    # Test numbered citation removal
    text_with_citations = "This is supported by prior work [1] and others [3,4,5]. See also (Smith et al., 2023)."
    processed = module.preprocess_citations(text_with_citations)
    
    assert "[1]" not in processed
    assert "[3,4,5]" not in processed
    assert "(Smith et al., 2023)" in processed
    
    print("\n✓ Citation preprocessing works:")
    print(f"  Input:  {text_with_citations}")
    print(f"  Output: {processed}")


def test_model_name_updates(module):
    """Test model name updates."""
    text = "We use Claude-2 and LLaMA-2 models. Also tested with Claude and Llama."
    updated = module.update_model_names(text)
    
    print(f"\n✓ Model name updates:")
    print(f"  Input:  {text}")
    print(f"  Output: {updated}")
    
    assert "Claude-3.5" in updated
    assert "LLaMA-3" in updated


def test_basic_cleanup(module):
    """Test basic style cleanup."""
    text = """This has [1] citations and [2,3] more.


Too many newlines above.

• Bullet 1
- Bullet 2

Uses Claude-2 and Llama-7B."""

    cleaned = module.basic_style_cleanup(text)
    
    print(f"\n✓ Basic cleanup:")
    print(f"  Original length: {len(text)}")
    print(f"  Cleaned length: {len(cleaned)}")
    print(f"  Cleaned text:\n{cleaned}")
    
    assert "[1]" not in cleaned
    assert "Claude-3.5" in cleaned
    assert "LLaMA-3" in cleaned


def test_default_template(module):
    """Test default template."""
    template = module.get_default_template()
    
    print(f"\n✓ Default template ({len(template)} chars):")
    print("-" * 40)
    print(template[:300])
    print("...")
    print("-" * 40)
    
    assert "Title:" in template
    assert "1. Problem Statement" in template
    assert "6. Fallback Plan" in template


def test_style_consistency_check(module):
    """Test style consistency checking."""
    FullProposal = module.FullProposal
    
    proposals = [
        FullProposal(
            title="Proposal 1",
            problem_statement="Problem 1",
            motivation="Motivation 1",
            proposed_method="Method 1",
            experiment_plan="Experiment 1",
            test_case_examples="Examples 1",
            fallback_plan="Fallback 1"
        ),
        FullProposal(
            title="Proposal 2",
            problem_statement="Problem 2",
            motivation="",  # Missing motivation
            proposed_method="Method 2",
            experiment_plan="Experiment 2",
            test_case_examples="",  # Missing examples
            fallback_plan="Fallback 2"
        ),
    ]
    
    metrics = module.check_style_consistency(proposals)
    
    print(f"\n✓ Style consistency metrics:")
    print(f"  Total proposals: {metrics['total_proposals']}")
    print(f"  Has all sections: {metrics['has_all_sections']}")
    print(f"  Section counts: {metrics['section_counts']}")


def test_validation(module):
    """Test normalization validation."""
    FullProposal = module.FullProposal
    
    original = FullProposal(
        title="Test Proposal Title",
        problem_statement="A detailed problem statement that describes the issue.",
        motivation="Strong motivation for this research.",
        proposed_method="Novel method with multiple steps.",
        experiment_plan="Comprehensive experiment plan.",
        test_case_examples="Example cases.",
        fallback_plan="Backup strategies."
    )
    
    # Good normalization (preserves content)
    good_normalized = FullProposal(
        title="Test Proposal Title",
        problem_statement="A detailed problem statement that describes the issue.",
        motivation="Strong motivation for this research direction.",
        proposed_method="Novel method with multiple steps for solving the problem.",
        experiment_plan="Comprehensive experiment plan.",
        test_case_examples="Example cases demonstrating the approach.",
        fallback_plan="Backup strategies if needed."
    )
    
    # Bad normalization (lost content)
    bad_normalized = FullProposal(
        title="Test",
        problem_statement="",  # Lost problem
        motivation="",
        proposed_method="Method",
        experiment_plan="",
        test_case_examples="",
        fallback_plan=""
    )
    
    assert module._validate_normalization(original, good_normalized) == True
    assert module._validate_normalization(original, bad_normalized) == False
    
    print("\n✓ Validation correctly identifies good and bad normalizations")


if __name__ == "__main__":
    print("="*60)
    print("STYLE NORMALIZATION MODULE TESTS")
    print("="*60)
    
    module = test_module_loads()
    test_format_proposal(module)
    test_parse_normalized(module)
    test_citation_preprocessing(module)
    test_model_name_updates(module)
    test_basic_cleanup(module)
    test_default_template(module)
    test_style_consistency_check(module)
    test_validation(module)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
