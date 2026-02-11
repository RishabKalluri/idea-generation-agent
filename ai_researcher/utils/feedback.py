"""
Human Feedback Utility Module.

Manages loading, saving, and formatting human feedback for the pipeline.
Feedback is stored in a local file (gitignored) so each user has their own.
"""

import os
from datetime import datetime
from typing import Optional

# Default feedback file location (inside config/, gitignored)
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FEEDBACK_FILE = os.path.join(_SCRIPT_DIR, "config", "human_feedback.txt")


def get_feedback_path() -> str:
    """Get the path to the feedback file, respecting settings if available."""
    try:
        import importlib.util
        settings_path = os.path.join(_SCRIPT_DIR, "config", "settings.py")
        spec = importlib.util.spec_from_file_location("settings", settings_path)
        settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(settings)
        return getattr(settings, 'HUMAN_FEEDBACK_FILE', DEFAULT_FEEDBACK_FILE)
    except Exception:
        return DEFAULT_FEEDBACK_FILE


def load_feedback() -> str:
    """
    Load all accumulated feedback from the feedback file.
    
    Returns:
        Combined feedback string, or empty string if no feedback exists.
    """
    path = get_feedback_path()
    if not os.path.exists(path):
        return ""
    
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except Exception:
        return ""


def append_feedback(feedback_text: str, topic: str = ""):
    """
    Append a new feedback entry to the feedback file.
    
    Args:
        feedback_text: The feedback from the human reviewer
        topic: The research topic this feedback relates to
    """
    path = get_feedback_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    entry = f"\n{'='*60}\n"
    entry += f"Date: {timestamp}\n"
    if topic:
        entry += f"Topic: {topic}\n"
    entry += f"{'='*60}\n"
    entry += f"{feedback_text}\n"
    
    with open(path, 'a') as f:
        f.write(entry)


def clear_feedback():
    """Clear all accumulated feedback."""
    path = get_feedback_path()
    if os.path.exists(path):
        os.remove(path)


def format_feedback_for_prompt(feedback: str) -> str:
    """
    Format accumulated feedback into a system-level instruction for LLM calls.
    
    Args:
        feedback: Raw feedback text
        
    Returns:
        Formatted instruction string to prepend to system messages
    """
    if not feedback:
        return ""
    
    return (
        "CRITICAL — HUMAN REVIEWER FEEDBACK FROM PAST OUTPUTS:\n"
        "The following feedback was provided by a human reviewer on previous "
        "outputs from this pipeline. You MUST take this feedback into account "
        "and adjust your output accordingly. Address the concerns raised and "
        "avoid repeating the same issues.\n\n"
        f"{feedback}\n\n"
        "END OF REVIEWER FEEDBACK.\n"
        "Keep this feedback in mind throughout your response.\n"
    )


def get_formatted_feedback() -> str:
    """
    Load feedback and return it formatted for LLM prompts.
    Convenience function combining load + format.
    
    Returns:
        Formatted feedback string ready for injection, or empty string.
    """
    raw = load_feedback()
    return format_feedback_for_prompt(raw)


def collect_feedback_interactive(topic: str = "") -> Optional[str]:
    """
    Interactively collect feedback from the user via stdin.
    
    Returns:
        The feedback text, or None if the user chose to skip.
    """
    print("\n" + "=" * 60)
    print("HUMAN-IN-THE-LOOP FEEDBACK")
    print("=" * 60)
    print("\nReview the proposals above. You can provide feedback that will")
    print("be saved and used to guide future pipeline runs.")
    print("\nType your feedback below (multi-line). When done:")
    print("  - Press Enter twice (empty line) to submit")
    print("  - Type 'skip' to skip without saving")
    print("  - Type 'clear' to clear all previous feedback")
    print("-" * 60)
    
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\nFeedback collection cancelled.")
            return None
        
        stripped = line.strip()
        
        # Check for commands
        if stripped.lower() == 'skip' and not lines:
            print("Skipping feedback.")
            return None
        
        if stripped.lower() == 'clear' and not lines:
            clear_feedback()
            print("✓ All previous feedback cleared.")
            return None
        
        # Empty line after content = submit
        if stripped == "" and lines:
            break
        
        if stripped != "":
            lines.append(line)
    
    feedback_text = "\n".join(lines)
    
    if feedback_text.strip():
        append_feedback(feedback_text, topic=topic)
        print(f"\n✓ Feedback saved to: {get_feedback_path()}")
        return feedback_text
    
    return None
