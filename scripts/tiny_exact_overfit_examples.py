"""The 32 fixed diagnostic examples for the tiny exact-overfit gate
(FOUNDATIONAL_GENERATION_RECOVERY.md, section B).

Deliberately separate from the sealed 8-item foundational proof gate
(``scripts/eval_foundational_proof_gate.py``) -- this file is allowed to
change; the proof gate is not. Each example is a ``(prompt, response)`` pair
in plain text; ``prompt`` already includes any inline evidence, exactly as
``scripts/eval_foundational_proof_gate.py`` folds evidence into the user
turn rather than the (unused-by-training) ``trusted_evidence`` kwarg.

Eight categories x four items = 32, covering: copying, arithmetic, polite
rewriting, evidence extraction, abstention, simple debugging, temporal
ordering, one-step planning.
"""

from __future__ import annotations

from typing import List, Tuple

TinyExample = Tuple[str, str]

TINY_EXAMPLES: List[TinyExample] = [
    # -- copying (exact echo; tests the model can reproduce input verbatim) --
    ("Repeat exactly: apple", "apple"),
    ("Repeat exactly: 42", "42"),
    ("Repeat exactly: hello world", "hello world"),
    ("Repeat exactly: cat dog bird", "cat dog bird"),
    # -- arithmetic (single-digit/short, deterministic) --
    ("Calculate 2 + 2. Give the numerical answer.", "4"),
    ("Calculate 5 + 3. Give the numerical answer.", "8"),
    ("Calculate 10 - 4. Give the numerical answer.", "6"),
    ("Calculate 6 + 7. Give the numerical answer.", "13"),
    # -- polite rewriting --
    ("Rewrite politely: Give me the file.", "Could you please give me the file?"),
    ("Rewrite politely: Send the report.", "Could you please send the report?"),
    ("Rewrite politely: Close the door.", "Could you please close the door?"),
    ("Rewrite politely: Answer the question.", "Could you please answer the question?"),
    # -- evidence extraction (evidence folded into the prompt, as the proof gate does) --
    ("Using only the trusted evidence, what is the device code?\n\n"
     "Trusted evidence: The device code is 17.", "17"),
    ("Using only the trusted evidence, what color is the box?\n\n"
     "Trusted evidence: The box is red.", "Red."),
    ("Using only the trusted evidence, how many items are there?\n\n"
     "Trusted evidence: There are 9 items.", "9"),
    ("Using only the trusted evidence, what is the room number?\n\n"
     "Trusted evidence: The room number is 12.", "12"),
    # -- abstention (evidence present but does not answer the question) --
    ("Using only the trusted evidence, what date did it launch?\n\n"
     "Trusted evidence: The device code is 17.", "Not provided in the evidence."),
    ("Using only the trusted evidence, who built it?\n\n"
     "Trusted evidence: The box is red.", "Not provided in the evidence."),
    ("Using only the trusted evidence, what is the price?\n\n"
     "Trusted evidence: There are 9 items.", "Not provided in the evidence."),
    ("Using only the trusted evidence, where is it located?\n\n"
     "Trusted evidence: The room number is 12.", "Not provided in the evidence."),
    # -- simple debugging --
    ("This loop should print 1 through 5 but misses 5: "
     "for i in range(1, 5): print(i). State the smallest fix.",
     "Use range(1, 6)."),
    ("This loop should print 0 through 3 but misses 3: "
     "for i in range(0, 3): print(i). State the smallest fix.",
     "Use range(0, 4)."),
    ("This loop should print 1 through 4 but misses 4: "
     "for i in range(1, 4): print(i). State the smallest fix.",
     "Use range(1, 5)."),
    ("This loop should print 2 through 6 but misses 6: "
     "for i in range(2, 6): print(i). State the smallest fix.",
     "Use range(2, 7)."),
    # -- temporal ordering --
    ("Event A is at 09:00. B happens 1 hour after A. What time is B?", "10:00."),
    ("Event A is at 10:00. B happens 2 hours after A. What time is B?", "12:00."),
    ("Event A is at 08:00. B happens 3 hours after A. What time is B?", "11:00."),
    ("Event A is at 14:00. B happens 1 hour after A. What time is B?", "15:00."),
    # -- one-step planning --
    ("Give one step to find every file containing the text TODO.",
     "Search the repository text for the string TODO."),
    ("Give one step to find every Python file in a folder.",
     "List the folder and keep files ending in .py."),
    ("Give one step to check if a file exists before reading it.",
     "Check the file path with an existence test first."),
    ("Give one step to count the lines in a text file.",
     "Read the file and count its newline characters."),
]

assert len(TINY_EXAMPLES) == 32, f"expected 32 examples, got {len(TINY_EXAMPLES)}"

CATEGORY_BOUNDARIES = {
    "copying": (0, 4), "arithmetic": (4, 8), "polite_rewriting": (8, 12),
    "evidence_extraction": (12, 16), "abstention": (16, 20),
    "simple_debugging": (20, 24), "temporal_ordering": (24, 28),
    "one_step_planning": (28, 32),
}

__all__ = ["TinyExample", "TINY_EXAMPLES", "CATEGORY_BOUNDARIES"]
