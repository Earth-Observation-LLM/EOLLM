"""prompt.py — MCQ prompt construction for the suite.

Design goals, in order:
  1. Comparable across every model — identical text, only vision-token placement
     differs (handled by each model's processor).
  2. Constrain the answer to a SINGLE LETTER. For ATTENTION WATCH this is
     essential: the whole answer is one token, so the attention on it is the
     decision-moment attention with nothing diluting it, and the choice token is
     trivially the one that decodes to A/B/C/D.
  3. Label the images explicitly as "Image 1 .. Image N", tied to the order they
     are passed, so the answer letters never collide with image identity.
  4. Describe only the task, never the measurement — revealing that attention is
     being measured could shift behaviour and contaminate the pattern.

The system prompt matches training (config.SYSTEM_PROMPT shape) so trained LoRA
models behave as they were tuned to.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert at analyzing satellite and street-level imagery of urban "
    "and natural environments. Answer the multiple-choice question by replying "
    "with ONLY the letter of the best option. Do not explain."
)


def build_user_text(record: dict, image_roles: list[str]) -> str:
    """The textual half of the user turn (image parts are added by the caller).

    `image_roles` are the role labels of the images actually shown (post-ablation),
    in the order they are passed to the model, so "Image k" maps to a real image.
    """
    question = record["question"]
    options = record["options"]
    valid = " or ".join(sorted(options.keys()))

    if image_roles:
        img_desc = "\n".join(
            f"Image {i + 1}: {role}" for i, role in enumerate(image_roles)
        )
        images_block = f"Images provided (in order):\n{img_desc}\n\n"
    else:
        images_block = "No images are provided.\n\n"

    opt_block = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

    return (
        f"{images_block}"
        f"Question: {question}\n\n"
        f"Options:\n{opt_block}\n\n"
        f"Reply with ONLY one letter ({valid}) and nothing else."
    )
