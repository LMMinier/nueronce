"""Framework-agnostic dialogue SFT data: turn encoding, batching, and a small
hand-written (prompt, response) dataset.

Deliberately has **no torch import** so both training backends can share the
exact same data and turn layout without either one requiring the other's
autograd engine to be installed:

- ``nueronce.training.sft`` — the production path, fine-tunes the real
  ``NUERONCEModel`` (PyTorch-backed).
- ``nueronce.engine`` — a from-scratch (NumPy-only) autograd engine; its
  ``MicroByteLM`` can run the same masked-SFT loss with zero external
  dependencies beyond NumPy, which is what ``tests/test_sft_engine.py``
  exercises.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..prompting import ASSISTANT, END, USER, format_training_example

# Must match nueronce.chat.Conversation.user_tag / .bot_tag exactly (checked in
# tests/test_sft.py) so a checkpoint tuned on this data drops straight into
# Conversation at inference time.
USER_TAG: str = USER
BOT_TAG: str = ASSISTANT

# Identifies the prompt layout these tags produce, stamped into checkpoint meta
# by nueronce.training.sharded_sft so inference can pick the matching format.
# "canonical" = the <|user|>/<|assistant|> markers from nueronce.prompting;
# checkpoints trained before this constant existed carry no prompt_format and
# are resolved to "legacy" (plain "User: "/"Assistant: ") by the chat loaders.
PROMPT_FORMAT: str = "canonical" if USER_TAG.startswith("<|") else "legacy"

SFTExample = Tuple[str, str]

# A small, hand-written prompt -> response dialogue set. Not a scraped corpus:
# just enough turn-taking signal (greetings, identity, small factual Q&A,
# courtesy, closings) to teach "respond to the user" instead of "continue the
# text". Real dialogue-quality tuning needs a much larger, curated dataset;
# this is sized to prove the training pass works end-to-end.
SFT_DATASET: List[SFTExample] = [
    ("Hello", "Hello! How can I help you today?"),
    ("Hi there", "Hi! What can I do for you?"),
    ("Hey", "Hey there! How can I help?"),
    ("Good morning", "Good morning! How can I help you today?"),
    ("Good evening", "Good evening! What can I do for you?"),
    ("Who are you?", "I am an assistant built on the NUERONCE byte model."),
    ("What are you?", "I am a small byte-level language model trained to hold a conversation."),
    ("What is your name?", "I do not have a personal name, but you can call me Assistant."),
    ("What can you do?", "I can answer questions and chat about a range of topics."),
    ("How are you?", "I am doing well, thank you for asking. How are you?"),
    ("Thank you", "You are welcome!"),
    ("Thanks", "You are welcome! Let me know if you need anything else."),
    ("Goodbye", "Goodbye! Have a great day."),
    ("Bye", "Bye! Take care."),
    ("What is two plus two?", "Two plus two equals four."),
    ("What is three plus five?", "Three plus five equals eight."),
    ("What is ten minus four?", "Ten minus four equals six."),
    ("What is one plus one?", "One plus one equals two."),
    ("What is five times five?", "Five times five equals twenty five."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What color is the sky?", "The sky is blue on a clear day."),
    ("What color is grass?", "Grass is usually green."),
    ("Tell me a fact.", "Water freezes at zero degrees Celsius."),
    ("Tell me something interesting.", "Honey never spoils if stored properly."),
    ("Can you help me?", "Yes, I would be glad to help. What do you need?"),
    ("Can you write a poem?", "I can try. Roses are red, violets are blue."),
    ("What day is it?", "I do not have access to the current date."),
    ("What time is it?", "I do not have access to the current time."),
    ("Are you a robot?", "I am a software program, not a physical robot."),
    ("Are you human?", "No, I am an artificial language model."),
    ("Do you have feelings?", "I do not have feelings the way humans do."),
    ("What is your favorite color?", "I do not have preferences, but blue is a popular choice."),
    ("Is the earth round?", "Yes, the earth is roughly spherical."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("How many months are in a year?", "There are twelve months in a year."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("What is the opposite of up?", "The opposite of up is down."),
    ("Spell the word cat.", "The word cat is spelled c a t."),
    ("How do you spell dog?", "The word dog is spelled d o g."),
    ("Count to five.", "One, two, three, four, five."),
    ("Give me a synonym for happy.", "A synonym for happy is glad."),
    ("Give me a synonym for big.", "A synonym for big is large."),
    ("What language are you speaking?", "I am speaking English."),
    ("Do you understand English?", "Yes, I was trained on English text."),
    ("What is a computer?", "A computer is a machine that processes information."),
    ("What is a dog?", "A dog is a common domesticated animal."),
    ("What is a cat?", "A cat is a small domesticated animal."),
    ("Is water wet?", "Water is commonly described as wet."),
    ("Sorry to bother you.", "No bother at all, I am here to help."),
    ("Nice to meet you.", "Nice to meet you too!"),
    ("See you later.", "See you later! Take care."),
    ("What should I do today?", "That depends on your goals for the day."),
    ("I am bored.", "You could try reading a book or going for a walk."),
    ("I am happy today.", "That is wonderful to hear!"),
    ("I am sad today.", "I am sorry to hear that. I hope things get better."),
    ("Can you tell a joke?", "Why did the computer go to the doctor? It had a virus."),
    ("What is the weather like?", "I do not have access to live weather data."),
    ("Repeat after me: hello world.", "Hello world."),
    ("What is the first letter of the alphabet?", "The first letter of the alphabet is A."),
    ("What is the last letter of the alphabet?", "The last letter of the alphabet is Z."),
]


def encode_example(prompt: str, response: str, *, system: str = "",
                    user_tag: str = USER_TAG, bot_tag: str = BOT_TAG) -> Tuple[bytes, List[bool]]:
    """Byte-encode one turn in ``Conversation``'s exact layout and return
    ``(bytes, mask)`` where ``mask[i]`` is True iff byte ``i`` belongs to the
    response (the SFT loss target), including the trailing stop newline."""
    if user_tag == USER_TAG and bot_tag == BOT_TAG:
        return format_training_example(
            system_message=system, user_request=prompt, assistant_response=response
        )
    prefix_parts = []
    if system:
        prefix_parts.append(system.strip())
    prefix_parts.append(f"{user_tag}\n{prompt}")
    prefix = "\n".join(prefix_parts) + "\n" + bot_tag + "\n"
    full = prefix + response + "\n" + END + "\n"
    pb, fb = prefix.encode("utf-8"), full.encode("utf-8")
    return fb, [False] * len(pb) + [True] * (len(fb) - len(pb))


def make_sft_batch(examples: Sequence[SFTExample], *, system: str = "",
                    user_tag: str = USER_TAG, bot_tag: str = BOT_TAG,
                    max_len: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Byte-pad a list of (prompt, response) pairs into a training batch:
    ``byte_ids`` [B,T] int64 and ``target_mask`` [B,T] bool (True at response
    bytes). Plain NumPy so either autograd backend can consume it directly."""
    encoded = [encode_example(p, r, system=system, user_tag=user_tag, bot_tag=bot_tag)
               for p, r in examples]
    return _pad_batch(encoded, max_len)


Message = Dict[str, str]


def encode_messages(messages: Sequence[Message], *, system: str = "",
                     user_tag: str = USER_TAG, bot_tag: str = BOT_TAG) -> Tuple[bytes, List[bool]]:
    """Byte-encode an arbitrary-length conversation (``[{"role": "user"/
    "assistant", "content": ...}, ...]``, ending on an assistant turn) in the
    same layout as :func:`encode_example`, generalized to N turns. The mask is
    True on every assistant turn's content bytes + its trailing stop newline,
    False on user turns, role tags, and (if present) the system preamble —
    exactly the "ignore the prompt, train on the response(s)" contract, now
    applied per-turn for multi-turn conversations."""
    text = f"<|system|>\n{system.strip() if system else ''}\n"
    mask: List[bool] = [False] * len(text.encode("utf-8"))
    for msg in messages:
        tag = user_tag if msg["role"] == "user" else bot_tag
        if msg["role"] == "user":
            chunk = f"{tag}\n{msg['content']}\n"
            text += chunk
            mask += [False] * len(chunk.encode("utf-8"))
        else:
            prefix = f"{tag}\n"
            body = f"{msg['content']}\n{END}\n"
            text += prefix + body
            mask += [False] * len(prefix.encode("utf-8")) + [True] * len(body.encode("utf-8"))
    full_bytes = text.encode("utf-8")
    assert len(full_bytes) == len(mask)
    return full_bytes, mask


def make_conversation_batch(conversations: Sequence[Sequence[Message]], *, system: str = "",
                            user_tag: str = USER_TAG, bot_tag: str = BOT_TAG,
                            max_len: Optional[int] = None) -> Dict[str, np.ndarray]:
    """``make_sft_batch`` for arbitrary-length (possibly multi-turn) message
    lists instead of flat (prompt, response) pairs."""
    encoded = [encode_messages(m, system=system, user_tag=user_tag, bot_tag=bot_tag) for m in conversations]
    return _pad_batch(encoded, max_len)


def _pad_batch(encoded: Sequence[Tuple[bytes, List[bool]]], max_len: Optional[int]) -> Dict[str, np.ndarray]:
    t = max(len(b) for b, _ in encoded)
    if max_len:
        t = min(t, max_len)
    byte_ids = np.zeros((len(encoded), t), dtype=np.int64)
    target_mask = np.zeros((len(encoded), t), dtype=bool)
    for i, (b, m) in enumerate(encoded):
        b, m = b[:t], m[:t]
        byte_ids[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
        target_mask[i, : len(m)] = m
    return {"byte_ids": byte_ids, "target_mask": target_mask}


def held_out_split(examples: Sequence[SFTExample], val_frac: float = 0.2,
                    seed: int = 0) -> Tuple[List[SFTExample], List[SFTExample]]:
    """Hold out a random slice of the dialogue set for validation, same
    document-level-holdout spirit as ``nueronce.corpus.dataset.ByteCorpus``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(examples))
    n_val = max(1, int(round(len(examples) * val_frac)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    train = [examples[i] for i in train_idx]
    val = [examples[i] for i in val_idx]
    return train, val


__all__ = [
    "SFTExample", "SFT_DATASET", "USER_TAG", "BOT_TAG",
    "encode_example", "make_sft_batch", "held_out_split", "encode_messages",
    "make_conversation_batch",
]
