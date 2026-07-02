#!/usr/bin/env python3
"""Official CFNA inference entry point.

Model-only mode runs the byte model directly with the canonical prompt.
Retrieval mode runs the cognitive pipeline: retrieval -> reasoning -> plan ->
evidence-conditioned generation -> verification -> at most one revision.
Tool-assisted mode may answer narrow deterministic questions, but reports that
source separately from model output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from cfna.chat import load_checkpoint
from cfna.coherent_inference import deterministic_answer, surface_failure_reason
from cfna.prompting import STOP_SEQUENCES, extract_assistant_continuation, format_inference_prompt


def _load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p}")
    return load_checkpoint(str(p))[0]


def _model_only(model, prompt: str, args) -> dict:
    rendered = format_inference_prompt(
        system_message="You are CFNA. Answer briefly and do not invent unsupported facts.",
        user_request=prompt,
        trusted_evidence="",
        response_plan="Answer the user request directly and cautiously.",
    )
    out = model.generate(
        rendered.encode("utf-8"),
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_sequences=STOP_SEQUENCES,
        greedy=(args.temperature <= 0),
        continuation_only=True,
        return_scores=True,
    )
    answer = extract_assistant_continuation(out["bytes"])
    failure_reason = surface_failure_reason(answer, prompt=prompt)
    return {
        "answer": answer,
        "final_source": "model",
        "valid_generation": failure_reason is None,
        "failure_reason": failure_reason,
        "terminated": bool(out.get("stopped")) if answer else False,
        "average_log_probability": out["avg_logprob"],
        "average_entropy": out["avg_entropy"],
        "trace": {
            "selected_evidence": [],
            "authority_decisions": [],
            "plan": "direct model-only response",
            "first_draft": answer,
            "verification": None,
            "revision": None,
        },
    }


def _pipeline(model, prompt: str, evidence: List[str], args) -> dict:
    from cfna import data, pipeline

    corpus = evidence or [s.strip() for s in data.CORPUS.split(".") if s.strip()]
    answer, report, trace = pipeline.respond(
        model,
        prompt,
        corpus,
        mode="DELIBERATE",
        max_rounds=2,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    failure_reason = surface_failure_reason(answer, prompt=prompt)
    return {
        "answer": answer,
        "final_source": "model",
        "valid_generation": failure_reason is None,
        "failure_reason": failure_reason,
        "verification_passed": report.passes,
        "trace": {
            "selected_evidence": trace.get("selected_evidence", []),
            "authority_decisions": trace.get("provenance", {}),
            "plan": trace.get("plan", {}),
            "first_draft": trace.get("first_draft"),
            "verification": trace.get("verification"),
            "revision": trace.get("revision"),
        },
    }


def _print(result: dict, *, json_output: bool, show_trace: bool):
    if json_output:
        print(json.dumps(result if show_trace else {"answer": result["answer"],
                                                    "final_source": result["final_source"]}, indent=2))
        return
    print(result["answer"])
    if show_trace:
        print("\n=== trace ===")
        print(json.dumps(result.get("trace", {}), indent=2, default=str))
        print(f"final source: {result.get('final_source')}")


def _run_one(model, prompt: str, args, evidence: List[str]) -> dict:
    if args.assist_tools:
        tool = deterministic_answer(prompt)
        if tool is not None:
            return {
                "answer": tool,
                "final_source": "tool",
                "valid_generation": True,
                "failure_reason": None,
                "trace": {
                    "selected_evidence": [],
                    "authority_decisions": [],
                    "plan": "deterministic tool answered before model generation",
                    "first_draft": None,
                    "verification": None,
                    "revision": None,
                },
            }
    if args.use_retrieval:
        return _pipeline(model, prompt, evidence, args)
    return _model_only(model, prompt, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cfna_chat.pt")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--assist-tools", action="store_true")
    ap.add_argument("--use-retrieval", action="store_true")
    ap.add_argument("--evidence", action="append", default=[], help="trusted evidence snippet; repeatable")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--json-output", action="store_true")
    ap.add_argument("--show-trace", action="store_true")
    args = ap.parse_args()

    if args.model_only:
        args.assist_tools = False
        args.use_retrieval = False

    model = _load_model(args.checkpoint)
    evidence = list(args.evidence)

    if args.interactive:
        while True:
            try:
                prompt = input("> ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit"}:
                break
            _print(_run_one(model, prompt, args, evidence),
                   json_output=args.json_output, show_trace=args.show_trace)
        return

    prompt = args.prompt.strip()
    if not prompt:
        raise SystemExit("provide --prompt or use --interactive")
    _print(_run_one(model, prompt, args, evidence),
           json_output=args.json_output, show_trace=args.show_trace)


if __name__ == "__main__":
    main()
