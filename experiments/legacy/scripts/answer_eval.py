#!/usr/bin/env python
"""Prepare or run paired retrieved/gold page answers using the same reader.

Preparation is offline. --execute explicitly authorizes reader calls. Mechanical
fact checks are screening aids: contradiction/context/citations require manual
review. Numeric facts are compared with units, never as loose substrings.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

from visual_bench import DEFAULT_EVAL, ROOT, digest, load_dataset, preflight, provenance, resolve_index


def normalized(text):
    return re.sub(r"\s+", " ", str(text).casefold().replace("³", "3").replace("²", "2")).strip()


def check_facts(answer, expected):
    text = normalized(answer)
    checks = []
    for fact in expected.get("facts", []):
        if fact["kind"] == "number":
            # Consume entire numbers, permitting Polish decimal commas/thousands spaces.
            unit = normalized(fact["unit"])
            aliases = [unit, *map(normalized, fact.get('unit_aliases', []))]
            if unit == 'pln':
                aliases.extend(['zł', 'złotych', 'zlotych'])
            unit_pattern = '(?:' + '|'.join(re.escape(v) for v in sorted(set(aliases), key=len, reverse=True)) + ')'
            matches = re.finditer(r"(?<![\w\d])[-+]?\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?\s*" + unit_pattern + r"(?!\w)", text)
            values = []
            for match in matches:
                number = re.match(r"[-+]?\d+(?: \d{3})*(?:[.,]\d+)?", match.group()).group()
                values.append(float(number.replace(" ", "").replace(",", ".")))
            passed = any(abs(v - float(fact["value"])) <= float(fact.get("tolerance", 0)) for v in values)
        elif fact["kind"] == "entity":
            passed = any(re.search(r"(?<!\w)" + re.escape(normalized(alias)) + r"(?!\w)", text)
                         for alias in [fact["value"], *fact.get("aliases", [])])
        elif fact["kind"] == "dimensions":
            numbers = [re.escape(str(v)) for v in fact["values"]]
            pattern = r"(?<!\d)" + r"\s*[x×]\s*".join(numbers) + r"\s*" + re.escape(normalized(fact["unit"])) + r"(?!\w)"
            passed = bool(re.search(pattern, text))
        else:
            raise ValueError(f"Unsupported fact kind: {fact['kind']}")
        checks.append({"fact": fact, "passed": passed})
    return {"checks": checks, "all_expected_facts_present": bool(checks) and all(c["passed"] for c in checks),
            "manual_review_required": True, "manual_review": None}


def screen_answer(answer, question):
    result = check_facts(answer, question['expected_answer'])
    text = normalized(answer)
    # Only screen within a clause: "Nie ma problemu, cena ..." is an answer.
    # This remains a review signal, not a factual-correctness judgment.
    refusal = bool(re.search(
        r'\bbrak\b[^.!?;,:]{0,45}(?:informacj|cen|danych|odpowied|pasując)'
        r'|\bnie (?:ma|podano|znalaz\w*|znalez\w*|zawier\w*)\b[^.!?;,:]{0,50}(?:informacj|cen|danych|odpowied|wartoś|kwot|pasując)'
        r'|\bnie mogę\b[^.!?;,:]{0,35}(?:ustalić|podać|odpowiedzieć|odczytać|potwierdzić)'
        r'|\bcannot (?:find|answer|determine)\b|\bnot (?:provided|available|found)\b'
        r'|\binsufficient (?:evidence|information)\b', text))
    result.update(expected_answerable=question['answerable'], abstention_signal=refusal,
                  abstention_screen_pass=not question['answerable'] and refusal,
                  possible_false_refusal=question['answerable'] and refusal and not result['all_expected_facts_present'])
    return result


def prepare(data, mapping, retrieval_report, mode="visual", limit=12):
    qs = {q["id"]: q for q in data["questions"]}
    # First warm repetition only, no selecting the best of repeated runs.
    rows = [r for r in retrieval_report["rows"] if r["mode"] == mode and r["phase"] == "warm" and r["round"] == 0]
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate first-round retrieval rows")
    selected = []
    # Fixed deterministic type interleaving yields a representative inspection subset.
    types = sorted({qs[r["question_id"]]["type"] for r in rows})
    buckets = {t: sorted([r for r in rows if qs[r["question_id"]]["type"] == t], key=lambda r: r["question_id"]) for t in types}
    while any(buckets.values()) and len(selected) < limit:
        for t in types:
            if buckets[t] and len(selected) < limit:
                r = buckets[t].pop(0)
                q = qs[r["question_id"]]
                gold = [[mapping[g.get("source_id", q["source_id"])], g["page"]] for g in q["gold"]]
                if len(gold) > 4:
                    raise ValueError(f"{q['id']}: complete gold evidence exceeds reader page budget")
                selected.append({"question": q, "retrieved_pages": r["ranking"][:4], "gold_pages": gold,
                                 "retrieval_error": r["error"], "answers": {}})
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--index", type=Path, default=ROOT / "index-px1")
    ap.add_argument("--retrieval", type=Path, required=True)
    ap.add_argument("--mode", choices=("visual", "lexical", "hybrid"), default="visual")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--allow-draft", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.limit < 1:
        ap.error("limit must be positive")
    os.environ["PIXELRAG_INDEX_DIR"] = str(args.index.resolve())
    args.index = resolve_index(args.index)
    data = load_dataset(args.eval, args.allow_draft)
    mapping = preflight(data, args.index)
    source = json.loads(args.retrieval.read_text(encoding="utf-8"))
    prov = provenance(args.index, args.eval)
    if source["provenance"]["dataset_sha256"] != digest(args.eval) or source["provenance"]["index_fingerprint"] != prov["index_fingerprint"]:
        ap.error("retrieval report belongs to another dataset/index")
    cases = prepare(data, mapping, source, args.mode, args.limit)
    report = {"provenance": prov, "review_status": data["review_status"], "cases": cases,
              "reader_pages": 4, "executed": args.execute,
              "manual_rubric": ["Verify exact numeric values and units in context", "Check entity and conditions", "Verify cited pages support all claims", "Check unsupported extra claims", "For negatives require explicit abstention"],
              "gold_negative_policy": "negative gold path supplies no pages; corpus-wide absence labels require human audit"}
    if args.execute:
        os.environ["PIXELRAG_INDEX_DIR"] = str(args.index.resolve())
        import rag
        import prompts
        report["reader_provenance"] = {"prompt_module_sha256": digest(Path(prompts.__file__)), "cache": False,
                                       "provider_model": "recorded from each actual response"}
        rng = random.Random(1729)
        for case in cases:
            order = ["retrieved", "gold"]
            rng.shuffle(order)
            case["arm_order"] = order
            for arm in order:
                try:
                    # The same production reader/prompt path; only evidence differs.
                    response = rag.run_agent(case["question"]["q"], mode="oneshot", retrieval=args.mode,
                                             n_pages=4, pages_override=[tuple(p) for p in case[arm + "_pages"]], cache=False)
                    answer = response["answer"]
                    case["answers"][arm] = {"text": answer, "response": response,
                                            "reader": {"provider": response.get("provider"), "model": response.get("model")},
                                            "fact_screen": screen_answer(answer, case["question"]), "error": None}
                except Exception as exc:
                    case["answers"][arm] = {"error": f"{type(exc).__name__}: {exc}", "fact_screen": {"all_expected_facts_present": False, "manual_review_required": True}}
        for arm in ("retrieved", "gold"):
            answered = [c["answers"][arm] for c in cases]
            costs = [a.get("response", {}).get("usage", {}).get("cost_usd") for a in answered]
            report.setdefault("summary", {})[arm] = {
                "fixed_denominator": len(cases), "errors": sum(bool(a.get("error")) for a in answered),
                "fact_screen_passes": sum(a["fact_screen"]["all_expected_facts_present"] for a in answered),
                "negative_trials": sum(not c['question']['answerable'] for c in cases),
                "abstention_screen_passes": sum(a['fact_screen'].get('abstention_screen_pass', False) for a in answered),
                "possible_false_refusals": sum(a['fact_screen'].get('possible_false_refusal', False) for a in answered),
                "known_cost_usd": sum(c for c in costs if c is not None),
                "unknown_cost_count": sum(c is None for c in costs), "human_review_pending": len(cases)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
