"""Check text/image gold pages; execution and optional Jev decisions are billed."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from PIL import Image

from gemini_index import Index
import index_manifest
import rag
import visual_query


def _image_dataset(indexed, dataset):
    digest = dataset["source_sha256"]
    if digest not in indexed:
        raise ValueError("Evaluation source hash does not match this index")
    doc = indexed[digest]
    questions = []
    for case in dataset["image_queries"]:
        page, rect = case["page"], case["rect"]
        if type(page) is not int or not 1 <= page <= doc["pages"]:
            raise ValueError("Image query page is outside its source")
        if (not isinstance(rect, list) or len(rect) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in rect)
                or not 0 <= rect[0] < rect[2] <= 1 or not 0 <= rect[1] < rect[3] <= 1):
            raise ValueError("Image query requires a normalized nonempty rectangle")
        questions.append({"id": case["id"], "q": case.get("q", ""),
                          "source_id": doc["source_id"], "answerable": True,
                          "gold": [{"page": case["gold_page"]}],
                          "image_crop": {"article_id": doc["article_id"], "page": page, "rect": rect}})
    return {"sources": [{"source_id": doc["source_id"], "sha256": digest}], "questions": questions}


def prepare(index, dataset):
    indexed = {d["sha256"]: d for d in index_manifest.load(index.snapshot_dir)["documents"]}
    if "image_queries" in dataset:
        dataset = _image_dataset(indexed, dataset)
    sources = {}
    for source in dataset["sources"]:
        if source["sha256"] not in indexed:
            raise ValueError("Evaluation source hash does not match this index")
        sources[source["source_id"]] = indexed[source["sha256"]]
    seen, cases = set(), []
    for question in dataset["questions"]:
        if question["id"] in seen:
            raise ValueError("Duplicate evaluation question ID")
        seen.add(question["id"])
        if not question["answerable"]:
            continue  # Retrieval ranks cannot establish that the reader abstains.
        gold = set()
        for evidence in question["gold"]:
            doc = sources[evidence.get("source_id", question["source_id"])]
            page = evidence["page"]
            if type(page) is not int or not 1 <= page <= doc["pages"]:
                raise ValueError("Gold page is outside its source")
            gold.add((doc["source_id"], page))
        if not gold:
            raise ValueError("Positive questions require a gold evidence page")
        if "image_crop" in question:
            _query_image(index, question)  # Validate the actual crop before any billed calls.
        cases.append((question, gold))
    if not cases:
        raise ValueError("No positive retrieval cases")
    return cases


def _query_image(index, question):
    crop = question.get("image_crop")
    if crop is None:
        return None
    with Image.open(index.page_path(crop["article_id"], crop["page"])) as source:
        left, top, right, bottom = crop["rect"]
        box = (round(left * source.width), round(top * source.height),
               round(right * source.width), round(bottom * source.height))
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Image query crop has no pixels")
        image = source.crop(box).convert("RGB")
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        raw = io.BytesIO()
        image.save(raw, format="PNG")
    return visual_query._normalize(raw.getvalue())


def _metrics(hits, gold):
    found = {(p["source_id"], p["page"]) for p in hits[:4]} & gold
    return {"hit_first": bool(hits and (hits[0]["source_id"], hits[0]["page"]) in gold),
            "hit_within_four": bool(found), "all_evidence_within_four": gold <= found,
            "evidence_recall": len(found) / len(gold)}


def _summary(rows):
    return {"positive_questions": len(rows),
            "evidence_first": sum(row["hit_first"] for row in rows),
            "evidence_within_four": sum(row["hit_within_four"] for row in rows),
            "all_evidence_within_four": sum(row["all_evidence_within_four"] for row in rows),
            "mean_evidence_recall": sum(row["evidence_recall"] for row in rows) / len(rows),
            "errors": sum(row["error"] is not None for row in rows)}


def run(index, dataset, *, jev_enabled=False, compare_jev=False):
    cases = prepare(index, dataset)
    rows, baseline_rows = [], []
    for question, gold in cases:
        row = {"id": question["id"], **_metrics([], gold), "pages": [], "error": None}
        baseline = dict(row)
        start = time.perf_counter()
        try:
            image = _query_image(index, question)
            retrieval = rag.retrieve(index, question["q"], image=image,
                                     jev_enabled=jev_enabled or compare_jev)
            row.update(_metrics(retrieval["pages"], gold), pages=retrieval["pages"],
                       jev=retrieval["jev"])
            # Both rankings share this exact embedding and shortlist, including on fallback.
            baseline.update(_metrics(retrieval["candidates"][:4], gold),
                            pages=retrieval["candidates"][:4])
        except Exception as exc:
            row["error"] = baseline["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        baseline_rows.append(baseline)
    report = {"review_status": dataset.get("review_status", "unreviewed"),
              "dataset_limit": dataset.get("limit"), **_summary(rows),
              "negative_policy": "Skipped: retrieval ranks do not measure reader abstention.",
              "reader_calls": 0, "rows": rows}
    if jev_enabled or compare_jev:
        decisions = [row.get("jev", {}) for row in rows]
        report["jev"] = {"applied": sum(d.get("status") == "applied" for d in decisions),
                         "fallbacks": sum(d.get("status") == "fallback" for d in decisions),
                         "calls": sum(d.get("calls", 0) for d in decisions),
                         "seconds": sum(d.get("seconds", 0) for d in decisions),
                         "input_tokens": sum((d.get("usage") or {}).get("input_tokens", 0) for d in decisions),
                         "unknown_usage_calls": sum(d.get("calls", 0) for d in decisions if d.get("usage") is None)}
    if compare_jev:
        report["baseline"] = {**_summary(baseline_rows), "rows": baseline_rows}
        report["comparison_policy"] = "Same embedding and eight candidates; fallbacks are reported, not Jev successes."
    return report


def main():
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(os.environ.get("PIXELRAG_INDEX_DIR", "index")))
    parser.add_argument("--eval", type=Path, default=Path("eval/echo_questions.json"))
    parser.add_argument("--execute", action="store_true", help="Make billed embedding/decision requests")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--jev", action="store_true", help="Evaluate experimental Jev image reranking")
    mode.add_argument("--compare-jev", action="store_true", help="Compare Jev to dense retrieval on the same candidates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        index = Index(args.index)
        raw = args.eval.read_bytes()
        dataset = json.loads(raw.decode("utf-8"))
        cases = prepare(index, dataset)
        report = (run(index, dataset, jev_enabled=args.jev, compare_jev=args.compare_jev) if args.execute else
                  {"executed": False, "positive_questions": len(cases),
                   "note": "Source hashes, gold pages and image crops checked. Add --execute for billed calls."})
        report.update(index=str(index.snapshot_dir), dataset_sha256=hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if report.get("errors") or report.get("jev", {}).get("fallbacks"):
        parser.exit(1)


if __name__ == "__main__":
    main()
