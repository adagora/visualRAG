"""Ask the Gemini page-image index from the terminal."""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from gemini_index import Index
import rag
import visual_query


def main():
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="")
    parser.add_argument("--index", type=Path, default=Path(os.environ.get("PIXELRAG_INDEX_DIR", "index")))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--retrieve-only", action="store_true", help="Search without an answer-reader call; the query embedding is billed")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--jev", action=argparse.BooleanOptionalAction, default=None,
                        help="Use experimental Jev image decisions to select evidence pages")
    args = parser.parse_args()
    question = args.question.strip()
    if not question and args.image is None:
        parser.error("enter a question or pass --image")
    question = question or "Znajdź element pokazany na obrazie i opisz go na podstawie źródeł."
    try:
        index = Index(args.index)
        image = visual_query.read_image(args.image) if args.image else None
        result = (rag.retrieve(index, question, image=image, jev_enabled=args.jev) if args.retrieve_only else
                  rag.answer(index, question, image=image, jev_enabled=args.jev))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    if args.json or args.retrieve_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
        decision = result["retrieval"]["jev"]
        if decision["status"] == "fallback":
            import sys
            print("Jev fallback: " + decision["reason"], file=sys.stderr)
        for citation in result["citations"]:
            print(f"  {citation['document']} · s. {citation['page']}")


if __name__ == "__main__":
    main()
