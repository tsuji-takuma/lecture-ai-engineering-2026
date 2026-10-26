"""第1回 宿題（発展）: 応答時間とトークン数をモデル × 質問で計測する。

使い方:
    uv run python scripts/measure_lec01.py --dry-run
    uv run python scripts/measure_lec01.py --runs 1 --models gemini-3.5-flash-lite
    uv run python scripts/measure_lec01.py
"""

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app を import するため

from google import genai  # noqa: E402
from google.genai import errors  # noqa: E402

from app.ask import token_counts  # noqa: E402
from app.config import DEFAULT_MODEL, get_api_key  # noqa: E402

QUESTIONS = {
    "Q1": "LoRAとは何ですか？",
    "Q2": "Transformerの仕組みを、自己注意機構を中心に、数式も交えて詳しく説明してください。",
    "Q3": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
}
FIELDS = [
    "timestamp", "model", "question_id", "run", "latency_s",
    "input_tokens", "output_tokens", "thoughts_tokens", "total_tokens",
    "answer_chars", "status",
]


def call(client, model: str, question: str) -> tuple[dict, str]:
    """1回呼び出して計測する。429/503 は1回だけリトライする。"""
    for attempt in range(2):
        start = time.perf_counter()
        try:
            response = client.models.generate_content(model=model, contents=question)
        except errors.APIError as e:
            status = str(e.code) if e.code in (429, 503) else f"error:{type(e).__name__}"
            if status in ("429", "503") and attempt == 0:
                time.sleep(30)
                continue
            return {"status": status}, ""
        except Exception as e:
            return {"status": f"error:{type(e).__name__}"}, ""
        latency = time.perf_counter() - start
        t = token_counts(response)
        text = response.text or ""
        return {
            "latency_s": round(latency, 3),
            "input_tokens": t["input"],
            "output_tokens": t["output"],
            "thoughts_tokens": t["thoughts"],
            "total_tokens": t["total"],
            "answer_chars": len(text),
            "status": "ok",
        }, text


def summarize(rows: list[dict]) -> str:
    def mean(xs):
        xs = [int(x) for x in xs if x not in (None, "")]
        return f"{statistics.mean(xs):.0f}" if xs else "-"

    lines = [
        "| model | question | n(ok) | latency mean | median | min | max | stdev "
        "| input | output | thoughts | total | chars |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    keys = sorted({(r["model"], r["question_id"]) for r in rows})
    for model, qid in keys:
        group = [r for r in rows if r["model"] == model and r["question_id"] == qid]
        ok = [r for r in group if r["status"] == "ok"]
        lat = [float(r["latency_s"]) for r in ok]
        stats = (
            [statistics.mean(lat), statistics.median(lat), min(lat), max(lat),
             statistics.stdev(lat) if len(lat) > 1 else 0.0]
            if lat else []
        )
        lat_cols = " | ".join(f"{x:.2f}" for x in stats) if stats else " | ".join(["-"] * 5)
        tok_cols = " | ".join(
            mean(r[k] for r in ok)
            for k in ("input_tokens", "output_tokens", "thoughts_tokens", "total_tokens", "answer_chars")
        )
        lines.append(f"| {model} | {qid} | {len(ok)}/{len(group)} | {lat_cols} | {tok_cols} |")

    failures = [r for r in rows if r["status"] != "ok"]
    lines.append("")
    lines.append(f"失敗: {len(failures)} 件 " + ", ".join(
        f"{r['model']}/{r['question_id']}/run{r['run']}={r['status']}" for r in failures
    ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, "gemini-3.8-flash"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=13.0, help="リクエスト間の待ち時間（秒）")
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default="lectures/01/results")
    args = parser.parse_args()

    plan = [(m, q, r) for m in args.models for q in QUESTIONS for r in range(1, args.runs + 1)]
    print(f"[plan] models={args.models} questions={list(QUESTIONS)} runs={args.runs}")
    print(f"[plan] 総リクエスト数={len(plan)} (上限 {args.max_requests}) sleep={args.sleep}s")
    if len(plan) > args.max_requests:
        raise SystemExit("総リクエスト数が --max-requests を超えています。")
    if args.dry_run:
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=get_api_key())
    rows = []
    with open(out / "measure.csv", "w", newline="") as fcsv, open(out / "answers.jsonl", "w") as fj:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        writer.writeheader()
        for i, (model, qid, run) in enumerate(plan):
            if i > 0:
                time.sleep(args.sleep)
            result, text = call(client, model, QUESTIONS[qid])
            row = {k: "" for k in FIELDS} | {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model, "question_id": qid, "run": run,
            } | {k: ("" if v is None else v) for k, v in result.items()}
            writer.writerow(row)
            fcsv.flush()
            fj.write(json.dumps({"model": model, "question_id": qid, "run": run, "answer": text},
                                ensure_ascii=False) + "\n")
            rows.append(row)
            print(f"[{i + 1}/{len(plan)}] {model} {qid} run{run}: {row['status']} "
                  f"{row['latency_s']}s out={row['output_tokens']} thoughts={row['thoughts_tokens']}")

    (out / "summary.md").write_text(summarize(rows))
    print(f"[done] {out}/summary.md")


if __name__ == "__main__":
    main()
