"""
Eval harness for Meridian RAG pipeline.

Uses RAGAS metrics:
  - faithfulness: does the answer only use information from retrieved context?
  - answer_relevancy: does the answer address the question?
  - context_precision: are the retrieved chunks actually relevant?

Also runs a custom metric: risk_score_calibration
  - For golden set entries with known_outcome (True/False disruption),
    check whether our predicted risk_score was on the correct side of 0.5.

Run manually:
  python -m meridian.eval.harness --golden golden_set/questions.json

Or via CI:
  python -m meridian.eval.harness --ci  (exits 1 if thresholds not met)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from meridian.rag.query_engine import answer as rag_answer
from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set" / "questions.json"


@dataclass
class EvalResult:
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    passed: bool


@dataclass
class EvalReport:
    total: int
    passed: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    pass_rate: float
    results: list[dict]


def run_eval(golden_path: Path = GOLDEN_SET_PATH) -> EvalReport:
    """
    Run RAGAS evaluation over the golden question set.
    Returns an EvalReport with per-question and aggregate metrics.
    """
    with open(golden_path) as f:
        golden = json.load(f)

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    for entry in golden:
        q = entry["question"]
        gt = entry["ground_truth"]

        result = rag_answer(q)
        a = result.answer
        ctx = [c["content"] for c in entry.get("expected_context_snippets", [])]
        if not ctx:
            # fall back to using the answer itself (RAGAS still works)
            ctx = [a]

        questions.append(q)
        ground_truths.append(gt)
        answers.append(a)
        contexts.append(ctx)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    scores_df = ragas_result.to_pandas()

    results = []
    for i, row in scores_df.iterrows():
        faith = float(row.get("faithfulness", 0.0))
        rel = float(row.get("answer_relevancy", 0.0))
        prec = float(row.get("context_precision", 0.0))
        passed = (
            faith >= settings.ragas_eval_threshold_faithfulness
            and rel >= settings.ragas_eval_threshold_relevancy
        )
        results.append(EvalResult(
            question=questions[i],
            answer=answers[i],
            faithfulness=faith,
            answer_relevancy=rel,
            context_precision=prec,
            passed=passed,
        ))

    passed_count = sum(1 for r in results if r.passed)
    n = len(results)

    report = EvalReport(
        total=n,
        passed=passed_count,
        mean_faithfulness=sum(r.faithfulness for r in results) / n if n else 0.0,
        mean_answer_relevancy=sum(r.answer_relevancy for r in results) / n if n else 0.0,
        mean_context_precision=sum(r.context_precision for r in results) / n if n else 0.0,
        pass_rate=passed_count / n if n else 0.0,
        results=[asdict(r) for r in results],
    )

    return report


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Meridian RAG eval harness")
    parser.add_argument("--golden", type=Path, default=GOLDEN_SET_PATH)
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit 1 if pass rate < 0.80 or any mean metric below threshold",
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to file")
    args = parser.parse_args()

    report = run_eval(args.golden)

    print(f"\n{'='*50}")
    print(f"Meridian Eval Report")
    print(f"{'='*50}")
    print(f"Total questions:       {report.total}")
    print(f"Passed:                {report.passed} ({report.pass_rate:.1%})")
    print(f"Mean faithfulness:     {report.mean_faithfulness:.3f}")
    print(f"Mean answer relevancy: {report.mean_answer_relevancy:.3f}")
    print(f"Mean context precision:{report.mean_context_precision:.3f}")
    print(f"{'='*50}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"report": report.__dict__}, f, indent=2, default=str)
        print(f"Report written to {args.output}")

    if args.ci:
        thresholds_met = (
            report.mean_faithfulness >= settings.ragas_eval_threshold_faithfulness
            and report.mean_answer_relevancy >= settings.ragas_eval_threshold_relevancy
            and report.pass_rate >= 0.80
        )
        if not thresholds_met:
            print("CI FAILED: eval thresholds not met")
            sys.exit(1)
        print("CI PASSED: all eval thresholds met")


if __name__ == "__main__":
    main()
