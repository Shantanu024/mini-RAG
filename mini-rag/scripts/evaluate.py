"""
RAG Quality Evaluation Script
Tests the pipeline with 12 pre-defined questions and scores quality.
"""

import json
import time
import sys
from pathlib import Path

# Add backend to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app import RAGPipeline


# ─────────────────────────────────────────────
# Test Questions (derived from documents)
# ─────────────────────────────────────────────
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What factors affect construction project delays?",
        "expected_topics": ["weather", "supply chain", "labor", "permits", "design"],
        "source": "construction_policies.txt",
    },
    {
        "id": 2,
        "question": "What insurance is required for a project worth $200,000?",
        "expected_topics": ["general liability", "workers compensation", "1,000,000"],
        "source": "platform_faq.txt",
    },
    {
        "id": 3,
        "question": "How are contractor bids evaluated?",
        "expected_topics": ["price", "rating", "experience", "timeline"],
        "source": "platform_faq.txt",
    },
    {
        "id": 4,
        "question": "What are the concrete strength requirements for commercial construction?",
        "expected_topics": ["PSI", "4,000", "compressive"],
        "source": "technical_specs.txt",
    },
    {
        "id": 5,
        "question": "What safety equipment is mandatory on construction sites?",
        "expected_topics": ["hard hat", "PPE", "safety vest", "boots"],
        "source": "construction_policies.txt",
    },
    {
        "id": 6,
        "question": "How does the platform handle payment disputes?",
        "expected_topics": ["escrow", "mediation", "7 days", "arbitration"],
        "source": "construction_policies.txt",
    },
    {
        "id": 7,
        "question": "What are the requirements to become a contractor on the platform?",
        "expected_topics": ["license", "insurance", "references", "background check"],
        "source": "platform_faq.txt",
    },
    {
        "id": 8,
        "question": "What are the HVAC ventilation requirements?",
        "expected_topics": ["ASHRAE", "CFM", "fresh air"],
        "source": "technical_specs.txt",
    },
    {
        "id": 9,
        "question": "How much can a contractor subcontract?",
        "expected_topics": ["40%", "subcontractor", "verified"],
        "source": "platform_faq.txt",
    },
    {
        "id": 10,
        "question": "What roofing membrane thickness is required for flat roofs?",
        "expected_topics": ["60-mil", "TPO", "EPDM"],
        "source": "technical_specs.txt",
    },
    {
        "id": 11,
        "question": "What are the delay penalty clauses?",
        "expected_topics": ["liquidated damages", "$500", "$2,000", "force majeure"],
        "source": "construction_policies.txt",
    },
    {
        "id": 12,
        "question": "What environmental regulations must contractors follow?",
        "expected_topics": ["Clean Water Act", "EPA", "stormwater", "LEED"],
        "source": "platform_faq.txt",
    },
]


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────
def evaluate_retrieval(retrieved_chunks, expected_source, expected_topics):
    """Score retrieval quality."""
    scores = {}

    # Source match: did we retrieve from the right document?
    sources = [c["source"] for c in retrieved_chunks]
    scores["source_match"] = 1 if expected_source in sources else 0

    # Topic coverage: how many expected topics appear in retrieved text?
    all_text = " ".join(c["text"].lower() for c in retrieved_chunks)
    matched_topics = [t for t in expected_topics if t.lower() in all_text]
    scores["topic_coverage"] = len(matched_topics) / len(expected_topics) if expected_topics else 0
    scores["matched_topics"] = matched_topics
    scores["missing_topics"] = [t for t in expected_topics if t.lower() not in all_text]

    # Average similarity score of top-3
    top_scores = [c["similarity_score"] for c in retrieved_chunks[:3]]
    scores["avg_similarity"] = sum(top_scores) / len(top_scores) if top_scores else 0

    return scores


def evaluate_answer(answer, expected_topics):
    """Score answer quality heuristically."""
    scores = {}
    answer_lower = answer.lower()

    # Hallucination check: does answer contain disclaimer when unsure?
    refusal_phrases = ["cannot", "not available", "not mentioned", "not in the", "no information"]
    scores["has_refusal_capability"] = any(p in answer_lower for p in refusal_phrases)

    # Topic coverage in answer
    matched = [t for t in expected_topics if t.lower() in answer_lower]
    scores["answer_topic_coverage"] = len(matched) / len(expected_topics) if expected_topics else 0

    # Length check (reasonable answer length)
    scores["answer_length"] = len(answer.split())
    scores["reasonable_length"] = 20 <= scores["answer_length"] <= 400

    return scores


def run_evaluation():
    """Run full evaluation and print report."""
    print("=" * 70)
    print("  MINI RAG - QUALITY EVALUATION REPORT")
    print("=" * 70)

    pipeline = RAGPipeline()
    pipeline.initialize()

    results = []
    total_retrieval_score = 0
    total_answer_score = 0

    for test in TEST_QUESTIONS:
        print(f"\n[Q{test['id']}] {test['question']}")
        print("-" * 60)

        start = time.time()
        result = pipeline.query(test["question"], top_k=5)
        elapsed = time.time() - start

        ret_scores = evaluate_retrieval(
            result["retrieved_chunks"],
            test["expected_source"],
            test["expected_topics"],
        )
        ans_scores = evaluate_answer(result["answer"], test["expected_topics"])

        # Composite score
        retrieval_composite = (ret_scores["source_match"] * 0.4 + ret_scores["topic_coverage"] * 0.6)
        answer_composite = ans_scores["answer_topic_coverage"]

        total_retrieval_score += retrieval_composite
        total_answer_score += answer_composite

        print(f"  Retrieval:")
        print(f"    Source match:    {'✅' if ret_scores['source_match'] else '❌'}")
        print(f"    Topic coverage:  {ret_scores['topic_coverage']:.0%} ({', '.join(ret_scores['matched_topics']) or 'none'})")
        print(f"    Avg similarity:  {ret_scores['avg_similarity']:.3f}")
        if ret_scores["missing_topics"]:
            print(f"    Missing topics:  {', '.join(ret_scores['missing_topics'])}")

        print(f"  Answer:")
        print(f"    Topic coverage:  {ans_scores['answer_topic_coverage']:.0%}")
        print(f"    Word count:      {ans_scores['answer_length']}")
        print(f"    Length OK:       {'✅' if ans_scores['reasonable_length'] else '⚠️'}")
        print(f"  Latency:         {elapsed:.2f}s")
        print(f"  Answer preview:  {result['answer'][:200]}...")

        results.append({
            "question_id": test["id"],
            "question": test["question"],
            "retrieval_score": retrieval_composite,
            "answer_score": answer_composite,
            "latency": elapsed,
            "chunks_retrieved": len(result["retrieved_chunks"]),
        })

    # Summary
    n = len(TEST_QUESTIONS)
    avg_retrieval = total_retrieval_score / n
    avg_answer = total_answer_score / n
    avg_latency = sum(r["latency"] for r in results) / n

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Questions tested:         {n}")
    print(f"  Avg retrieval score:      {avg_retrieval:.1%}")
    print(f"  Avg answer quality score: {avg_answer:.1%}")
    print(f"  Avg latency:              {avg_latency:.2f}s")
    print(f"  Overall score:            {(avg_retrieval + avg_answer) / 2:.1%}")
    print("=" * 70)

    # Save JSON report
    report = {
        "summary": {
            "total_questions": n,
            "avg_retrieval_score": round(avg_retrieval, 3),
            "avg_answer_score": round(avg_answer, 3),
            "avg_latency_seconds": round(avg_latency, 3),
            "overall_score": round((avg_retrieval + avg_answer) / 2, 3),
        },
        "results": results,
    }
    report_path = Path(__file__).parent / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
