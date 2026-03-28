from dataset_eval import aggregate_metrics


def test_aggregate_metrics_computes_expected_means():
    results = [
        {
            "correctness": 1.0,
            "relevance": 1.0,
            "groundedness": 1.0,
            "hallucination": False,
            "explanation": "Fully correct",
        },
        {
            "correctness": 0.0,
            "relevance": 0.5,
            "groundedness": 0.0,
            "hallucination": True,
            "explanation": "Unsupported answer",
        },
    ]

    aggregated = aggregate_metrics(results)

    assert aggregated["avg_correctness"] == 0.5
    assert aggregated["avg_relevance"] == 0.75
    assert aggregated["avg_groundedness"] == 0.5
    assert aggregated["hallucination_rate"] == 0.5
    assert aggregated["sample_count"] == 2
