from supportlens_ml.modeling import (
    build_tfidf_logistic_regression,
    calculate_core_metrics,
)


def test_pipeline_can_fit_and_predict() -> None:
    texts = [
        "cancel my order",
        "please cancel the purchase",
        "where is my delivery",
        "track my package",
        "cancel this transaction",
        "delivery status please",
    ]
    labels = [
        "cancel_order",
        "cancel_order",
        "track_order",
        "track_order",
        "cancel_order",
        "track_order",
    ]

    pipeline = build_tfidf_logistic_regression(
        min_df=1,
        max_features=1_000,
        c_value=1.0,
    )
    pipeline.fit(texts, labels)

    predictions = pipeline.predict(
        ["please cancel order", "where is the package"]
    )

    assert len(predictions) == 2
    assert set(predictions).issubset({"cancel_order", "track_order"})


def test_core_metrics_return_expected_keys() -> None:
    metrics = calculate_core_metrics(
        ["a", "a", "b"],
        ["a", "b", "b"],
    )

    assert set(metrics) == {"accuracy", "macro_f1", "weighted_f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
