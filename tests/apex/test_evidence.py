from apex_bot.evidence import review_context


def test_stale_or_removed_references_are_excluded_from_review():
    record = {
        "current": True,
        "last_success_at": 1000,
        "snapshot": {"content": {"text": "source"}, "revision": "1", "sha256": "hash"},
    }
    state = {"context": {"risk_state": "neutral"}, "references": {"a": record}}
    result = review_context(state, "BTCUSDT", 1001)
    assert result["reference_extracts"][0]["excerpt"] == "source"
    assert "reference_extracts" not in review_context(state, "BTCUSDT", 5000)
    record["current"] = False
    assert "reference_extracts" not in review_context(state, "BTCUSDT", 1001)
    assert state["context"] == {"risk_state": "neutral"}


def test_sheet_review_excludes_capital_and_unrelated_symbols():
    state = {
        "references": {
            "ledger": {
                "current": True,
                "last_success_at": 1000,
                "snapshot": {
                    "content": {
                        "ranges": [
                            {
                                "range": "Settings!A1:E180",
                                "values": [["private capital", 123456789]],
                            },
                            {
                                "range": "Counts!A1:K250",
                                "values": [
                                    ["Symbol", "Count"],
                                    ["BTCUSDT", "impulse"],
                                    ["ETHUSDT", "other"],
                                ],
                            },
                        ]
                    }
                },
            }
        }
    }
    result = review_context(state, "BTCUSDT", 1000)
    text = str(result)
    assert "private capital" not in text and "ETHUSDT" not in text
    assert "BTCUSDT" in text
