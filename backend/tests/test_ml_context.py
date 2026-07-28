from unittest.mock import patch

import pytest

from app.services import ml_context


def test_empty_batch():
    assert ml_context.batch_token_surprisal([]) == []


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        ml_context.batch_token_surprisal(
            ["hello", "world"],
            batch_size=0,
        )


def test_invalid_negative_batch_size():
    with pytest.raises(ValueError):
        ml_context.batch_token_surprisal(
            ["hello", "world"],
            batch_size=-1,
        )


def test_single_token_invalid_index():
    assert ml_context.token_surprisal(
        ["hello"],
        -1,
    ) == 0.0


def test_single_token_index_too_large():
    assert ml_context.token_surprisal(
        ["hello"],
        5,
    ) == 0.0


def test_pipeline_uses_batch_inference():
    from app.services import pipeline

    fake_image = type(
        "FakeImage",
        (),
        {"shape": (100, 200, 3)},
    )()

    fake_words = [
        {
            "text": "Name:",
            "conf": 95.0,
            "left": 0,
            "top": 0,
            "width": 20,
            "height": 10,
            "block_num": 1,
            "par_num": 1,
            "line_num": 1,
        },
        {
            "text": "Anish",
            "conf": 95.0,
            "left": 25,
            "top": 0,
            "width": 30,
            "height": 10,
            "block_num": 1,
            "par_num": 1,
            "line_num": 1,
        },
    ]

    fake_ocr_result = {
        "text": "Name: Anish",
        "words": fake_words,
    }

    with patch.object(
        pipeline.preprocessing,
        "preprocess_image",
        return_value=fake_image,
    ), patch.object(
        pipeline.ocr_service,
        "run_ocr",
        return_value=fake_ocr_result,
    ), patch.object(
        pipeline.ml_context,
        "batch_token_surprisal",
        return_value=[0.1, 0.2],
    ) as batch_mock, patch.object(
        pipeline.correction,
        "suggest_corrections",
        return_value=[],
    ):

        result = pipeline.process_document("fake.jpg")

    batch_mock.assert_called_once_with(
        ["Name:", "Anish"]
    )

    assert len(result["fields"]) == 2

    assert result["fields"][0]["ml_surprisal"] == 0.1

    assert result["fields"][1]["ml_surprisal"] == 0.2