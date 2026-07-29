from app.services.document_context import (
    field_kind,
    field_rule_flag,
    should_run_context_model,
)


def test_selected_document_type_gives_id_number_context():
    assert field_kind("ID Number", "pan") == "pan"
    assert field_rule_flag("ABCDE1234F", "ID Number", "pan") is False
    assert field_rule_flag("ABC1234F", "ID Number", "pan") is True


def test_date_context_rejects_impossible_calendar_date():
    assert field_rule_flag("31/02/2020", "Date of Birth") is True
    assert field_rule_flag("29/02/2020", "Date of Birth") is False


def test_structured_values_skip_expensive_language_model():
    assert should_run_context_model(
        {"text": "123456", "field_label": "PIN code", "is_label_token": False},
        "generic",
    ) is False


def test_name_value_uses_context_model():
    assert should_run_context_model(
        {"text": "Anish", "field_label": "Name", "is_label_token": False},
        "generic",
    ) is True
