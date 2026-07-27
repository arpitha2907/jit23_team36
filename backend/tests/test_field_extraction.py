from app.services.field_extraction import annotate_field_labels


def make_word(text, line_num=1):
    return {
        "text": text,
        "block_num": 1,
        "par_num": 1,
        "line_num": line_num,
    }


def test_name_field_with_colon():
    words = [
        make_word("Name:"),
        make_word("Anish"),
        make_word("Raj"),
    ]

    result = annotate_field_labels(words)

    assert result[0]["is_label_token"] is True
    assert result[1]["field_label"] == "Name"
    assert result[2]["field_label"] == "Name"


def test_date_of_birth_field():
    words = [
        make_word("Date"),
        make_word("of"),
        make_word("Birth:"),
        make_word("15/08/1998"),
    ]

    result = annotate_field_labels(words)

    assert result[0]["is_label_token"] is True
    assert result[1]["is_label_token"] is True
    assert result[2]["is_label_token"] is True
    assert result[3]["field_label"] == "Date of Birth"


def test_id_number_without_colon():
    words = [
        make_word("ID"),
        make_word("Number"),
        make_word("123456789"),
    ]

    result = annotate_field_labels(words)

    assert result[0]["is_label_token"] is True
    assert result[1]["is_label_token"] is True
    assert result[2]["field_label"] == "ID Number"


def test_gender_field():
    words = [
        make_word("Gender:"),
        make_word("Male"),
    ]

    result = annotate_field_labels(words)

    assert result[0]["is_label_token"] is True
    assert result[1]["field_label"] == "Gender"


def test_unknown_field():
    words = [
        make_word("Random"),
        make_word("Information"),
        make_word("12345"),
    ]

    result = annotate_field_labels(words)

    for word in result:
        assert word["is_label_token"] is False
        assert word["field_label"] is None


def test_multiple_lines():
    words = [
        make_word("Name:", 1),
        make_word("Anish", 1),
        make_word("Gender:", 2),
        make_word("Male", 2),
    ]

    result = annotate_field_labels(words)

    assert result[1]["field_label"] == "Name"
    assert result[3]["field_label"] == "Gender"


def test_original_data_not_modified():
    words = [
        make_word("Name:"),
        make_word("Anish"),
    ]

    annotate_field_labels(words)

    assert "is_label_token" not in words[0]
    assert "field_label" not in words[0]