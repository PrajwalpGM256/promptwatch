from tools.redact import BODY_CAP, cap, clean, redact


def test_clean_strips_invisible_characters():
    assert clean("a͏͏  b­c  d") == "a bc d"


def test_clean_collapses_blank_line_runs():
    assert clean("one\n\n\n\n\ntwo") == "one\n\ntwo"


def test_long_tracking_urls_are_collapsed():
    url = "https://www.adzuna.com/search?loc=1&se=" + "X" * 90
    assert "example.com" in redact(f"see {url} end")
    assert "adzuna" not in redact(f"see {url} end")


def test_short_public_urls_survive():
    assert redact("see https://www.cspring.com ok") == "see https://www.cspring.com ok"


def test_email_addresses_are_masked():
    assert redact("mail me at someone@gmail.com") == "mail me at noreply@example.com"


def test_phone_numbers_are_masked():
    assert redact("call 872-213-1050 now") == "call 555-010-0000 now"


def test_requisition_ids_are_zeroed():
    assert "R000100000" in redact("role R000107071 at Acme")
    assert "2026-50000" in redact("posting 2026-52068 closed")


def test_extra_replacements_run_before_shared_patterns():
    out = redact("Nikky Potter wrote", extra=[(r"Nikky Potter", "Robin Alder")])
    assert out == "Robin Alder wrote"


def test_cap_leaves_short_text_alone():
    assert cap("short") == "short"


def test_cap_truncates_on_a_word_boundary_with_a_marker():
    text = "word " * 1000
    out = cap(text)
    assert out.endswith("[...truncated]")
    assert len(out) <= BODY_CAP + len("\n\n[...truncated]")
    assert "wor\n" not in out
