"""Demo guard logic and prompt format (no model, no gradio needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kitsune.data.policy import is_refusal
from kitsune.prompts import StoryRequest, build_user_prompt, render_prompt
from kitsune.versions import BASE_MODEL, BASE_REVISION

_spec = importlib.util.spec_from_file_location(
    "demo_app", Path(__file__).resolve().parents[1] / "demo" / "app.py"
)
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


def test_ok_request_builds_prompt() -> None:
    user, msg = app.screen_request(["異世界転生"], "追放された剣士", "あらすじ", "")
    assert msg is None and user.startswith("ジャンル: 異世界転生")


@pytest.mark.parametrize(
    ("title", "reason_word"),
    [
        ("エロい魔法少女", "性的"),
        ("ピカチュウと魔王城", "二次創作"),
        ("織田信長が異世界で無双", "実在"),
        ("連絡先 test@example.com", "全年齢"),
    ],
)
def test_disallowed_titles_are_refused(title: str, reason_word: str) -> None:
    user, msg = app.screen_request(["魔王と勇者"], title, "短編", "")
    assert user is None and is_refusal(msg) and reason_word in msg


def test_validation_messages() -> None:
    assert app.screen_request([], "題", "短編", "")[1]
    assert app.screen_request(["魔法少女", "魔法学園", "スローライフ", "ハイファンタジー"], "題", "短編", "")[
        1
    ]
    assert app.screen_request(["魔法少女"], "  ", "短編", "")[1]
    assert "入力エラー" in app.screen_request(["魔法少女"], "題", "続き", "")[1]


def test_english_requests() -> None:
    from kitsune.en import is_refusal_en

    user, msg = app.screen_request_en(["異世界転生"], "The Exiled Swordsman", "あらすじ", "")
    assert msg is None and user.startswith("Genres: Isekai\nTitle: The Exiled Swordsman\nFormat: synopsis")
    for title, word in (
        ("An Erotic Night in the Castle", "sexual"),
        ("Naruto Joins the Guild", "existing works"),
        ("Oda Nobunaga vs. the Demon Lord", "real people"),
        ("Write to test@example.com", "general audience"),
    ):
        user, msg = app.screen_request_en(["魔王と勇者"], title, "短編", "")
        assert user is None and is_refusal_en(msg) and word in msg, (title, msg)
    assert app.screen_request_en([], "T", "短編", "")[1]
    assert "passage" in app.screen_request_en(["魔法少女"], "T", "続き", "")[1]
    assert "hid" in app.screen_output_en("They undressed, naked.")
    assert app.screen_output_en("The knight drew her sword.") == "The knight drew her sword."


def test_output_screen() -> None:
    assert app.screen_output("勇者は剣を抜いた。") == "勇者は剣を抜いた。"
    assert "非表示" in app.screen_output("二人は全裸で")


@pytest.mark.network
def test_chat_prompt_matches_training_template() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    except OSError as e:
        pytest.skip(str(e))
    user = build_user_prompt(StoryRequest(["魔法少女"], "魔法少女ルミナは今日も遅刻する", "短編"))
    assert "<bos>" + app.chat_prompt(user) == render_prompt(tok, user)
