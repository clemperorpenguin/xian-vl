# Xian-VL — Core Vision-Language orchestration engine.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)

"""Per-line translation through the pinned Hy-MT2 model."""

import asyncio
from types import SimpleNamespace

import pytest

from xian.translate import (
    BATCH_SEPARATOR,
    MAX_TERMS_PER_REQUEST,
    NPU_TRANSLATION_MODEL,
    TRANSLATION_MODEL,
    LineTranslator,
    TranslationModelUnavailable,
    build_translation_prompt,
    relevant_terms,
    translation_pull_body,
)


class FakeCompletions:
    """Records every request and answers from a scripted table."""

    def __init__(self, answers=None, fail=False):
        self.answers = answers or {}
        self.fail = fail
        self.requests = []
        self.concurrent = 0
        self.peak_concurrent = 0

    async def create(self, *, model, messages, **kwargs):
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            self.requests.append({"model": model, "messages": messages})
            await asyncio.sleep(0)
            if self.fail:
                raise RuntimeError("backend refused")
            prompt = messages[-1]["content"]
            # Every Hy-MT2 template ends with the instruction, a blank line,
            # then the source text.
            body = prompt.rsplit("\n\n", 1)[-1]
            answer = self.answers.get(body, f"[{body}]")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
        finally:
            self.concurrent -= 1


def fake_processor(answers=None, *, installed=(TRANSLATION_MODEL,), fail=False, routed="Some-Chat-Model"):
    completions = FakeCompletions(answers, fail=fail)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    processor = SimpleNamespace(
        client=client,
        router=SimpleNamespace(downloaded_model_ids=list(installed)),
        get_translation_model_name=lambda: routed,
    )
    return processor, completions


# ── the prompt ───────────────────────────────────────────────────────

def test_the_prompt_is_one_bare_user_turn():
    """Hy-MT2 translates instructions rather than following them.

    Anything resembling a system prompt comes back translated into the target
    language and painted onto the screen.
    """
    processor, completions = fake_processor()
    translator = LineTranslator(processor=processor)

    asyncio.run(translator.translate_lines(["门"], "Chinese", "English"))

    messages = completions.requests[0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_the_prompt_is_hy_mt2_s_published_default_format():
    """Not an improvised instruction: the model has a documented one."""
    prompt = build_translation_prompt("门", "English")

    assert prompt == (
        "Translate the following text into English. Note that you should only output "
        "the translated result without any additional explanation:\n\n门"
    )


def test_the_prompt_does_not_name_the_source_language():
    """The published default format carries only the target."""
    assert "Chinese" not in build_translation_prompt("门", "English")


def test_a_style_uses_the_style_template():
    prompt = build_translation_prompt("门", "English", style="formal")

    assert "must strictly conform to [formal]" in prompt
    assert prompt.endswith("\n\n门")


def test_a_batch_uses_the_delimiter_template():
    """Losing one separator misaligns every line after it, so the model is
    told about them explicitly."""
    prompt = build_translation_prompt("a\nb", "English", delimited=True)

    assert "retain the exact same number of delimiters" in prompt


# ── the fan-out ──────────────────────────────────────────────────────

def test_every_line_gets_its_own_request():
    processor, completions = fake_processor({"一": "one", "二": "two", "三": "three"})
    translator = LineTranslator(processor=processor)

    result = asyncio.run(translator.translate_lines(["一", "二", "三"], "Chinese", "English"))

    assert result == ["one", "two", "three"]
    assert len(completions.requests) == 3


def test_results_come_back_in_the_caller_s_order():
    """They are gathered concurrently; the overlay indexes them positionally."""
    processor, _ = fake_processor({"a": "A", "b": "B", "c": "C"})
    translator = LineTranslator(processor=processor)

    assert asyncio.run(translator.translate_lines(["c", "a", "b"], "zh", "en")) == ["C", "A", "B"]


def test_concurrency_is_bounded():
    """A screen of thirty lines must not open thirty sockets at once."""
    processor, completions = fake_processor()
    translator = LineTranslator(processor=processor, concurrency=4)

    asyncio.run(translator.translate_lines([str(index) for index in range(20)], "zh", "en"))

    assert completions.peak_concurrent <= 4


def test_the_npu_model_is_sent_one_line_at_a_time():
    """The NPU serves requests in turn; a fan-out only queues them, slower."""
    processor, completions = fake_processor(installed=[NPU_TRANSLATION_MODEL])
    translator = LineTranslator(processor=processor, model=NPU_TRANSLATION_MODEL, concurrency=8)

    result = asyncio.run(translator.translate_lines(["a", "b", "c"], "zh", "en"))

    assert result == ["[a]", "[b]", "[c]"]
    assert completions.peak_concurrent == 1
    assert {request["model"] for request in completions.requests} == {NPU_TRANSLATION_MODEL}


def test_the_npu_model_is_registered_with_its_flm_checkpoint():
    """It is not in Lemonade's catalogue, so a pull by name finds nothing."""
    body = translation_pull_body(NPU_TRANSLATION_MODEL)

    assert body["model_name"] == NPU_TRANSLATION_MODEL
    assert body["recipe"] == "flm"
    assert body["checkpoints"] == {"main": "hy-mt2-flash:1.8b"}
    assert translation_pull_body(TRANSLATION_MODEL) is None


def test_a_missing_npu_model_says_how_to_register_it():
    processor, _ = fake_processor(installed=["Some-Chat-Model"])
    translator = LineTranslator(processor=processor, model=NPU_TRANSLATION_MODEL)

    with pytest.raises(TranslationModelUnavailable, match="--recipe flm"):
        asyncio.run(translator.translate_lines(["a"], "zh", "en"))


def test_nothing_to_translate_makes_no_requests():
    processor, completions = fake_processor()

    assert asyncio.run(LineTranslator(processor=processor).translate_lines([], "zh", "en")) == []
    assert completions.requests == []


def test_a_blank_line_is_passed_through_untouched():
    processor, completions = fake_processor()

    result = asyncio.run(LineTranslator(processor=processor).translate_lines(["  "], "zh", "en"))

    assert result == ["  "]
    assert completions.requests == []


# ── failure ──────────────────────────────────────────────────────────

def test_a_failed_line_falls_back_to_its_source():
    """A blank overlay box over live text is worse than the original text."""
    processor, _ = fake_processor(fail=True)

    result = asyncio.run(LineTranslator(processor=processor).translate_lines(["门"], "zh", "en"))

    assert result == ["门"]


def test_a_failed_line_is_not_cached():
    """Caching a failure would make it permanent for the rest of the session."""
    processor, _ = fake_processor(fail=True)
    translator = LineTranslator(processor=processor)

    asyncio.run(translator.translate_lines(["门"], "zh", "en"))

    assert len(translator.cache) == 0


# ── the cache ────────────────────────────────────────────────────────

def test_a_repeated_line_is_not_requested_twice():
    processor, completions = fake_processor({"门": "door"})
    translator = LineTranslator(processor=processor)

    asyncio.run(translator.translate_lines(["门"], "zh", "en"))
    result = asyncio.run(translator.translate_lines(["门"], "zh", "en"))

    assert result == ["door"]
    assert len(completions.requests) == 1


def test_only_the_uncached_lines_are_requested():
    processor, completions = fake_processor({"门": "door", "窗": "window"})
    translator = LineTranslator(processor=processor)

    asyncio.run(translator.translate_lines(["门"], "zh", "en"))
    completions.requests.clear()
    result = asyncio.run(translator.translate_lines(["门", "窗"], "zh", "en"))

    assert result == ["door", "window"]
    assert len(completions.requests) == 1


# ── model resolution ─────────────────────────────────────────────────

def test_the_pinned_model_is_used():
    processor, completions = fake_processor()

    asyncio.run(LineTranslator(processor=processor).translate_lines(["门"], "zh", "en"))

    assert completions.requests[0]["model"] == TRANSLATION_MODEL


def test_a_missing_model_fails_with_something_actionable():
    """Quietly using a different model turns into a quality bug report."""
    processor, _ = fake_processor(installed=["Some-Chat-Model"])

    with pytest.raises(TranslationModelUnavailable) as caught:
        asyncio.run(LineTranslator(processor=processor).translate_lines(["门"], "zh", "en"))

    assert TRANSLATION_MODEL in str(caught.value)


def test_the_fallback_is_opt_in():
    processor, completions = fake_processor(installed=["Some-Chat-Model"], routed="Some-Chat-Model")
    translator = LineTranslator(processor=processor, allow_fallback=True)

    asyncio.run(translator.translate_lines(["门"], "zh", "en"))

    assert completions.requests[0]["model"] == "Some-Chat-Model"


def test_an_undiscovered_server_is_not_treated_as_a_missing_model():
    """An empty model list means discovery has not run yet, not that the
    model is absent — failing there breaks a first tick that would work."""
    processor, completions = fake_processor(installed=[])

    asyncio.run(LineTranslator(processor=processor).translate_lines(["门"], "zh", "en"))

    assert completions.requests[0]["model"] == TRANSLATION_MODEL


# ── terminology ──────────────────────────────────────────────────────

def test_the_glossary_rides_along_as_reference_pairs():
    """Hy-MT2 has a terminology format, so a glossary does reach it — through
    the prompt, in the shape the wiki already stores it: source name to
    canonical target name."""
    prompt = build_translation_prompt("李云走进房间", "English", glossary={"李云": "Cloud Li"})

    assert prompt.startswith("Reference the following translations:\n李云 translates to Cloud Li\n\n")


def test_only_the_terms_in_this_line_are_referenced():
    """A wiki with five hundred entries would otherwise put all of them in
    front of every request."""
    glossary = {"李云": "Cloud Li", "碧刀门": "Jade Blade Sect"}

    assert relevant_terms("李云走进房间", glossary) == {"李云": "Cloud Li"}


def test_the_reference_list_is_capped():
    glossary = {f"名{index}": f"Name{index}" for index in range(40)}
    text = "".join(glossary)

    assert len(relevant_terms(text, glossary)) == MAX_TERMS_PER_REQUEST


def test_longer_terms_are_referenced_first():
    """A two-word term outranks a one-word term it contains."""
    glossary = {"碧刀": "Jade Blade", "碧刀门": "Jade Blade Sect"}

    assert list(relevant_terms("碧刀门弟子", glossary)) == ["碧刀门", "碧刀"]


def test_no_relevant_terms_leaves_the_prompt_plain():
    prompt = build_translation_prompt("门", "English", glossary={"李云": "Cloud Li"})

    assert not prompt.startswith("Reference")


def test_the_translator_passes_its_glossary_through(monkeypatch):
    processor, completions = fake_processor()
    translator = LineTranslator(processor=processor, glossary={"李云": "Cloud Li"})

    asyncio.run(translator.translate_lines(["李云"], "Chinese", "English"))

    assert "李云 translates to Cloud Li" in completions.requests[0]["messages"][0]["content"]


def test_the_translator_passes_its_style_through():
    processor, completions = fake_processor()
    translator = LineTranslator(processor=processor, style="terse")

    asyncio.run(translator.translate_lines(["门"], "Chinese", "English"))

    assert "conform to [terse]" in completions.requests[0]["messages"][0]["content"]


# ── the batched fallback ─────────────────────────────────────────────

def test_the_batched_path_sends_one_request():
    """Kept for a backend that will not serve the fan-out concurrently."""
    joined = BATCH_SEPARATOR.join(["一", "二"])
    processor, completions = fake_processor({joined: BATCH_SEPARATOR.join(["one", "two"])})
    translator = LineTranslator(processor=processor, batched=True)

    result = asyncio.run(translator.translate_lines(["一", "二"], "zh", "en"))

    assert result == ["one", "two"]
    assert len(completions.requests) == 1


def test_a_batch_that_loses_a_separator_falls_back_whole():
    """Index-mapped splitting misaligns every line after a dropped separator,
    which paints confident, wrong translations over live text."""
    joined = BATCH_SEPARATOR.join(["一", "二", "三"])
    processor, _ = fake_processor({joined: BATCH_SEPARATOR.join(["one", "two"])})
    translator = LineTranslator(processor=processor, batched=True)

    assert asyncio.run(translator.translate_lines(["一", "二", "三"], "zh", "en")) == ["一", "二", "三"]
