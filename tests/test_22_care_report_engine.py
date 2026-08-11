from __future__ import annotations

from types import SimpleNamespace

from care_report_engine import generate_care_report, normalize_care_report


def valid_report():
    return {
        "status": "signals_present",
        "summary": "建议本周主动联系一次，听听近况。",
        "signals": [
            {
                "area": "psychological",
                "action_type": "contact_companionship",
                "topic": "陪伴与联系",
                "strength": "low",
                "trend": "repeated",
                "family_action": "本周主动打一次电话，耐心听听近况。",
                "source_turn_ids": [1, 2],
            }
        ],
    }


def test_normalizer_rejects_diagnosis_and_unverifiable_sources():
    report = valid_report()
    report["summary"] = "已经确诊某种疾病。"
    assert normalize_care_report(report, {1, 2}) is None

    report = valid_report()
    report["signals"][0]["source_turn_ids"] = [99]
    normalized = normalize_care_report(report, {1, 2})
    assert normalized["status"] == "insufficient_or_conflicting"
    assert normalized["signals"] == []
    assert normalized["summary"] == "建议本周主动联系一次，听听近况。"


def test_normalizer_rejects_copied_source_words_and_incompatible_actions():
    report = valid_report()
    report["summary"] = "不要把这句话展示给家人，建议多联系。"
    assert normalize_care_report(report, {1, 2}, ["不要把这句话展示给家人，这是聊天原话。"]) is None

    report = valid_report()
    report["signals"][0]["action_type"] = "safety_check"
    normalized = normalize_care_report(report, {1, 2})
    assert normalized["status"] == "insufficient_or_conflicting"
    assert normalized["signals"] == []


def test_generator_sends_only_elder_turns_and_requires_json():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='''{"status":"signals_present","summary":"建议联系一次。","signals":[{"area":"psychological","action_type":"contact_companionship","topic":"陪伴","strength":"low","trend":"new","family_action":"本周主动问候一次。","source_turn_ids":[1,2]}]}'''
                        )
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    report = generate_care_report(
        client,
        "deepseek-v4-flash",
        [
            {"role": "assistant", "content": "assistant-only text"},
            {"role": "user", "content": "今天有点安静。"},
            {"role": "assistant", "content": "assistant-only follow-up"},
            {"role": "user", "content": "最近很少有人陪我聊天。"},
        ],
    )

    assert report["status"] == "signals_present"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "assistant-only text" not in calls[0]["messages"][1]["content"]
