from datetime import date

import pytest

from src.harness.relative_time import add_time_explanations, find_relative_times


@pytest.mark.parametrize(("text", "value", "granularity"), [
    ("yesterday", "2023-05-20", "day"),
    ("YESTERDAY", "2023-05-20", "day"),
    ("the day before yesterday", "2023-05-19", "day"),
    ("the day after tomorrow", "2023-05-23", "day"),
    ("today", "2023-05-21", "day"),
    ("明天", "2023-05-22", "day"),
    ("我昨天去了", "2023-05-20", "day"),
    ("大前天", "2023-05-18", "day"),
    ("大后天", "2023-05-24", "day"),
    ("last month", "2023-04", "month"),
    ("last   month", "2023-04", "month"),
    ("this month", "2023-05", "month"),
    ("下个月", "2023-06", "month"),
    ("上上个月", "2023-03", "month"),
    ("last year", "2022", "year"),
    ("前年", "2021", "year"),
    ("后年", "2025", "year"),
    ("last week", "2023-05-08..2023-05-14", "week"),
    ("本周", "2023-05-15..2023-05-21", "week"),
    ("下周", "2023-05-22..2023-05-28", "week"),
    ("上周五", "2023-05-12", "day"),
    ("下星期一", "2023-05-22", "day"),
    ("本周日", "2023-05-21", "day"),
    ("three days ago", "2023-05-18", "day"),
    ("a month ago", "2023-04", "month"),
    ("two weeks ago", "2023-05-07", "day"),
    ("two days from now", "2023-05-23", "day"),
    ("three months from now", "2023-08", "month"),
    ("twenty-one days ago", "2023-04-30", "day"),
    ("九天前", "2023-05-12", "day"),
    ("两个月前", "2023-03", "month"),
    ("十二天前", "2023-05-09", "day"),
    ("二十三天前", "2023-04-28", "day"),
    ("十年以前", "2013", "year"),
    ("〇天前", "2023-05-21", "day"),
])
def test_supported_expressions_keep_their_precision(text, value, granularity):
    matches = find_relative_times(text, date(2023, 5, 21))
    assert [(m.value, m.granularity) for m in matches] == [(value, granularity)]
    assert text[matches[0].start:matches[0].end] == matches[0].expression


@pytest.mark.parametrize(("source", "text", "expected"), [
    (date(2024, 3, 1), "yesterday", "2024-02-29"),
    (date(2023, 3, 1), "yesterday", "2023-02-28"),
    (date(2023, 1, 31), "last month", "2022-12"),
    (date(2024, 3, 31), "a month ago", "2024-02"),
    (date(2023, 12, 31), "next month", "2024-01"),
    (date(2023, 1, 2), "last week", "2022-12-26..2023-01-01"),
])
def test_calendar_boundaries_do_not_invent_a_day(source, text, expected):
    assert find_relative_times(text, source)[0].value == expected


@pytest.mark.parametrize("text", [
    "recently", "in recent years", "最近", "此前", "后来", "前几天", "几个月前", "三四天前",
    "两百年前", "two or three days ago", "3-4 days ago", "about two or three days ago",
    "more than two days ago", "不到三天前", "至少两个月前",
    "three months in advance", "three days before the wedding", "the next day",
    'She said "I will go tomorrow".', "She said 'I'll go tomorrow'.", "我说过“明天出发”", "`yesterday`",
    'The song is called "Yesterday".', 'The song is called \'Yesterday\'.', "```python\nyesterday\n```",
    "Rome wasn't built in a day.", "7 international trips in a year", "three audiobooks in six weeks",
    "It's hard to see everything in one day.", "The deadline is in two weeks.",
])
def test_ambiguous_event_relative_or_quoted_expressions_are_unchanged(text):
    assert add_time_explanations(text, date(2023, 5, 21)) == (text, [])


def test_each_occurrence_is_annotated_inline_without_an_explanation_paragraph():
    body = "I left yesterday. I called yesterday. 我下个月回来，去年也来过。"
    result, matches = add_time_explanations(body, date(2023, 5, 21))
    assert result == (
        "I left yesterday（2023/05/20）. I called yesterday（2023/05/20）. "
        "我下个月（2023/06）回来，去年（2022）也来过。")
    assert "Time clarification" not in result
    assert len(matches) == 4


def test_conjunction_between_fixed_dates_does_not_turn_them_into_a_quantity_range():
    assert [m.value for m in find_relative_times("yesterday and today", date(2023, 5, 21))] == [
        "2023-05-20", "2023-05-21"]


@pytest.mark.parametrize(("expression", "meaning"), [
    ("yesterday", "2023/05/07"),
    ("last month", "2023/04"),
    ("last year", "2022"),
    ("last week", "2023/05/01 ~ 2023/05/07"),
])
def test_inline_annotation_preserves_precision(expression, meaning):
    result, _ = add_time_explanations(expression, date(2023, 5, 8))
    assert result == expression + f"（{meaning}）"


@pytest.mark.parametrize(("text", "value", "granularity", "displayed"), [
    ("about two days ago", "2023-05-06", "day", "2023/05/06"),
    ("roughly two weeks ago", "2023-04-24", "day", "2023/04/24"),
    ("around a month ago", "2023-04", "month", "2023/04"),
    ("approximately last year", "2022", "year", "2022"),
    ("大约三天前", "2023-05-05", "day", "2023/05/05"),
    ("大概两个月前", "2023-03", "month", "2023/03"),
    ("约三年前", "2020", "year", "2020"),
])
def test_approximations_preserve_the_marker_and_original_precision(text, value, granularity, displayed):
    rendered, matches = add_time_explanations(text, date(2023, 5, 8))
    assert len(matches) == 1
    match = matches[0]
    assert match.expression == text and match.approximate is True
    assert match.value == value and match.granularity == granularity
    assert rendered == f"{text}（约{displayed}）"


@pytest.mark.parametrize(("source", "displayed"), [
    (date(2023, 1, 10), "2023/01/10"),
    (date(2024, 2, 29), "2024/02/29"),
    (date(2023, 12, 31), "2023/12/31"),
])
def test_day_display_has_zero_padding_without_weekday(source, displayed):
    assert add_time_explanations("today", source)[0] == f"today（{displayed}）"


@pytest.mark.parametrize(("source", "text", "value"), [
    (date(2023, 5, 8), "last Friday", "2023-05-05"),
    (date(2023, 5, 8), "next Monday", "2023-05-15"),
    (date(2023, 5, 8), "last Monday", "2023-05-01"),
    (date(2023, 5, 12), "next Friday", "2023-05-19"),
    (date(2023, 5, 14), "next Monday", "2023-05-15"),
    (date(2023, 5, 14), "LAST FRIDAY", "2023-05-12"),
    (date(2024, 1, 1), "last Sunday", "2023-12-31"),
    (date(2023, 12, 31), "next Monday", "2024-01-01"),
])
def test_english_weekdays_use_strictly_previous_or_next_occurrence(source, text, value):
    match = find_relative_times(text, source)[0]
    assert match.expression == text and match.value == value
    assert match.granularity == "day" and not match.approximate


def test_quoted_new_rules_remain_untouched_and_approximation_does_not_leak():
    text = '"about two days ago" and "next Monday". 大约三天前去了，yesterday回来了。'
    matches = find_relative_times(text, date(2023, 5, 8))
    assert [(m.expression, m.approximate) for m in matches] == [("大约三天前", True), ("yesterday", False)]
