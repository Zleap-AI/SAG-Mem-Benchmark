"""Deterministic explanations of source-date-relative English and Chinese text.

These rules resolve expressions, not events. They do not infer a date for
durations, quoted speech, or phrases that refer to another event.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class TimeExplanation:
    expression: str
    value: str
    granularity: str
    start: int
    end: int
    approximate: bool = False


_FIXED: dict[str, tuple[str, int]] = {}
for offset, phrases in (
    (-3, ("大前天",)),
    (-2, ("the day before yesterday", "day before yesterday", "前天")),
    (-1, ("yesterday", "昨天", "昨日")),
    (0, ("today", "今天", "今日")),
    (1, ("tomorrow", "明天", "明日")),
    (2, ("the day after tomorrow", "day after tomorrow", "后天")),
    (3, ("大后天",)),
):
    _FIXED.update({phrase: ("day", offset) for phrase in phrases})
for unit, groups in (
    ("month", (
        (-2, ("the month before last", "上上个月")),
        (-1, ("last month", "上个月", "上月")),
        (0, ("this month", "这个月", "本月", "这月")),
        (1, ("next month", "下个月", "下月")),
        (2, ("下下个月",)),
    )),
    ("year", (
        (-2, ("the year before last", "前年")),
        (-1, ("last year", "去年")),
        (0, ("this year", "今年")),
        (1, ("next year", "明年")),
        (2, ("后年",)),
    )),
    ("week", (
        (-2, ("上上周", "上上星期")),
        (-1, ("last week", "上周", "上星期", "上个星期", "上礼拜")),
        (0, ("this week", "本周", "这周", "这星期", "这个星期", "本星期")),
        (1, ("next week", "下周", "下星期", "下个星期", "下礼拜")),
        (2, ("下下周", "下下星期")),
    )),
):
    for offset, phrases in groups:
        _FIXED.update({phrase: (unit, offset) for phrase in phrases})

_SMALL = dict(zip(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"], range(20), strict=True))
_TENS = dict(zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10),
                 strict=True))
_SMALL.update({"a": 1, "an": 1})
_ONES = "one|two|three|four|five|six|seven|eight|nine"
_EN_NUMBER = (r"(?:[0-9]{1,3}|(?:" + "|".join(_TENS) + r")(?:[-\s](?:" + _ONES + r"))?|"
              + "|".join(sorted(_SMALL, key=len, reverse=True)) + ")")
_CN_DIGITS = "零〇一二两三四五六七八九"
_CN_VALUES = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_NUMBER = rf"(?:[0-9]{{1,3}}|[一二两三四五六七八九]?十[一二三四五六七八九]?|[{_CN_DIGITS}])"
_EN_UNIT = r"(?P<unit>days?|weeks?|months?|years?)"
_FIXED_PATTERN = "|".join(re.escape(p).replace(r"\ ", r"\s+")
                          for p in sorted(_FIXED, key=len, reverse=True))
_PATTERNS = (
    (re.compile(r"(?<![A-Za-z])(?:" + _FIXED_PATTERN + r")(?![A-Za-z])", re.IGNORECASE), "fixed"),
    (re.compile(rf"(?<![A-Za-z0-9-])(?P<number>{_EN_NUMBER})\s+{_EN_UNIT}\s+ago\b", re.IGNORECASE), "past"),
    (re.compile(rf"(?<![A-Za-z0-9-])(?P<number>{_EN_NUMBER})\s+{_EN_UNIT}\s+from\s+now\b", re.IGNORECASE),
     "future"),
    (re.compile(rf"(?<![{_CN_DIGITS}十百千万0-9])(?P<number>{_CN_NUMBER})(?:个)?"
                r"(?P<unit>天|日|周|星期|礼拜|月|年)(?P<direction>以前|之前|以后|之后|前|后)"), "counted_zh"),
    (re.compile(r"(?P<direction>上上|下下|上|下|本|这)(?:个)?(?:周|星期|礼拜)"
                r"(?P<weekday>[一二三四五六日天])"), "weekday_zh"),
    (re.compile(r"\b(?P<direction>last|next)\s+"
                r"(?P<weekday>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
                re.IGNORECASE), "weekday_en"),
)
_QUOTED = re.compile(
    r'''```.*?(?:```|\Z)|`[^`\n]*`|"[^"\n]*"|(?<!\w)'(?:[^'\n]|(?<=\w)'(?=\w))*'(?!\w)'''
    r"|“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』", re.DOTALL)
_APPROXIMATE_PREFIX = re.compile(r"(?:\b(?:about|around|roughly|approximately)|大约|大概|约)\s*$",
                                 re.IGNORECASE)
_UNSUPPORTED_PREFIX = re.compile(
    r"(?:\b(?:almost|nearly|over|more than|less than|at least|at most)|将近|不到|至少|至多|超过)\s*$",
    re.IGNORECASE)
_RANGE_PREFIX = re.compile(r"(?:\b(?:or|and|to)|或|到|至)\s*$", re.IGNORECASE)


def _number(text: str) -> int:
    if text.isascii() and text.isdigit():
        return int(text)
    if text in _CN_DIGITS:
        return _CN_VALUES[text]
    if "十" in text:
        tens, ones = text.split("十")
        return (10 if not tens else _number(tens) * 10) + (_number(ones) if ones else 0)
    words = text.lower().replace("-", " ").split()
    return sum((_TENS | _SMALL)[word] for word in words)


def _value(source_date: date, unit: str, offset: int) -> tuple[str, str]:
    if unit == "day":
        return (source_date + timedelta(days=offset)).isoformat(), "day"
    if unit == "week":
        monday = source_date - timedelta(days=source_date.weekday()) + timedelta(weeks=offset)
        return f"{monday.isoformat()}..{(monday + timedelta(days=6)).isoformat()}", "week"
    if unit == "month":
        year, month = divmod(source_date.year * 12 + source_date.month - 1 + offset, 12)
        return date(year, month + 1, 1).strftime("%Y-%m"), "month"
    return str(date(source_date.year + offset, 1, 1).year), "year"


def find_relative_times(text: str, source_date: date) -> list[TimeExplanation]:
    """Resolve supported expressions relative to this message's local calendar date.

    Calendar weeks start on Monday. Counted weeks ("two weeks ago") mean a
    multiple of seven days; counted months and years retain month/year precision.
    English last/next weekdays are strictly before/after the source date.
    Approximate expressions retain uncertainty without inventing a date range.
    Bare "in N days" (which may be a duration), quoted text and code are skipped.
    """
    ignored = [(m.start(), m.end()) for m in _QUOTED.finditer(text)]
    matches = sorted(((m, kind) for pattern, kind in _PATTERNS for m in pattern.finditer(text)),
                     key=lambda item: (item[0].start(), -len(item[0][0])))
    result = []
    end = 0
    for match, kind in matches:
        prefix_start = max(0, match.start() - 40)
        prefix = text[prefix_start:match.start()]
        approximation = _APPROXIMATE_PREFIX.search(prefix)
        start = prefix_start + approximation.start() if approximation else match.start()
        qualifier_prefix = prefix[:approximation.start()] if approximation else prefix
        if start < end or any(start < b and match.end() > a for a, b in ignored):
            continue
        if _UNSUPPORTED_PREFIX.search(qualifier_prefix) or (kind in {"past", "future", "counted_zh"}
                                                          and _RANGE_PREFIX.search(qualifier_prefix)):
            continue
        if kind == "fixed":
            unit, offset = _FIXED[" ".join(match[0].lower().split())]
            value, granularity = _value(source_date, unit, offset)
        elif kind == "weekday_zh":
            weeks = {"上上": -2, "上": -1, "本": 0, "这": 0, "下": 1, "下下": 2}[match["direction"]]
            weekday = "一二三四五六日".index(match["weekday"].replace("天", "日"))
            value, granularity = _value(source_date, "day", 7 * weeks + weekday - source_date.weekday())
        elif kind == "weekday_en":
            weekday = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday").index(
                match["weekday"].lower())
            if match["direction"].lower() == "last":
                offset = -((source_date.weekday() - weekday) % 7 or 7)
            else:
                offset = (weekday - source_date.weekday()) % 7 or 7
            value, granularity = _value(source_date, "day", offset)
        else:
            offset = _number(match["number"])
            direction = -1 if kind == "past" or (kind == "counted_zh" and "前" in match["direction"]) else 1
            unit = match["unit"].lower().rstrip("s")
            unit = {"天": "day", "日": "day", "周": "week", "星期": "week", "礼拜": "week",
                    "月": "month", "年": "year"}.get(unit, unit)
            offset *= direction
            if unit == "week":
                unit, offset = "day", offset * 7
            value, granularity = _value(source_date, unit, offset)
        result.append(TimeExplanation(text[start:match.end()], value, granularity, start, match.end(),
                                      approximate=approximation is not None))
        end = match.end()
    return result


def add_time_explanations(text: str, source_date: date) -> tuple[str, list[TimeExplanation]]:
    """Insert a date in fullwidth parentheses after every matched expression."""
    matches = find_relative_times(text, source_date)
    if not matches:
        return text, matches
    pieces = []
    cursor = 0
    for match in matches:
        displayed = match.value.replace("-", "/").replace("..", " ~ ")
        pieces.append(text[cursor:match.end])
        pieces.append(f"（{'约' if match.approximate else ''}{displayed}）")
        cursor = match.end
    pieces.append(text[cursor:])
    return "".join(pieces), matches
