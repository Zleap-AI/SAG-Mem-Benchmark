"""Verbatim Mem0 Judge prompt, pinned to the authors' published evaluation code."""

# Prompt from mem0ai/mem0, Copyright 2023 Taranjeet Singh.
# Licensed under Apache-2.0; see docs/licenses/mem0-LICENSE.
# The surrounding module is adapted; ACCURACY_PROMPT is unchanged.

SOURCE_COMMIT = "ec1d7a45d3d0a11894a16cd7d05d6ba0257141f6"
SOURCE_URL = (
    f"https://github.com/mem0ai/mem0/blob/{SOURCE_COMMIT}/evaluation/metrics/llm_judge.py"
)
# Verbatim ACCURACY_PROMPT, including whitespace and Unicode quotation marks.
ACCURACY_PROMPT = (
    '\n'
    'Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given'
    ' the following data:\n'
    '    (1) a question (posed by one user to another user), \n'
    '    (2) a ’gold’ (ground truth) answer, \n'
    '    (3) a generated answer\n'
    'which you will score as CORRECT/WRONG.\n'
    '\n'
    'The point of the question is to ask about something one user should know about the other'
    ' user based on their prior conversations.\n'
    'The gold answer will usually be a concise and short answer that includes the referenced '
    'topic, for example:\n'
    'Question: Do you remember what I got the last time I went to Hawaii?\n'
    'Gold answer: A shell necklace\n'
    'The generated answer might be much longer, but you should be generous with your grading '
    '- as long as it touches on the same topic as the gold answer, it should be counted as CO'
    'RRECT. \n'
    '\n'
    'For time related questions, the gold answer will be a specific date, month, year, etc. T'
    'he generated answer might be much longer or use relative time references (like "last Tue'
    'sday" or "next month"), but you should be generous with your grading - as long as it ref'
    'ers to the same date or time period as the gold answer, it should be counted as CORRECT.'
    ' Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it\'s th'
    'e same date.\n'
    '\n'
    'Now it’s time for the real question:\n'
    'Question: {question}\n'
    'Gold answer: {gold_answer}\n'
    'Generated answer: {generated_answer}\n'
    '\n'
    'First, provide a short (one sentence) explanation of your reasoning, then finish with CO'
    'RRECT or WRONG. \n'
    'Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation '
    'script.\n'
    '\n'
    'Just return the label CORRECT or WRONG in a json format with the key as "label".\n'
)
