"""Verbatim prompt constants from the official Zep LongMemEval harness.

Source: getzep/zep benchmarks/longmemeval/zep_longmem_eval.py
These prompts follow the official Zep harness, with a project-specific
abstention grading rule added for LongMemEval abstention questions.
"""

READER_SYSTEM_PROMPT = """
        You are a helpful expert assistant answering questions from lme_experiment users based on the provided context.
        """

READER_USER_PROMPT = """
        Your task is to briefly answer the question. You are given the following context from the previous conversation. If you don't know how to answer the question, abstain from answering.
        
        Context: {context}
        
        Question: {question}
        """

GRADER_SYSTEM_PROMPT = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

GRADING_PROMPTS = {
    "temporal-reasoning": """
 I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. 

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
Accept abstention only when the correct answer itself indicates that the information is unavailable or not mentioned; otherwise, abstention is incorrect.
""",
    "knowledge-update": """
 I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer. 

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
Accept abstention only when the correct answer itself indicates that the information is unavailable or not mentioned; otherwise, abstention is incorrect.
""",
    "single-session-preference": """
 I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly. 

<QUESTION>
B: {question}
</QUESTION>
<RUBRIC>
{gold_answer}
</RUBRIC>
<RESPONSE>
A: {response}
</RESPONSE>
Accept abstention only when the correct answer itself indicates that the information is unavailable or not mentioned; otherwise, abstention is incorrect.
""",
    "default": """
 I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. 

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
Accept abstention only when the correct answer itself indicates that the information is unavailable or not mentioned; otherwise, abstention is incorrect.
""",
}

EVIDENCE_TEMPLATE = """
<EVIDENCE>
{evidence}
</EVIDENCE>
"""

FUSION_TEMPLATE = """
<MEMORY>
{memory_context}
</MEMORY>

<EVIDENCE>
{evidence}
</EVIDENCE>
"""
