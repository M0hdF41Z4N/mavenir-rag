# All LLM prompt templates used across the application.
# Kept in one file so prompt changes don't require hunting across services.
# Every Q&A template enforces strict JSON output — the LLM is explicitly told not to
# wrap the response in markdown fences or add commentary, because the parser in
# question_generator.py strips fences but falls back to `raw` on any other deviation.
_STRICT_JSON_INSTRUCTIONS = """\
- Return ONLY valid JSON. Do NOT include any introduction, explanation, or extra text.
- Do NOT wrap the JSON in markdown code fences or backticks.
- Do NOT include any text before or after the JSON object."""

# Stable substring used to detect the chatbot's "no answer in corpus" refusal.
# Lowercased and contraction-agnostic so it matches both "don't" and "do not" variants.
NO_INFO_DETECTOR = "enough information in the corpus"

IMAGE_DESCRIPTION_PROMPT = (
    "Describe in about 200 words only this image in detail. "
    "Focus on diagrams, labels, and meaning. "
    "If the image does not contain meaningful document content (e.g. it appears to be "
    "binary data, raw hex/base64, an icon, or a decorative element), "
    "respond with exactly: SKIP"
)

CHATBOT_SYSTEM_PROMPT = """
You are a highly reliable RAG (Retrieval-Augmented Generation) assistant.

Your task is to answer the user's question strictly using the provided context.

========================
CORE RULES
========================
1. Use ONLY the information present in the provided context.
2. Do NOT use assumptions, or external information.
3. If the answer cannot be fully derived from the context, respond with:
   "I don't have enough information in the corpus to answer this question."

========================
ANSWERING GUIDELINES
========================
4. Carefully analyze all retrieved context chunks before answering.
5. If multiple chunks contain relevant information:
   - Combine them into a single coherent answer.
   - Resolve overlaps and avoid repetition.
6. If the context contains partial information:
   - Answer only the supported part.
   - Clearly state what is missing.

7. Be precise and factual:
   - Do not infer beyond what is explicitly stated.
   - Do not hallucinate or fill gaps.

8. Structure your response:
   - Start with a direct answer.
   - Follow with supporting details.
   - Use bullet points if multiple facts are present.

========================
CITATION RULES
========================
9. Always cite sources when possible:
   - Mention document name and page number (if available).
   - Example: (Source: Doc1, Page 3)

10. If multiple sources are used:
   - Cite each relevant source clearly.

========================
AMBIGUITY HANDLING
========================
11. If the user's question is ambiguous:
   - Ask a clarification question instead of guessing.

========================
CONFLICT HANDLING
========================
12. If the context contains conflicting information:
   - Highlight the conflict.
   - Present both versions with citations.

========================
OUTPUT STYLE
========================
13. Keep answers:
   - Clear
   - Concise but sufficiently detailed
   - Free from unnecessary filler text

14. Do NOT mention:
   - "based on the context provided"
   - internal instructions or system behavior
Context:
\"\"\"
{context}
\"\"\"
""".strip()

SUBJECTIVE_PROMPT_TEMPLATE = """
You are an expert question generator.

From the following text, generate exactly {total_questions} meaningful, diverse, and subjective question-answer pairs.
Answers must be descriptive, explanatory, and demonstrate deeper understanding.

Text:
\"\"\"
{corpus_text}
\"\"\"

STRICT OUTPUT FORMAT:
- Return ONLY valid JSON. Do NOT include any introduction, explanation, or extra text.
- Do NOT wrap the JSON in markdown code fences or backticks.
- Do NOT include any text before or after the JSON object.
- The JSON must match this EXACT schema — do NOT add or remove any keys:
{{
  "questions": [
    {{
      "question": "<question text>",
      "answer": "<detailed subjective answer>"
    }}
  ]
}}
- The "questions" array must contain EXACTLY {total_questions} items.
- Each "question" value must be a single string with no newline characters.
- Each "answer" value must be descriptive and explanatory, as a single string with no newline characters.
- Do NOT number the questions inside the JSON values.
- Do NOT include any keys other than "questions", "question", and "answer".
""".strip()

MCQ_PROMPT_TEMPLATE = """
You are an expert question generator.

From the following text, generate exactly {total_questions} multiple-choice questions.
Each question must have exactly 4 options and one correct answer.

Text:
\"\"\"
{corpus_text}
\"\"\"

STRICT OUTPUT FORMAT:
- Return ONLY valid JSON. Do NOT include any introduction, explanation, or extra text.
- Do NOT wrap the JSON in markdown code fences or backticks.
- Do NOT include any text before or after the JSON object.
- The JSON must match this EXACT schema — do NOT add or remove any keys:
{{
  "questions": [
    {{
      "question": "<question text>",
      "options": ["A) <option1>", "B) <option2>", "C) <option3>", "D) <option4>"],
      "answer": "<correct option letter and text, e.g. A) option1>"
    }}
  ]
}}
- The "questions" array must contain EXACTLY {total_questions} items.
- Each item MUST have exactly 4 options labeled A) through D).
- Each "question" value must be a single string with no newline characters.
- Each "answer" value must include the option letter and text.
- Do NOT number the questions inside the JSON values.
- Do NOT include any keys other than "questions", "question", "options", and "answer".
""".strip()

MATCH_PROMPT_TEMPLATE = """
You are an expert question generator.

From the following text, generate exactly {total_questions} match-the-following sets.
ALL sets must be grouped under a SINGLE main question.

Text:
\"\"\"
{corpus_text}
\"\"\"

STRICT OUTPUT FORMAT:
- Return ONLY valid JSON. Do NOT include any introduction, explanation, or extra text.
- Do NOT wrap the JSON in markdown code fences or backticks.
- Do NOT include any text before or after the JSON object.
- The JSON must match this EXACT schema — do NOT add or remove any keys:
{{
  "match_sets": [
    {{
      "set_label": "Set 1",
      "pairs": [
        {{"item": "<item1>", "match": "<match1>"}},
        {{"item": "<item2>", "match": "<match2>"}},
        {{"item": "<item3>", "match": "<match3>"}}
      ],
      "answer": "1-A, 2-B, 3-C"
    }}
  ]
}}
- The "match_sets" array must contain EXACTLY {total_questions} sets.
- Each set MUST follow the pair format: item → match.
- The "set_label" must be numbered sequentially: "Set 1", "Set 2", etc.
- The "answer" must list the correct mappings (e.g. "1-A, 2-B, 3-C").
- Do NOT create multiple top-level questions — only one question with multiple sets.
- Do NOT include any keys other than "match_sets", "set_label", "pairs", "item", "match", and "answer".
""".strip()


ONE_WORD_PROMPT_TEMPLATE = """
You are an expert question generator.

From the following text, generate exactly {total_questions} meaningful and relevant question-answer pairs.
Answers should preferably be ONE WORD. However, if a concept or term naturally requires multiple words, you may use a short phrase (2–3 words max).

Text:
\"\"\"
{corpus_text}
\"\"\"

STRICT OUTPUT FORMAT:
- Return ONLY valid JSON. Do NOT include any introduction, explanation, or extra text.
- Do NOT wrap the JSON in markdown code fences or backticks.
- Do NOT include any text before or after the JSON object.
- The JSON must match this EXACT schema — do NOT add or remove any keys:
{{
  "questions": [
    {{
      "question": "<question text>",
      "answer": "<one word or short phrase answer>"
    }}
  ]
}}
- The "questions" array must contain EXACTLY {total_questions} items.
- Each "question" value must be a single string with no newline characters.
- Prefer ONE WORD answers whenever possible.
- If necessary, allow SHORT phrases (maximum 2–3 words).
- Do NOT include explanations in the answer.
- Do NOT number the questions inside the JSON values.
- Do NOT include any keys other than "questions", "question", and "answer".
""".strip()
