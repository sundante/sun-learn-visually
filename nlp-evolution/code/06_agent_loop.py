"""
06 · A minimal agent loop: tool calling + retrieval (RAG)
=========================================================
RAG: Lewis et al., 2020. ReAct (reason + act): Yao et al., 2022.
Function/tool calling became a standard LLM API feature in 2023; the Model
Context Protocol (MCP, Anthropic, Nov 2024) standardised how tools are exposed.

Every agent framework (LangGraph, the Claude Agent SDK, OpenAI Agents, ...) is a
dressed-up version of this loop:

    messages -> LLM -> { tool_call? -> run tool -> append result -> loop | final answer -> stop }

The LLM here is MOCKED so the file runs offline. Swap `mock_llm` for a real
chat-completions call that returns the same {"tool": ...} / {"answer": ...} shape.

Run:  python 06_agent_loop.py
"""
import json
import math
import re
from collections import Counter

# ---------- 1. A tiny knowledge base + retriever (the "R" in RAG) ----------
DOCS = [
    "Mixtral 8x7B routes each token to 2 of 8 experts, about 13B active parameters.",
    "Grouped-query attention shares key/value heads across query heads to shrink the KV cache.",
    "LoRA freezes pre-trained weights and trains low-rank matrices A and B.",
    "RoPE rotates query and key vectors so attention depends on relative position.",
    "DPO aligns a model with preference pairs without training a separate reward model.",
]

def embed(text):            # bag-of-words stand-in for a real embedding model
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))

def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a)
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())) or 1)

def search_docs(query: str, k: int = 2):
    q = embed(query)
    ranked = sorted(DOCS, key=lambda d: cosine(q, embed(d)), reverse=True)
    return ranked[:k]

def calculator(expression: str):
    if not re.fullmatch(r"[\d\s.+\-*/()]+", expression):
        return "error: only arithmetic is allowed"
    return str(eval(expression))   # safe here: input is restricted to digits and operators

# ---------- 2. Tool registry with JSON-schema descriptions (what the LLM sees) ----------
TOOLS = {
    "search_docs": {"fn": search_docs, "description": "Search the knowledge base.",
                    "parameters": {"query": "string"}},
    "calculator": {"fn": calculator, "description": "Evaluate arithmetic.",
                   "parameters": {"expression": "string"}},
}

# ---------- 3. Mock LLM: decides the next action from the conversation so far ----------
def mock_llm(messages):
    tool_results = [m for m in messages if m["role"] == "tool"]
    if not tool_results:
        return {"tool": "search_docs", "args": {"query": "Mixtral experts active parameters"}}
    if len(tool_results) == 1:
        return {"tool": "calculator", "args": {"expression": "13 / 47 * 100"}}
    pct = float(tool_results[-1]["content"])
    return {"answer": f"Mixtral activates 2 of 8 experts per token, so only ~13B of ~47B "
                      f"parameters ({pct:.0f}%) do work for each token."}

# ---------- 4. The agent loop ----------
def run_agent(question, llm=mock_llm, max_steps=5):
    messages = [{"role": "system", "content": "Tools: " + json.dumps({k: v["description"] for k, v in TOOLS.items()})},
                {"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        action = llm(messages)
        if "answer" in action:
            print(f"[{step}] ANSWER  {action['answer']}")
            return action["answer"]
        result = TOOLS[action["tool"]]["fn"](**action["args"])
        print(f"[{step}] TOOL    {action['tool']}({action['args']}) -> {result}")
        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "tool", "content": result if isinstance(result, str) else json.dumps(result)})
    return "stopped: step limit reached"   # always cap agent loops


if __name__ == "__main__":
    answer = run_agent("What share of Mixtral's parameters is active per token?")
    assert "28%" in answer
    print("✓ Agent completed retrieve -> calculate -> answer in 3 steps")
