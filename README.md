# Chain-of-Agents Framework for Long-Text Summarization

This repository contains a Chain-of-Agents framework for long-text summarization using OpenAI's GPT-4 model. The framework splits a long document into logical parts, processes each part, and synthesizes a final summary.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/richardwhiteii/chain-of-agents.git
    cd chain-of-agents
    ```

2. Install the required dependencies:
    ```bash
    pip install openai
    ```

3. Configure your OpenAI API key:
    ```python
    import openai
    openai.api_key = "YOUR_API_KEY"
    ```

## Functions in `main.py`

### `split_document_with_llm(document)`
Uses an LLM to dynamically split a document into logical parts.

### `worker_agent(chunk, previous_output)`
Processes a single chunk of input and combines it with the previous output.

### `manager_agent(worker_outputs)`
Combines outputs from all worker agents into a final coherent result.

### `chain_of_agents(long_text, chunk_size=100)`
Implements the Chain-of-Agents framework for long-text summarization.

## Example Usage

```python
from main import chain_of_agents

long_text = """
Insert a very long text here for testing the Chain-of-Agents framework.
It can be a multi-paragraph document or article.
"""
final_summary = chain_of_agents(long_text)
print("Final Summary:\n", final_summary)
```
