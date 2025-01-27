# Chain-of-Agents Framework

The Chain-of-Agents framework is designed to process and summarize long texts using a series of worker and manager agents. Each worker agent processes a chunk of the text and provides a summary, which is then combined by the manager agent into a final coherent summary.

## Overview

The framework consists of three main components:
1. **Worker Agent**: Processes a single chunk of input and combines it with the previous output.
2. **Manager Agent**: Combines outputs from all worker agents into a final coherent result.
3. **Chain of Agents**: Implements the framework for long-text summarization by splitting the input text into chunks, processing each chunk with worker agents, and then combining the results with the manager agent.

## Usage Instructions

1. **Configure OpenAI API Key**: Set your OpenAI API key in the `main.py` file.
    ```python
    openai.api_key = "YOUR_API_KEY"
    ```

2. **Worker Agent**: Define the worker agent function to process a single chunk of input and combine it with the previous output.
    ```python
    def worker_agent(chunk, previous_output):
        prompt = f"""
        You are a worker agent in a Chain-of-Agents system.
        Here is your task:
        1. Read the following text chunk.
        2. Summarize the key points in a concise manner.
        3. Combine the new summary with the previous information.

        Previous output: {previous_output}
        Current chunk: {chunk}

        Provide an updated summary:
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response['choices'][0]['message']['content'].strip()
    ```

3. **Manager Agent**: Define the manager agent function to combine outputs from all worker agents into a final coherent result.
    ```python
    def manager_agent(worker_outputs):
        prompt = f"""
        You are the manager agent in a Chain-of-Agents system.
        Your task is to synthesize the summaries provided by worker agents into a coherent final output.

        Worker outputs: {worker_outputs}

        Provide the final summary:
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response['choices'][0]['message']['content'].strip()
    ```

4. **Chain of Agents**: Define the chain of agents function to implement the framework for long-text summarization.
    ```python
    def chain_of_agents(long_text, chunk_size=100):
        chunks = [long_text[i:i + chunk_size] for i in range(0, len(long_text, chunk_size))]
        
        previous_output = ""
        worker_outputs = []

        for chunk in chunks:
            previous_output = worker_agent(chunk, previous_output)
            worker_outputs.append(previous_output)

        final_output = manager_agent(worker_outputs)
        return final_output
    ```

## Example Usage

Here is an example of how to use the Chain-of-Agents framework to summarize a long text:

```python
long_text = """
Insert a very long text here for testing the Chain-of-Agents framework.
It can be a multi-paragraph document or article.
"""
final_summary = chain_of_agents(long_text)
print("Final Summary:\n", final_summary)
```

Replace `"YOUR_API_KEY"` with your actual OpenAI API key and `"Insert a very long text here for testing the Chain-of-Agents framework."` with the text you want to summarize.
