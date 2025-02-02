import openai

# Configure OpenAI API Key
openai.api_key = "YOUR_API_KEY"

def split_document_with_llm(document):
    """
    Use an LLM to dynamically split a document into logical parts.
    """
    prompt = f"""
    You are an AI assistant that specializes in analyzing and splitting large documents into logical sections. 
    The document provided below may include headings, sections, or paragraphs. Your task is:
    1. Identify logical divisions in the document based on context (e.g., headings, topics, or major ideas).
    2. Provide a list of section titles or a brief summary of each section.
    3. If the document lacks clear headings, suggest appropriate ones.

    Document: {document[:4000]}  # Truncate to fit token limits
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    
    # Extract sections as a list from the response
    sections = response['choices'][0]['message']['content'].strip().split("\n")
    return sections
    
def worker_agent(chunk, previous_output):
    """
    Processes a single chunk of input and combines it with the previous output.
    """
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

def manager_agent(worker_outputs):
    """
    Combines outputs from all worker agents into a final coherent result.
    """
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

def chain_of_agents(long_text, chunk_size=100):
    """
    Implements the Chain-of-Agents framework for long-text summarization.
    """
    # Split the input text into chunks
    chunks = [long_text[i:i + chunk_size] for i in range(0, len(long_text), chunk_size)]
    
    previous_output = ""
    worker_outputs = []

    # Stage 1: Worker Agents
    for chunk in chunks:
        previous_output = worker_agent(chunk, previous_output)
        worker_outputs.append(previous_output)

    # Stage 2: Manager Agent
    final_output = manager_agent(worker_outputs)
    return final_output

# Example Usage
long_text = """
Insert a very long text here for testing the Chain-of-Agents framework.
It can be a multi-paragraph document or article.
"""
final_summary = chain_of_agents(long_text)
print("Final Summary:\n", final_summary)
