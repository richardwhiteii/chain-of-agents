# Chain of Agents

Chain of Agents (CoA) is a Python implementation of a document processing system that uses multiple Large Language Model (LLM) agents to analyze and process long documents. The system is based on the paper "Chain of Agents: Large Language Models Collaborating on Long-Context Tasks".

## Features

- Dynamic document splitting based on semantic analysis
- Multi-agent processing pipeline with worker and manager agents
- OpenAI GPT integration with configurable models
- Asynchronous processing for better performance
- Rich CLI interface with progress tracking
- Comprehensive logging and error handling
- Interactive debugging shell
- Unit testing framework

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/chain-of-agents.git](https://github.com/richardwhiteii/chain-of-agents.git)
cd chain-of-agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Basic Command Line Usage

Process a document with default settings:
```bash
python chain_of_agents.py --document input.txt --goal "Summarize the key findings"
```

Advanced usage with all options:
```bash
python chain_of_agents.py \
    --document input.txt \
    --goal "Extract methodology and results" \
    --model gpt-4 \
    --chunk-size 4000 \
    --temperature 0.5 \
    --output results.json \
    --verbose
```

### Command Line Arguments

- `--document`: Path to the input document file (required)
- `--goal`: Processing goal for the agents (required)
- `--model`: LLM model to use (default: gpt-3.5-turbo)
- `--chunk-size`: Maximum chunk size in tokens (default: 8000)
- `--temperature`: LLM temperature (default: 0.3)
- `--output`: Path to save results JSON
- `--verbose`: Enable verbose output

### Python API Usage

```python
import asyncio
from chain_of_agents import ChainOfAgents

async def process_document():
    # Initialize the Chain of Agents
    chain = ChainOfAgents(
        model_name="gpt-3.5-turbo",
        chunk_size=8000,
        temperature=0.3
    )
    
    # Read your document
    with open("input.txt", "r") as f:
        document = f.read()
    
    # Process the document
    result = await chain.process_document(
        document=document,
        user_goal="Summarize the key findings"
    )
    
    # Access results
    print(result["manager_response"])
    
# Run the async function
asyncio.run(process_document())
```

### Interactive Shell

Launch the interactive debugging shell:
```bash
python -c "from chain_of_agents import run_interactive_shell; run_interactive_shell()"
```

## Architecture

### Components

1. **Document Chunking**
   - LLM-based semantic analysis
   - Token-aware splitting
   - Metadata preservation

2. **Worker Agents**
   - Sequential processing of document chunks
   - Information accumulation
   - Context-aware analysis

3. **Manager Agent**
   - Final synthesis of worker outputs
   - Goal-oriented response generation
   - Uncertainty handling

4. **Processing Pipeline**
   ```
   Document → Chunks → Worker Agents → Manager Agent → Final Response
   ```

## Development

### Running Tests

```bash
python -c "from chain_of_agents import run_tests; run_tests()"
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Requirements

- openai>=0.27.0
- tiktoken>=0.3.0
- tenacity>=8.0.1
- rich>=10.0.0
- asyncio>=3.4.3

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

This code was inspired by the citation below:
```bibtex
@article{zhang2024chain,
  title={Chain of Agents: Large Language Models Collaborating on Long-Context Tasks},
  author={Zhang, Yusen and Sun, Ruoxi and Chen, Yanfei and Pfister, Tomas and Zhang, Rui and Arik, Sercan Ö.},
  journal={arXiv preprint arXiv:2406.02818},
  year={2024}
}
```

## Acknowledgments

- Based on the paper "Chain of Agents" by Zhang et al.
