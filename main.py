import os
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
import asyncio
from dataclasses import dataclass
import logging
import json
from datetime import datetime
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Agent:
    """Base class for agents in the system"""
    id: str
    type: str
    context: str = ""
    communication_unit: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.metadata['created_at'] = datetime.now().isoformat()

class DocumentChunk:
    """Represents a chunk of the document with metadata"""
    def __init__(self, content: str, start_idx: int, end_idx: int, metadata: Dict[str, Any] = None):
        self.content = content
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.metadata = metadata or {}
        self.token_count = 0
        
    def __str__(self):
        return f"Chunk({self.start_idx}:{self.end_idx})[{self.token_count} tokens]: {self.content[:50]}..."

class ChainOfAgents:
    """Main class implementing the Chain of Agents system"""
    
    def __init__(self, 
                 api_key: str = None,
                 model_name: str = "gpt-3.5-turbo",
                 chunk_size: int = 8000,
                 temperature: float = 0.3,
                 max_retries: int = 3):
        """
        Initialize the Chain of Agents system.
        
        Args:
            api_key: OpenAI API key (can also be set via OPENAI_API_KEY env variable)
            model_name: Name of the LLM model to use
            chunk_size: Maximum chunk size in tokens
            temperature: Temperature for LLM responses (0.0 to 1.0)
            max_retries: Maximum number of retries for API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided")
            
        openai.api_key = self.api_key
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.max_retries = max_retries
        
        self.worker_agents: List[Agent] = []
        self.manager_agent: Optional[Agent] = None
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        
        # Session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_metadata = {
            "model_name": model_name,
            "chunk_size": chunk_size,
            "temperature": temperature,
            "start_time": datetime.now().isoformat()
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_llm(self, prompt: str, system_message: str = None) -> str:
        """Make an API call to the LLM"""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise

    async def process_document(self, document: str, user_goal: str) -> Dict[str, Any]:
        """Main method to process a document using the Chain of Agents approach"""
        try:
            processing_start = datetime.now()
            
            # Split document into chunks
            document_chunks = await self.split_document(document)
            logger.info(f"Split document into {len(document_chunks)} chunks")
            
            # Create agents
            await self.create_agents(len(document_chunks))
            logger.info(f"Created {len(self.worker_agents)} worker agents and 1 manager agent")
            
            # Execute workflow
            result = await self.execute_workflow(document_chunks, user_goal)
            
            # Add processing metadata
            processing_end = datetime.now()
            processing_duration = (processing_end - processing_start).total_seconds()
            
            result["metadata"] = {
                **self.session_metadata,
                "processing_duration": processing_duration,
                "num_chunks": len(document_chunks),
                "total_tokens": sum(chunk.token_count for chunk in document_chunks)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    async def split_document(self, document: str) -> List[DocumentChunk]:
        """Split document into logical chunks using LLM analysis"""
        
        # First, get a structural analysis from the LLM
        analysis_prompt = f"""
        Analyze the following document and identify logical break points based on:
        1. Section headers
        2. Topic changes
        3. Natural paragraph groupings
        4. Semantic coherence
        
        Document: {document[:2000]}... [truncated]
        
        Return your analysis as JSON with the following structure:
        {{
            "break_points": [
                {{"position": "index", "reason": "explanation"}}
            ],
            "document_type": "detected document type",
            "structure_notes": "any notable structural patterns"
        }}
        """
        
        analysis_response = await self._call_llm(
            analysis_prompt,
            system_message="You are an expert document analyzer. Provide detailed structural analysis."
        )
        
        try:
            analysis = json.loads(analysis_response)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM analysis response as JSON. Using fallback splitting.")
            analysis = {"break_points": []}
        
        # Combine LLM analysis with token-based splitting
        chunks = await self._create_chunks(document, analysis)
        
        # Count tokens for each chunk
        for chunk in chunks:
            chunk.token_count = len(self.tokenizer.encode(chunk.content))
            
        return chunks

    async def _create_chunks(self, document: str, analysis: Dict[str, Any]) -> List[DocumentChunk]:
        """Create document chunks based on LLM analysis and token limits"""
        chunks = []
        current_position = 0
        
        # Get suggested break points from analysis
        suggested_breaks = sorted([
            bp["position"] for bp in analysis.get("break_points", [])
            if isinstance(bp.get("position"), int)
        ])
        
        # Combine with token-based splitting
        doc_tokens = self.tokenizer.encode(document)
        
        while current_position < len(document):
            # Find next break point
            next_position = len(document)
            
            # Check suggested breaks
            for break_point in suggested_breaks:
                if break_point > current_position and break_point - current_position <= self.chunk_size:
                    next_position = break_point
                    break
            
            # If no suitable break point, use token limit
            if next_position == len(document):
                tokens_remaining = doc_tokens[current_position:]
                if len(tokens_remaining) > self.chunk_size:
                    next_position = current_position + self._find_next_sentence_end(
                        document[current_position:current_position + self.chunk_size * 4]
                    )
            
            chunk_content = document[current_position:next_position]
            chunks.append(DocumentChunk(
                content=chunk_content,
                start_idx=current_position,
                end_idx=next_position,
                metadata={
                    "document_type": analysis.get("document_type", "unknown"),
                    "structure_notes": analysis.get("structure_notes", "")
                }
            ))
            
            current_position = next_position
            
        return chunks

    def _find_next_sentence_end(self, text: str) -> int:
        """Find the next sentence ending within the text"""
        sentence_endings = ['. ', '! ', '? ', '\n\n']
        min_position = float('inf')
        
        for ending in sentence_endings:
            pos = text.find(ending)
            if pos != -1 and pos < min_position:
                min_position = pos + len(ending)
                
        return min_position if min_position != float('inf') else len(text)

    async def create_agents(self, num_chunks: int):
        """Create worker and manager agents"""
        
        # Create worker agents
        self.worker_agents = [
            Agent(
                id=f"worker-{i}",
                type="worker",
                metadata={
                    "session_id": self.session_id,
                    "sequence": i
                }
            )
            for i in range(num_chunks)
        ]
        
        # Create manager agent
        self.manager_agent = Agent(
            id="manager",
            type="manager",
            metadata={
                "session_id": self.session_id
            }
        )

    def generate_worker_prompt(self, 
                             chunk: DocumentChunk, 
                             previous_cu: str, 
                             user_goal: str, 
                             index: int) -> Tuple[str, str]:
        """Generate system message and prompt for worker agent"""
        
        system_message = f"""
        You are Worker Agent {index + 1} in a Chain-of-Agents system.
        Your role is to:
        1. Analyze your assigned text chunk
        2. Integrate information with previous agent findings
        3. Extract information relevant to the user's goal
        4. Communicate findings clearly to the next agent
        
        Be concise but thorough. Focus on information relevant to the goal.
        """
        
        prompt = f"""
        GOAL: {user_goal}
        
        PREVIOUS FINDINGS:
        {previous_cu or 'No previous findings available.'}
        
        CURRENT TEXT CHUNK:
        {chunk.content}
        
        Based on the above:
        1. What are the key points from this chunk relevant to the goal?
        2. How does this information relate to or extend previous findings?
        3. What critical information should be passed to the next agent?
        
        Provide your response in a clear, structured format.
        """
        
        return system_message, prompt

    def generate_manager_prompt(self, final_cu: str, user_goal: str) -> Tuple[str, str]:
        """Generate system message and prompt for manager agent"""
        
        system_message = """
        You are the Manager Agent in a Chain-of-Agents system.
        Your role is to:
        1. Synthesize information from all worker agents
        2. Ensure comprehensive coverage of the user's goal
        3. Provide a clear, well-structured final response
        4. Highlight any uncertainties or areas needing clarification
        """
        
        prompt = f"""
        GOAL: {user_goal}
        
        ACCUMULATED FINDINGS FROM WORKER AGENTS:
        {final_cu}
        
        Please provide a comprehensive response that:
        1. Directly addresses the user's goal
        2. Synthesizes all relevant information from workers
        3. Is well-structured and easy to understand
        4. Notes any important caveats or limitations
        
        Format your response clearly and logically.
        """
        
        return system_message, prompt

    async def execute_workflow(self, chunks: List[DocumentChunk], user_goal: str) -> Dict[str, Any]:
        """Execute the Chain of Agents workflow"""
        
        current_cu = ""
        workflow_start = datetime.now()
        
        # Sequential processing by worker agents
        for i, chunk in enumerate(chunks):
            worker_start = datetime.now()
            
            system_message, worker_prompt = self.generate_worker_prompt(
                chunk=chunk,
                previous_cu=current_cu,
                user_goal=user_goal,
                index=i
            )
            
            worker_response = await self._call_llm(worker_prompt, system_message)
            current_cu = worker_response
            
            self.worker_agents[i].communication_unit = worker_response
            self.worker_agents[i].metadata['processing_time'] = \
                (datetime.now() - worker_start).total_seconds()
                
            logger.info(f"Worker {i+1} completed processing in "
                       f"{self.worker_agents[i].metadata['processing_time']:.2f}s")
            
        # Manager agent processing
        manager_start = datetime.now()
        system_message, manager_prompt = self.generate_manager_prompt(current_cu, user_goal)
        final_response = await self._call_llm(manager_prompt, system_message)
        
        if self.manager_agent:
            self.manager_agent.communication_unit = final_response
            self.manager_agent.metadata['processing_time'] = \
                (datetime.now() - manager_start).total_seconds()
        
        workflow_duration = (datetime.now() - workflow_start).total_seconds()
        
        return {
            "worker_responses": [
                {
                    "agent_id": w.id,
                    "response": w.communication_unit,
                    "metadata": w.metadata
                }
                for w in self.worker_agents
            ],
            "manager_response": final_response,
            "workflow_duration": workflow_duration
        }

# CLI Interface
async def run_cli():
    import argparse
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    console = Console()
    
    parser = argparse.ArgumentParser(description='Chain of Agents Document Processor')
    parser.add_argument('--document', type=str, required=True, help='Path to document file')
    parser.add_argument('--goal', type=str, required=True, help='Processing goal')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='LLM model to use')
    parser.add_argument('--chunk-size', type=int, default=8000, help='Maximum chunk size in tokens')
    parser.add_argument('--temperature', type=float, default=0.3, help='LLM temperature')
    parser.add_argument('--output', type=str, help='Path to save results JSON')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Set up logging based on verbose flag
        logging_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=logging_level)
        
        # Read document
        console.print(Panel(f"Reading document: {args.document}", title="Input"))
        try:
            with open(args.document, 'r', encoding='utf-8') as f:
                document = f.read()
        except Exception as e:
            console.print(f"[red]Error reading document: {str(e)}[/red]")
            return
        
        # Initialize Chain of Agents
        chain = ChainOfAgents(
            model_name=args.model,
            chunk_size=args.chunk_size,
            temperature=args.temperature
        )
        
        # Process document with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing document...", total=None)
            
            result = await chain.process_document(document, args.goal)
            progress.update(task, completed=True)
        
        # Display results
        console.print("\n[bold green]Processing Complete![/bold green]\n")
        
        console.print(Panel("Worker Agent Results", title="Results"))
        for worker_result in result["worker_responses"]:
            console.print(Panel(
                worker_result["response"],
                title=f"Worker {worker_result['agent_id']}",
                border_style="blue"
            ))
        
        console.print(Panel(
            result["manager_response"],
            title="Manager Response",
            border_style="green"
        ))
        
        # Display metadata
        if args.verbose:
            console.print(Panel(
                f"""
                Total Processing Time: {result['workflow_duration']:.2f}s
                Number of Chunks: {len(result['worker_responses'])}
                Model: {args.model}
                Temperature: {args.temperature}
                """,
                title="Processing Metadata",
                border_style="yellow"
            ))
        
        # Save results if output path provided
        if args.output:
            try:
                output_path = args.output
                if not output_path.endswith('.json'):
                    output_path += '.json'
                    
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                console.print(f"\n[green]Results saved to: {output_path}[/green]")
            except Exception as e:
                console.print(f"[red]Error saving results: {str(e)}[/red]")
    
    except Exception as e:
        console.print(f"[red]Error during processing: {str(e)}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())

def main():
    """Main entry point for the CLI application"""
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example usage:
"""
# Basic usage
python chain_of_agents.py --document input.txt --goal "Summarize the key findings"

# Advanced usage with all options
python chain_of_agents.py \
    --document input.txt \
    --goal "Extract methodology and results" \
    --model gpt-4 \
    --chunk-size 4000 \
    --temperature 0.5 \
    --output results.json \
    --verbose
"""

# Interactive Shell for debugging and development
def run_interactive_shell():
    """Run an interactive shell for testing and debugging"""
    import code
    import readline
    import rlcompleter
    
    # Create a Chain of Agents instance
    chain = ChainOfAgents()
    
    # Set up readline with tab completion
    readline.parse_and_bind("tab: complete")
    
    # Create interactive shell variables
    variables = {
        'chain': chain,
        'Agent': Agent,
        'DocumentChunk': DocumentChunk,
        'process_document': chain.process_document,
        'split_document': chain.split_document,
    }
    
    banner = """
    Chain of Agents Interactive Shell
    --------------------------------
    Available objects:
    - chain: ChainOfAgents instance
    - Agent: Agent class
    - DocumentChunk: DocumentChunk class
    - process_document: Method to process documents
    - split_document: Method to split documents
    
    Example:
    >>> document = "Your document text here..."
    >>> goal = "Your processing goal here..."
    >>> result = await chain.process_document(document, goal)
    """
    
    # Start interactive shell
    code.InteractiveConsole(variables).interact(banner=banner)

# Development tools
def run_tests():
    """Run unit tests for the Chain of Agents system"""
    import unittest
    import sys
    
    class TestChainOfAgents(unittest.TestCase):
        def setUp(self):
            self.chain = ChainOfAgents()
        
        async def test_document_splitting(self):
            document = "Test document with multiple sentences. This is another sentence."
            chunks = await self.chain.split_document(document)
            self.assertTrue(len(chunks) > 0)
        
        async def test_agent_creation(self):
            await self.chain.create_agents(3)
            self.assertEqual(len(self.chain.worker_agents), 3)
            self.assertIsNotNone(self.chain.manager_agent)
        
        # Add more tests as needed
    
    # Run tests
    unittest.main(argv=[sys.argv[0]])

if __name__ == "__main__":
    main()