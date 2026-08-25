from smolagents import Tool
import json
from pathlib import Path
import yaml
from scripts.response_kb.retrieval_ex_kb import ResponseKBRetrievalTool
from typing import Dict, Any, Optional
from smolagents import Tool, LiteLLMModel, ToolCallingAgent
import os
class ResponseTool(Tool):
    """A smolagent Tool that checks whether a given response likely constitutes
    a jailbreak behavior by comparing it to known jailbreak outputs (internally).
    It returns a decision [YES/NO].
    """
    

    name = "jailbreak_response_check"
    description = (
        "Given a model response and the prompt id, determines whether it resembles jailbreak behavior "
        "Call the knowledge base retrieval to perform prompt detection, then integrates those results to produce and return the final detection outcome"
        "Returns '[YES/NO] --- explanation'."
    )

    inputs = {
        "response": {
            "type": "string",
            "description": "The model's response to a user prompt.",
            "nullable": True
        },
        "id": {
            "type": "integer",
            "description": "prompt id",
            "nullable": True
        }
    }

    output_type = "string"

    

    def __init__(self):
        super().__init__()

        tools = [
            ResponseKBRetrievalTool(),
        ]
        config = yaml.safe_load(open("scripts/prompts/output_agent.yaml", "r"))
        SYSTEM_PROMPT = config['system_prompt']['response']
        self.system_prompt = SYSTEM_PROMPT

        
        if str(Path(os.getenv("MODELID"))).startswith("gpt"):
            model = LiteLLMModel(model_id=str(Path(os.getenv("MODELID"))))
        else:
            model = LiteLLMModel(model_id=str(Path(os.getenv("MODELID"))), api_key="your-gemini-api-key-here")
        self.kb_agent = ToolCallingAgent(tools=tools, model=model, add_base_tools=False)

    def forward(self, id: Optional[int] = None, response: Optional[str] = None) -> str:
        full_prompt = self.system_prompt + response + "<<<PROMPT END>>>"
        verdict = self.kb_agent.run(full_prompt)
        try:
            from pathlib import Path
            import os, json
            base_dir = Path(os.getenv("OUTPUTDIR"))
            out_path = base_dir / "response_output.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"prompt_id": id, "response": response, "output_verdict": verdict}
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] write response_output.jsonl failed for id={id}: {e}")
        return verdict
 