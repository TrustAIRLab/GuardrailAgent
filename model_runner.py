import os
from pathlib import Path
from scripts.output_agent import ResponseTool
from scripts.input_agent import KnowledgeBaseVerdictTool


def run_model_jd_agent(id, prompt, response):
    inorout = str(Path(os.getenv("MODULE")))
    if inorout == "INPUT":
        input_agent = KnowledgeBaseVerdictTool()
        input_agent.forward(id, prompt)
    else:
        output_agent = ResponseTool()
        output_agent.forward(id, response)
