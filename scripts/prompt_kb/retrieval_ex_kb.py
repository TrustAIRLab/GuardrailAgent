import json
from typing import Optional

from smolagents import Tool
from embedchain import App
from filelock import FileLock
import os

from pathlib import Path
_GLOBAL_APP = None
class PromptKBRetrievalTool(Tool):

    name = "prompt_KB_retrieval"
    description = (
        '''
        Given a prompt, check its similarity against a local jailbreak prompt knowledge base.
        Instead of returning raw retrieved texts or similarity scores, this tool returns a decision-only message:
        - If the mean similarity of the top-3 retrieved items > 0.70 → classified as definite jailbreak prompt (score=2).
        - If the mean similarity < 0.68 → classified as not similar (score=0).
        - Otherwise (0.68–0.70) → classified as likely jailbreak (score=1).
        '''
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "prompt to be detected",
            "nullable": True,
        },
        "top_k": {
            "type": "integer",
            "description": "return top-k retrieval result",
            "default": 3,
            "nullable": True,
        },
        "prompt_id": {
            "type": "integer",
            "description": "prompt id",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(
        self,
        kb_dir: str = str(Path(os.getenv("INPUTRETRIEVALDIR"))),
        model_name: str = "intfloat/e5-base-v2",
    ):
        super().__init__()

        self.tau = 0.5
        global _GLOBAL_APP
        model_path = './hf-cache/models/e5-base-v2'
        if _GLOBAL_APP is None:
            cfg = {
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": model_path},
                },
                "vectordb": {
                    "provider": "chroma",
                    "config": {"collection_name": "jailbreak_prompts", "dir": kb_dir},
                },
            }

            lock = FileLock("chroma_init.lock")
            with lock:
                _GLOBAL_APP = App.from_config(config=cfg)

        self.app = _GLOBAL_APP


    @staticmethod
    def _to_similarity(distance: float) -> float:
        return 1 - distance    # ∈ (0, 1)

    def forward(self, prompt_id: Optional[int] = None, prompt: Optional[str] = None, top_k: Optional[int] = 5) -> str:
        _, sources = self.app.query(
            prompt,
            citations=True,
            dry_run=True,
            number_documents=top_k
        )

        results = []
        for chunk, meta in sources:
            dist = meta.get("distance") or meta.get("score", 0)
            score = self._to_similarity(dist)

            results.append({
                "text": chunk,
                "jailbreak_type": meta.get("jailbreak_type", ""),
                "score": score,
            })
        try:
            base_dir = (
                Path(os.getenv("OUTPUTDIR"))
            )
            out_path = base_dir / "retrieval_output.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            record = {
                "prompt_id": prompt_id,
                "retrieval_result": results,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] write retrieval_output.jsonl failed for id={prompt_id}: {e}")


        try:
            top3 = [float(r.get("score", 0.0)) for r in results[:3] if r.get("score") is not None]
            mean3 = sum(top3) / len(top3) if top3 else 0.0
        except Exception:
            try:
                top2 = [float(r.get("score", 0.0)) for r in results[:2] if r.get("score") is not None]
                mean3 = sum(top2) / len(top2) if top2 else 0.0
            except Exception:
                try:
                    top1 = [float(r.get("score", 0.0)) for r in results[:1] if r.get("score") is not None]
                    means3 = top1
                except Exception:
                    mean3 = 0.0

        HI, LO = 0.70, 0.68
        if mean3 > HI:
            msg = (
                "The retrieval indicates this prompt is 100% a jailbreak prompt and is highly similar to "
                "templates in the knowledge base. I assign this prompt a jailbreak score of 2."
            )
        elif mean3 < LO:
            msg = (
                "The retrieval indicates this prompt does not show high similarity to any jailbreak prompt "
                "in the knowledge base. Please apply subsequent methods for careful evaluation. "
                "I assign this prompt a jailbreak score of 0."
            )
        else:
            msg = (
                "The retrieval indicates this prompt is similar to some jailbreak prompts in the knowledge base "
                "and is likely a jailbreak prompt. Please use subsequent methods to further confirm. "
                "I assign this prompt a jailbreak score of 1."
            )

        return msg