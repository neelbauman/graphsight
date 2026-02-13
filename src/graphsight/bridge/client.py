import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

class NodeMermaidBridge:
    """Node.js経由でMermaid公式パーサーを利用するブリッジ"""
    
    SCRIPT_PATH = Path(__file__).parent / "mermaid_parser.mjs"

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("node") is not None and cls.SCRIPT_PATH.exists()

    @classmethod
    def parse(cls, mermaid_code: str) -> Dict[str, Any]:
        """
        Mermaidコードをパースして辞書形式のグラフ構造を返す。
        
        Returns:
            {
                "direction": "TD",
                "nodes": [{"id": "A", "label": "Text", "shape": "rect"}, ...],
                "edges": [{"src": "A", "dst": "B", "label": "Yes", "style": "-->"}, ...]
            }
        """
        if not cls.is_available():
            raise RuntimeError("Node.js is not installed or bridge script is missing.")

        try:
            # node プロセスを起動
            process = subprocess.Popen(
                ["node", str(cls.SCRIPT_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cls.SCRIPT_PATH.parent) # package.jsonがある場所などを考慮
            )
            
            stdout, stderr = process.communicate(input=mermaid_code)
            
            if process.returncode != 0:
                logger.error(f"Mermaid Parser Error: {stderr}")
                raise ValueError(f"Failed to parse Mermaid code: {stderr.strip()}")
            
            if not stdout.strip():
                raise ValueError("Mermaid Parser returned empty output.")

            return json.loads(stdout)

        except json.JSONDecodeError as e:
            logger.error(f"Bridge JSON Decode Error. Output was: {stdout}")
            raise e
        except Exception as e:
            logger.error(f"Bridge Execution Error: {e}")
            raise e

