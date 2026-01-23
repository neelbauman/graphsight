import json
from mermaid_to_json import mermaid_to_graph

with open("samples/sample-7.mmd", "r", encoding='utf-8') as f:
    mermaid = f.read()

result = mermaid_to_graph(mermaid)

with open("samples/sample-7.json", "w", encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
