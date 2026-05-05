import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.llama_inference import LlamaResponder, ResponderConfig
from src.llm.rag.retriever import RAGRetriever

# Test RAG retrieval
retriever = RAGRetriever()
docs = retriever.query("CreateUser AttachUserPolicy privilege escalation", top_k=4)
print(f"Retrieved {len(docs)} docs")
for i, d in enumerate(docs): print(f"  [{i+1}] {d[:80]}...")

# Test LLM inference (real mode, expect 3-10 tok/s on your laptop)
config = ResponderConfig(mock_mode=False, backend="ollama", prompt_version="v2")
responder = LlamaResponder(config)
report = responder.generate_report(
    anomalous_logs=["CreateUser <*>", "AttachUserPolicy <*>"],
    context={"session_id": 1, "total_events": 50, "anomaly_ratio": 0.04,
             "event_names": ["CreateUser", "AttachUserPolicy"],
             "timestamps": ["2024-01-01T10:00:00Z", "2024-01-01T10:00:05Z"],
             "source_ips": ["1.2.3.4", "1.2.3.4"],
             "user_arns": ["arn:aws:iam::123456789012:user/attacker", 
                          "arn:aws:iam::123456789012:user/attacker"]},
    rag_context=docs,
)
print(report)