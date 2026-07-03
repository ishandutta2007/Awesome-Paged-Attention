# The Contiguous Over-Allocation Era (Traditional LLM Serving)

In early serving infrastructures for Large Language Models (LLMs) before 2023, GPU memory allocation for the Key-Value (KV) cache was rigid and contiguous.

## Overview
Traditional systems reserved a contiguous memory region sized to the model's maximum possible sequence length (e.g., 2048 or 4096 tokens) for each request, regardless of the prompt's actual length.

```mermaid
block-beta
  columns 8
  t1["Token 1"] t2["Token 2"] t3["Token 3"] t4["Empty"] t5["Empty"] t6["Empty"] t7["Empty"] t8["Empty"]
  style t1 fill:#2ecc71,stroke:#27ae60
  style t2 fill:#2ecc71,stroke:#27ae60
  style t3 fill:#2ecc71,stroke:#27ae60
  style t4 fill:#e74c3c,stroke:#c0392b
  style t5 fill:#e74c3c,stroke:#c0392b
  style t6 fill:#e74c3c,stroke:#c0392b
  style t7 fill:#e74c3c,stroke:#c0392b
  style t8 fill:#e74c3c,stroke:#c0392b
```

## Key Challenges
* **Internal Fragmentation:** Unused allocated space within the pre-allocated block.
* **External Fragmentation:** Free memory blocks too small to satisfy new requests.
* **Low Batch Concurrency:** Massive VRAM waste restricted the number of parallel requests a GPU could handle.

---
[← Back to README](file:///C:/Users/ishan/Documents/Projects/Awesome-Paged-Attention/README.md)
