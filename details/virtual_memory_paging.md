# The Virtual Memory Paging Revolution (PagedAttention / vLLM)

PagedAttention adapts virtual memory paging concepts from traditional operating systems to GPU VRAM for storing the Key-Value (KV) cache.

## Overview
Instead of allocating contiguous blocks, PagedAttention partitions the KV cache of a sequence into small, non-contiguous physical blocks.

```mermaid
flowchart LR
    subgraph Logical Space
        L0["Logical Block 0 (Tokens 0-15)"]
        L1["Logical Block 1 (Tokens 16-31)"]
    end
    subgraph Block Table
        T0["L0 -> Physical Block 102"]
        T1["L1 -> Physical Block 45"]
    end
    subgraph Physical VRAM
        P45["Physical Block 45"]
        P102["Physical Block 102"]
    end
    L0 --> T0 --> P102
    L1 --> T1 --> P45
```

## Benefits
* **Zero Fragmentation:** Recovers ~96% of wasted memory.
* **Higher Concurrency:** Enables 4x to 5x higher throughput compared to traditional systems.
* **Flexible Sharing:** Easy caching of shared prompts.

---
[← Back to README](file:///C:/Users/ishan/Documents/Projects/Awesome-Paged-Attention/README.md)
