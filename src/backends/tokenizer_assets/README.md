# BGE embedding tokenizer

`bge-large-en-v1.5.json` is the unmodified official `tokenizer.json` from
`BAAI/bge-large-en-v1.5`, revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`.

Source: https://huggingface.co/BAAI/bge-large-en-v1.5/blob/d4aa6901d3a41ba39fb536a557fa166f842b0e09/tokenizer.json

SHA-256: `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66`

The model card declares MIT; the upstream FlagEmbedding license is included.
The original file was already available locally and is copied byte-for-byte.
Only the exact model name, BAAI-qualified name, and deployment alias
`text-embedding-bge-large-en-v1.5` are supported. Runtime downloads, padding,
and silent tokenizer truncation are disabled. Counts include CLS and SEP.
Model input limit: 512 tokens, leaving at most 510 for the serialized text.

Mem0 batches count its serialized role/content text after newline replacement.
Splits use original character offsets and recount each candidate, preserving
case, Unicode and whitespace. They never decode WordPiece IDs to reconstruct text.
