# Simple text chunking (splits text into overlapping chunks)

def simple_chunks(text: str, max_chars: int = 1500, overlap: int = 200):
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        yield text[i:j]
        i = j - overlap
        if i < 0: i = 0
