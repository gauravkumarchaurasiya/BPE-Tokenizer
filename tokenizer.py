import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=1256):
        self.vocab_size = vocab_size
        self.merges = {}  # Stores merged token pairs
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # Initialize base vocabulary

    def train(self, text):
        tokens = list(text.encode("utf-8"))
        ids = tokens.copy()
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[str(pair)] = idx  # Convert tuple keys to strings
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        self.final_ids = ids

    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(str(p), float("inf")))  # Convert to string
            if str(pair) not in self.merges:
                break
            idx = self.merges[str(pair)]
            tokens = self.merge(tokens, pair, idx)
        return " ".join(map(str, tokens))

    def decode(self, token_string):
        token_list = [int(t) for t in token_string.split() if t.strip().isdigit()]
        tokens = b"".join(self.vocab[idx] for idx in token_list)
        return tokens.decode("utf-8", errors="replace")
