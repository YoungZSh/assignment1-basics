from __future__ import annotations

import os
import heapq
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
import regex as re
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PATTERN = re.compile(PAT)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_each_chunk(chunk: str, special_tokens: list[str]) -> list[list[int]]:
    # 先删除所有 special tokens
    if special_tokens:
        # 转义特殊字符并组合成正则表达式
        escaped_tokens = [re.escape(token) for token in special_tokens]
        pattern_special = re.compile("|".join(escaped_tokens))
        chunk = pattern_special.sub("", chunk)

    docs: list[list[int]] = []

    for match in PATTERN.finditer(chunk):
        token = match.group()
        if not token:
            continue

        token_bytes = token.encode("utf-8")
        ids = [b for b in token_bytes]
        if ids:
            docs.append(ids)

    return docs


# 包装类处理频率与 tie-break 情况
class RevLexKey:
    """
    让频率高的 pair 在 heapq 里变成“更小”，若频率一致，(ba, bb)
    字典序更大的也会被认为“更小”，这样就能直接比较，而不必额外取负。
    """

    __slots__ = ("freq", "ba", "bb")

    def __init__(self, freq: int, ba: bytes, bb: bytes):
        self.freq = freq
        self.ba = ba
        self.bb = bb

    def __lt__(self, other: RevLexKey) -> bool:
        if self.freq != other.freq:
            return self.freq > other.freq
        # 注意这里用的是 > ，保证字典序大的 pair 先弹出
        return (self.ba, self.bb) > (other.ba, other.bb)


class TokenNode:
    """简单双向链表节点，支持增量更新。"""

    __slots__ = ("token", "prev", "next", "alive")

    def __init__(self, token: int):
        self.token = token
        self.prev: TokenNode | None = None
        self.next: TokenNode | None = None
        self.alive = True


def make_heap_item(pair: tuple[int, int], freq: int, vocab: dict[int, bytes]):
    a, b = pair
    ba, bb = vocab[a], vocab[b]
    # RevLexKey 同时封装频率和字典序，准确定义 heap 中的优先级
    return (RevLexKey(freq, ba, bb), pair)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 初始化词表
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    print("[INFO] Pretokenizer begins.")

    chunks: list[str] = []
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, 512, b"<|endoftext|>")
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

    all_docs: list[list[int]] = []
    with Pool(processes=cpu_count()) as pool:
        para_lst = [(chunk, special_tokens) for chunk in chunks]
        results = pool.starmap(process_each_chunk, para_lst)

    for docs in results:
        all_docs.extend(docs)

    print("[INFO] Pretokenizer ends.")
    print("[INFO] BPE training begins.")

    # 开始训练
    next_vocab_id = 256
    num_merges = vocab_size - 256 - len(special_tokens)

    # 将文档转换为节点序列，配合 pair occurrences 做局部增量更新
    doc_nodes: list[list[TokenNode]] = []
    for doc in all_docs:
        if not doc:
            doc_nodes.append([])
            continue
        nodes = [TokenNode(token) for token in doc]
        for idx in range(len(nodes) - 1):
            nodes[idx].next = nodes[idx + 1]
            nodes[idx + 1].prev = nodes[idx]
        doc_nodes.append(nodes)

    pair_freq: dict[tuple[int, int], int] = {}
    pair_occurrences: dict[tuple[int, int], set[TokenNode]] = {}

    def add_pair_from_node(node: TokenNode, track: set[tuple[int, int]] | None = None) -> None:
        if node is None or not node.alive:
            return
        nxt = node.next
        if nxt is None or not nxt.alive:
            return
        pair = (node.token, nxt.token)
        pair_freq[pair] = pair_freq.get(pair, 0) + 1
        occurrence_set = pair_occurrences.setdefault(pair, set())
        occurrence_set.add(node)
        if track is not None:
            track.add(pair)

    def remove_pair_from_node(node: TokenNode, track: set[tuple[int, int]] | None = None) -> None:
        if node is None or not node.alive:
            return
        nxt = node.next
        if nxt is None or not nxt.alive:
            return
        pair = (node.token, nxt.token)
        occurrence_set = pair_occurrences.get(pair)
        if occurrence_set is None or node not in occurrence_set:
            return
        occurrence_set.remove(node)
        if not occurrence_set:
            pair_occurrences.pop(pair, None)
        current_freq = pair_freq.get(pair)
        if current_freq is not None:
            if current_freq <= 1:
                pair_freq.pop(pair, None)
            else:
                pair_freq[pair] = current_freq - 1
        if track is not None:
            track.add(pair)

    for nodes in doc_nodes:
        for node in nodes[:-1]:
            add_pair_from_node(node)

    heap = [make_heap_item(p, freq, vocab) for p, freq in pair_freq.items()]
    heapq.heapify(heap)

    progress_bar = tqdm(range(num_merges), desc="BPE merges", unit="merge")
    for _ in progress_bar:
        # heap 懒更新策略
        while heap:
            key, best_pair = heapq.heappop(heap)
            current_freq = pair_freq.get(best_pair)
            if current_freq and key.freq == current_freq:
                break
        else:
            break

        occurrence_nodes = list(pair_occurrences.get(best_pair, ()))
        if not occurrence_nodes:
            pair_freq.pop(best_pair, None)
            continue

        changed_pairs: set[tuple[int, int]] = set()
        merged_any = False
        new_token_id = next_vocab_id

        for left_node in occurrence_nodes:
            if not left_node.alive:
                continue
            right_node = left_node.next
            if right_node is None or not right_node.alive:
                continue
            if left_node.token != best_pair[0] or right_node.token != best_pair[1]:
                continue

            prev_node = left_node.prev
            next_node = right_node.next

            remove_pair_from_node(left_node, track=changed_pairs)
            if prev_node is not None:
                remove_pair_from_node(prev_node, track=changed_pairs)
            if right_node is not None:
                remove_pair_from_node(right_node, track=changed_pairs)

            left_node.token = new_token_id
            left_node.next = next_node
            if next_node is not None:
                next_node.prev = left_node

            right_node.alive = False
            right_node.prev = None
            right_node.next = None

            if prev_node is not None:
                add_pair_from_node(prev_node, track=changed_pairs)
            if left_node.next is not None:
                add_pair_from_node(left_node, track=changed_pairs)

            merged_any = True

        pair_occurrences.pop(best_pair, None)
        pair_freq.pop(best_pair, None)

        if not merged_any:
            progress_bar.set_postfix({"pair": "skip", "freq": 0, "vocab": next_vocab_id})
            continue

        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        next_vocab_id += 1

        # 必须在所有受影响的区域处理完后再进行入堆
        for pair in changed_pairs:
            freq = pair_freq.get(pair)
            if freq:
                heapq.heappush(heap, make_heap_item(pair, freq, vocab))

        progress_bar.set_postfix({"pair": f"{best_pair[0]}+{best_pair[1]}", "freq": key.freq, "vocab": next_vocab_id})

    progress_bar.close()

    # 添加 special tokens 到词汇表
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_vocab_id] = token_bytes
            next_vocab_id += 1

    print("[INFO] BPE training ends.")
    return vocab, merges
