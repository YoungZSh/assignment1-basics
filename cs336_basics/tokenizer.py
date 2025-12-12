from collections.abc import Iterable, Iterator
import json
import regex as re
import heapq


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # ---- 1. id_to_token ----
        self.id_to_token: dict[int, bytes] = dict(sorted(vocab.items()))
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens or []

        # ---- 2. 在构建反向查找之前添加特殊标记 ----
        if self.special_tokens:
            next_vocab_id = max(self.id_to_token) + 1 if self.id_to_token else 0
            existing_tokens = set(self.id_to_token.values())  # 字节集合

            for token in self.special_tokens:
                tok_bytes = token.encode("utf-8")
                if tok_bytes in existing_tokens:
                    continue
                self.id_to_token[next_vocab_id] = tok_bytes
                existing_tokens.add(tok_bytes)
                next_vocab_id += 1

        # ---- 3. 构建 token_to_id ----
        self.token_to_id: dict[bytes, int] = {
            tok: idx for idx, tok in self.id_to_token.items()
        }

        # ---- 4. 构建 merges_rank ----
        self.merges_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._pretoken_pattern = re.compile(PAT)

    # =====================================================
    # 工厂方法：从 vocab.json + merges.txt 加载
    # =====================================================
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        # ---- 加载 vocab.json ----
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw: dict[str, int] = json.load(f)

        vocab: dict[int, bytes] = {
            token_id: token.encode("utf-8")
            for token, token_id in vocab_raw.items()
        }

        # ---- 加载 merges.txt ----
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid merge line: {line}")
                a, b = parts
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        # ---- 让 __init__ 处理其余部分 ----
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _bpe_encode_bytes(self, word: bytes) -> list[int]:
        """
        对表示为 UTF-8 字节的单个预标记应用 BPE 合并。
        初始符号是单个字节；重复合并具有最小合并阶（最早学习的）的相邻对。
        """
        if not word:
            return []

        # ---- 初始符号：单个字节 ----
        syms: list[bytes] = [bytes([b]) for b in word]
        n = len(syms)
        if n == 1:
            tid = self.token_to_id.get(syms[0])
            if tid is None:
                # 如果词汇表一致，单个字节应该总是存在
                raise ValueError(
                    f"Byte {syms[0]!r} not found in vocabulary. "
                    f"This indicates an incomplete vocabulary."
                )
            return [tid]

        # ---- 索引 0..n-1 上的链表 ----
        nxt = list(range(1, n)) + [-1]
        prev = [-1] * n
        # 根据nxt初始化prev数组
        for i in range(n):
            if nxt[i] != -1:
                prev[nxt[i]] = i
        alive = [True] * n

        # 版本戳，用于在合并后使过时的堆条目失效，其中的数组代表每个sym被修改过多少次
        ver = [0] * n

        heap: list[tuple[int, int, int, int, int]] = []
        # 堆项：(rank, left_i, right_i, ver_left, ver_right)

        def push_pair(i: int) -> None:
            j = nxt[i]
            if j == -1:
                return
            pair = (syms[i], syms[j])
            rank = self.merges_rank.get(pair)
            if rank is None:
                return
            heapq.heappush(heap, (rank, i, j, ver[i], ver[j]))

        # 用所有可合并的相邻对初始化堆
        for i in range(n - 1):
            push_pair(i)

        def is_valid(rank: int, i: int, j: int, vi: int, vj: int) -> bool:
            if i < 0 or j < 0:
                return False
            if not alive[i] or not alive[j]:
                return False
            if nxt[i] != j:
                return False
            if ver[i] != vi or ver[j] != vj: # 这里必须有这个判断
                return False
            return self.merges_rank.get((syms[i], syms[j])) == rank

        # ---- 合并循环 ----
        while heap:
            rank, i, j, vi, vj = heapq.heappop(heap)
            if not is_valid(rank, i, j, vi, vj):
                continue

            # 将 i + j 合并到 i
            syms[i] = syms[i] + syms[j]
            ver[i] += 1

            # 移除 j
            alive[j] = False
            nj = nxt[j]
            nxt[i] = nj
            if nj != -1:
                prev[nj] = i

            # 更新邻居：(prev[i], i) 和 (i, nxt[i])
            pi = prev[i]
            if pi != -1:
                push_pair(pi)
            push_pair(i)

        # ---- 按顺序收集最终符号 ----
        # 找到头节点（第一个存活的）
        head = 0
        while head < n and not alive[head]:
            head += 1
        if head >= n:
            return []

        out: list[int] = []
        cur = head
        while cur != -1:
            tok = syms[cur]
            tid = self.token_to_id.get(tok)
            if tid is None:
                # 如果词汇表一致，不应该发生这种情况
                raise ValueError(
                    f"Token {tok!r} not found in vocabulary. "
                    f"This indicates an inconsistency between the vocabulary and BPE merges."
                )
            out.append(tid)
            cur = nxt[cur]

        return out

    def encode(self, text: str) -> list[int]:
        """
        非流式接口，对于大文本采用分块处理以节省内存。
        将文本分成固定大小的块，使用encode_iterable的流式处理机制。
        这样可以避免一次性处理整个大文本，减少内存占用。
        """
        if not text:
            return []

        # 对于小文本，直接使用encode_iterable处理（避免重复代码）
        # 对于大文本，分块处理以节省内存
        # 使用512KB作为块大小（按字节计算）
        CHUNK_SIZE_BYTES = 512 * 1024  # 512KB
        
        # 为了避免先编码整个文本来检查大小，我们使用字符数作为粗略估计
        # 保守估计：假设平均每个字符最多4字节（考虑emoji等），所以使用CHUNK_SIZE_BYTES // 4
        # 对于纯ASCII文本，这个估计会偏保守，但更安全
        CHUNK_SIZE_CHARS = CHUNK_SIZE_BYTES // 4  # 约128K字符，对应512KB（如果全是4字节字符）
        
        if len(text) <= CHUNK_SIZE_CHARS:
            # 小文本直接处理
            return list(self.encode_iterable([text]))
        
        # 大文本分块处理
        def chunk_generator() -> Iterator[str]:
            for i in range(0, len(text), CHUNK_SIZE_CHARS):
                yield text[i:i + CHUNK_SIZE_CHARS]
        
        return list(self.encode_iterable(chunk_generator()))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        流式 / 分块 tokenizer
        - iterable: 每个元素是一个 str chunk
        - 生成 token ids
        """

        # ---------- special token regex（最长优先） ----------
        specials = sorted(self.special_tokens, key=len, reverse=True)
        sp_pat = None
        max_sp_len = 0
        if specials:
            sp_pat = re.compile("|".join(re.escape(s) for s in specials))
            max_sp_len = len(specials[0])

        # ---------- buffer 尾部保护长度 ----------
        # 要 >= 最大 special token 长度，且给 regex 留安全余量
        KEEP_TAIL = max(512, max_sp_len + 8)

        # buffer 用于累积流式输入的文本块，避免在块边界处切断 pre-token 或 special token（外部分chunk的时候可能会自动切割）
        # 每次从 buffer 中安全地切出一个前缀进行处理，剩余部分保留等待下一个块，相当于将上一个chunk的尾部拼接到下一个chunk处理
        buffer = ""

        # ---------- 工具函数 ----------
        def emit_text(text: str) -> Iterator[int]:
            """对一段完整文本做 encode（可含 special tokens）"""
            if not text:
                return

            if sp_pat is None:
                # 无 special token：直接 pretokenize + BPE
                for m in self._pretoken_pattern.finditer(text):
                    piece = m.group(0)
                    if piece:
                        yield from self._bpe_encode_bytes(piece.encode("utf-8"))
                return

            last = 0
            for m in sp_pat.finditer(text):
                s, e = m.span()

                # 普通文本段
                if s > last:
                    for mm in self._pretoken_pattern.finditer(text[last:s]):
                        piece = mm.group(0)
                        if piece:
                            yield from self._bpe_encode_bytes(piece.encode("utf-8"))

                # special token（原子）
                tok = m.group(0).encode("utf-8")
                tid = self.token_to_id.get(tok)
                if tid is not None:
                    yield tid
                else:
                    # 防御性回退（理论上不会发生）
                    raise ValueError(
                        f"Token {tok!r} not found in vocabulary. "
                        f"This indicates an inconsistency between the vocabulary and BPE merges."
                    )

                last = e

            # 尾部
            if last < len(text):
                for mm in self._pretoken_pattern.finditer(text[last:]):
                    piece = mm.group(0)
                    if piece:
                        yield from self._bpe_encode_bytes(piece.encode("utf-8"))

        def split_safe_prefix(buf: str) -> tuple[str, str]:
            """
            从 buffer 中切出一个"安全前缀"：
            - 不切断 pre-token
            - 不切断 special token
            """
            if len(buf) <= KEEP_TAIL:
                return "", buf

            cutoff = len(buf) - KEEP_TAIL
            last_safe = 0

            # 找到最后一个 pretoken 结束位置 <= cutoff
            # 注意：由于 KEEP_TAIL >= max_sp_len + 8，special token 不会被切断，无需额外检查
            for m in self._pretoken_pattern.finditer(buf):
                if m.end() <= cutoff:
                    last_safe = m.end()
                else:
                    break

            if last_safe == 0:
                return "", buf

            return buf[:last_safe], buf[last_safe:]

        # ---------- 主循环 ----------
        for chunk in iterable:
            if not chunk:
                continue

            buffer += chunk
            safe, buffer = split_safe_prefix(buffer)

            if safe:
                yield from emit_text(safe)

        # ---------- 收尾 ----------
        if buffer:
            yield from emit_text(buffer)

    def decode(self, ids: list[int]) -> str:
        # 将所有token的字节序列连接起来，然后整体解码为UTF-8字符串
        # GPT-2的token可能不是完整的UTF-8字符，需要整体解码
        byte_sequence = b""
        for id in ids:
            byte_sequence += self.id_to_token[id]
        return byte_sequence.decode("utf-8", errors="replace")
