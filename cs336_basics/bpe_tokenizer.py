from collections.abc import Iterable, Iterator
import json
import os
import regex as re


class MyBpeTokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # 将special_tokens添加到vocab中
        if special_tokens:
            max_key = max(self.vocab.keys())
            for token in special_tokens:
                if token.encode('utf-8') not in self.vocab.values():
                    max_key += 1
                    self.vocab[max_key] = token.encode('utf-8')

        # 创建反向映射字典
        self.token_to_id = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            # 确保所有键都是整数类型
            vocab = {int(k): v.encode('utf-8') if isinstance(v, str) else v
                     for k, v in vocab_data.items()}
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                # 只使用中间的空格分割
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    token1, token2 = parts
                    merges.append(
                        (token1.encode('utf-8'), token2.encode('utf-8')))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            # 按长度降序排序特殊标记，确保先匹配较长的标记
            sorted_special_tokens = sorted(
                self.special_tokens, key=len, reverse=True)
            special_tokens_pattern = "|".join(
                re.escape(token) for token in sorted_special_tokens)
            # Split the chunk by special tokens
            segments = re.split(f"({special_tokens_pattern})", text)
        else:
            segments = [text]

        result = []
        # Process each segment
        for segment in segments:
            # Skip empty segments
            if not segment:
                continue
            if self.special_tokens and segment in self.special_tokens:
                result.append(self.token_to_id.get(segment.encode('utf-8')))
                continue

            # pretokenize
            tokens = []
            # 修改正则表达式以更好地处理 Unicode 字符
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            matches = re.finditer(PAT, segment, re.UNICODE)
            for match_token in matches:
                tokens.append(match_token.group(0))

            token_result = []

            for token in tokens:
                token_bytes = token.encode('utf-8')
                token_tuple = tuple(bytes([b]) for b in token_bytes)
                for merge in self.merges:
                    # 如果token_tuple中存在merge，则替换为merge
                    for i in range(len(token_tuple)-1):
                        if token_tuple[i:i+2] == merge:
                            token_tuple = token_tuple[:i] + \
                                (b''.join(merge),) + token_tuple[i+2:]

                token_result.append(token_tuple)

            for token_tuple in token_result:
                for token in token_tuple:
                    if token in self.token_to_id:
                        result.append(int(self.token_to_id.get(token)))
                    else:
                        result.append(token)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            # 对每个字符串进行编码
            encoded_tokens = self.encode(text)
            # 逐个生成token ID
            for token_id in encoded_tokens:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        res = b''
        for id in ids:
            if id in self.vocab:
                res += self.vocab[id]
            else:
                res += id
        return res.decode('utf-8', errors='replace')


if __name__ == "__main__":
    # vocab = {
    #     0: b' ', 1: b'a', 2:
    #     b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'
    # }
    # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'),
    #           (b' a', b't')]
    # tokenizer = MyBpeTokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    # decode_ex = tokenizer.encode("the cat ate")
    # print(decode_ex)
    # print(tokenizer.decode(decode_ex))
    vocab_path = "/root/scratch/assignment1-basics/vocab_gpt2.json"
    merges_path = "/root/scratch/assignment1-basics/merges_gpt2.txt"
    tokenizer = MyBpeTokenizer.from_files(vocab_path, merges_path, special_tokens=[
                                          "<|endoftext|>", "<|endoftext|><|endoftext|>"])
    test_string = " dedicated"
    encoded_ids = tokenizer.encode(test_string)
    print(encoded_ids)
    decoded_string = tokenizer.decode(encoded_ids)
    print(decoded_string)
