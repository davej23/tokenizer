class Tokenizer:
    def __init__(self, nb_merges: int):
        self.nb_merges = nb_merges
        self._merges = {}
        self._vocab = []

    @staticmethod
    def _get_byte_pairs(bytes_list: list[int]) -> dict[tuple[int, int], int]:
        """Convert list of bytes into counter of byte pairs"""

        byte_pair_counter = {}
        for c1, c2 in zip(bytes_list, bytes_list[1:]):
            if (c1, c2) not in byte_pair_counter:
                byte_pair_counter[(c1, c2)] = 1
            else:
                byte_pair_counter[(c1, c2)] += 1

        return byte_pair_counter

    @staticmethod
    def _get_sorted_byte_pairs(byte_pair_counter: dict[tuple[int, int], int]) -> list[tuple[int, tuple[int, int]]]:
        """Convert counter of byte pairs into sorted list of most to least occurrent pairs"""

        list_bps = list((v, k) for k, v in byte_pair_counter.items())
        return list(sorted(list_bps, key=lambda a: a[0], reverse=True))

    def _tokenize(self, text_bytes: list[int], merges: dict[tuple[int, int], int]) -> tuple[list[int], dict[tuple[int, int], int]]:
        """Compress list of bytes {nb_merge} times"""

        reduced_text_bytes = text_bytes.copy()

        byte_pairs_by_frequency = Tokenizer._get_byte_pairs(text_bytes)
        sorted_byte_pairs_by_frequency = Tokenizer._get_sorted_byte_pairs(byte_pairs_by_frequency)

        while True:
            if len(sorted_byte_pairs_by_frequency) > 0 and len(text_bytes) - self.nb_merges < len(reduced_text_bytes):
                for (c1_i, c1), (c2_i, c2) in zip(enumerate(reduced_text_bytes), enumerate(reduced_text_bytes[1:], start=1)):
                    if (c1, c2) == sorted_byte_pairs_by_frequency[0][1]:
                        self._vocab.append(self._vocab[-1] + 1)
                        merges[(c1, c2)] = self._vocab[-1]
                        reduced_text_bytes[c1_i] = self._vocab[-1]
                        reduced_text_bytes.pop(c2_i)
                sorted_byte_pairs_by_frequency.pop(0)
            else:
                break

        return reduced_text_bytes, merges

    def tokenize(self, text: str, nb_rounds: int = 2) -> any:
        """Tokenize text into list of bytes, using {nb_rounds} iterative compression each with {nb_merges} merges"""

        merges = {}
        text_bytes = list(text.encode("utf-8"))
        iter_text_bytes = text_bytes.copy()
        self._vocab = list(set(text_bytes.copy()))
        self._vocab.sort()

        for _ in range(nb_rounds):
            iter_text_bytes, merges = self._tokenize(iter_text_bytes, merges)

        return {"text_bytes": text_bytes, "compressed_text_bytes": iter_text_bytes, "merges": merges}
