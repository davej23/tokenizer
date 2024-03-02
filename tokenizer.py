import logging
import sys

from datetime import datetime


# log_file = f"tokenizer-log-{datetime.now().strftime('%Y%m%d-%H-%M-%S')}.log"
log_file = "tokenizer-log.log"
logging.basicConfig(
    filename=log_file, filemode="a", level=logging.DEBUG
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(log_file, "w"))

# Print to stdout
# logger.addHandler(logging.StreamHandler(sys.stdout))


class Tokenizer:
    """Tokenizer class
    This tokenizer will be trainable using an input corpus, creating a set of merges and vocabulary

    Args:
        nb_merges: number of byte-pairs to merge into new token
        max_vocab_size: maximum size of vocabulary
        log_level: level for logger

    """
    def __init__(self, nb_merges: int = 10, max_vocab_size: int = 100000, log_level: str = "DEBUG"):
        self._nb_merges = nb_merges
        self._max_vocab_size = max_vocab_size
        self._byte_pair_map = {}
        self._vocab = list(range(256))

        logger.setLevel(log_level)

    @staticmethod
    def _get_byte_pairs(bytes_list: list[int]) -> dict[tuple[int, int], int]:
        """Convert list of bytes into counter of byte pairs"""

        logger.debug("_get_byte_pairs : started converting byte pair list to counters")

        byte_pair_counter = {}
        for c1, c2 in zip(bytes_list, bytes_list[1:]):
            if (c1, c2) not in byte_pair_counter:
                byte_pair_counter[(c1, c2)] = 1
            else:
                byte_pair_counter[(c1, c2)] += 1

        logger.debug(f"_get_byte_pairs : byte pair counters complete {byte_pair_counter}")

        return byte_pair_counter

    @staticmethod
    def _sort_byte_pairs(byte_pair_counter: dict[tuple[int, int], int]) -> list[tuple[int, tuple[int, int]]]:
        """Convert counter of byte pairs into sorted list of most to least occurrent pairs"""

        list_bps = list((v, k) for k, v in byte_pair_counter.items())
        return list(sorted(list_bps, key=lambda a: a[0], reverse=True))

    @staticmethod
    def convert_to_byte_list(text: str) -> list[int]:
        """Convert text to list of bytes"""

        return list(text.encode("utf-8", errors="replace"))

    @staticmethod
    def onehot_encode(tokens: list[int], vocab: list[int]) -> list[list[float]]:
        """Convert tokens into one-hot tokens (N tokens with M vocab size --> NxM array)"""

        return [[1.0 if vocab.index(token) == i else 0.0 for i in range(len(vocab))] for token in tokens]

    def _train(self, text_bytes: list[int]) -> None:
        """Train tokenizer until vocab size >= {self._max_vocab_size}"""

        logger.debug("_train : started train function")

        compressed_text_bytes = text_bytes.copy()
        byte_pairs_by_frequency = Tokenizer._get_byte_pairs(text_bytes)
        sorted_byte_pairs_by_frequency = Tokenizer._sort_byte_pairs(byte_pairs_by_frequency)

        while True:
            logger.debug(f"_train : remaining byte pairs to replace {len(sorted_byte_pairs_by_frequency)}, " +
                         f"len text bytes {len(text_bytes)}, len compressed text bytes {len(compressed_text_bytes)}, " +
                         f"vocab size {len(self._vocab)}, max vocab size {self._max_vocab_size}")

            if len(sorted_byte_pairs_by_frequency) > 0 and \
                    (len(self._vocab) <= self._max_vocab_size and
                     len(text_bytes) - self._nb_merges < len(compressed_text_bytes)):
                for (c1_i, c1), (c2_i, c2) in zip(enumerate(compressed_text_bytes),
                                                  enumerate(compressed_text_bytes[1:], start=1)):
                    if (c1, c2) == sorted_byte_pairs_by_frequency[0][1]:
                        self._vocab.append(self._vocab[-1] + 1)
                        self._byte_pair_map[(c1, c2)] = self._vocab[-1]
                        compressed_text_bytes[c1_i] = self._vocab[-1]
                        compressed_text_bytes.pop(c2_i)
                sorted_byte_pairs_by_frequency.pop(0)
            else:
                break

    def get_vocab(self) -> list[int]:
        """Get vocabulary"""

        return self._vocab

    def get_byte_pair_map(self) -> dict[tuple[int, int], int]:
        """Get byte-pair to token map"""

        return self._byte_pair_map

    def train(self, text: str) -> None:
        """Train tokenizer using input text/corpus"""

        logger.debug(f"train : started training")

        byte_list = Tokenizer.convert_to_byte_list(text)

        logger.debug(f"train : corpus converted to byte list")
        logger.debug(f"train : number of unique characters in corpus = {len(set(byte_list))}")

        # If size of vocabulary already larger than maximum vocabulary size, stop
        if len(self._vocab) >= self._max_vocab_size:
            logger.warning(f"train: initial vocab size {len(self._vocab)} >= max vocab size {self._max_vocab_size}")
            return None

        self._train(byte_list)

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into list of bytes"""

        text_bytes = list(text.encode("utf-8", errors="replace"))
        text_bytes = [text_byte if text_byte in self._vocab else -1 for text_byte in text_bytes]
        encoded_text_bytes = text_bytes.copy()

        logger.debug(f"tokenize : Text to bytes yielded: {text_bytes}")

        for pair, token in self._byte_pair_map.items():
            for (c1_i, c1), (c2_i, c2) in zip(enumerate(encoded_text_bytes),
                                              enumerate(encoded_text_bytes[1:], start=1)):
                if (c1, c2) == pair:
                    logger.debug(f"tokenize : Pair {pair} found in text - inserting token {token}")
                    encoded_text_bytes[c1_i] = token
                    encoded_text_bytes.pop(c2_i)

        return encoded_text_bytes

    def untokenize(self, text_bytes: list[int]) -> str:
        """Convert list of tokens to string"""

        logger.debug("untokenize : started untokenize")

        decoded_text_bytes = text_bytes.copy()

        for pair, token in self._byte_pair_map.items():
            logger.debug(f"untokenize : Byte pair {pair} for token {token}")

            for c_i, c in enumerate(decoded_text_bytes):
                logger.debug(f"untokenize : \tText byte {c} at position {c_i}")
                if c == token:
                    logger.debug(f"untokenize : \tText byte {c} FOUND in byte pair map - inserting {pair} into byte list")
                    decoded_text_bytes[c_i] = pair[0]
                    decoded_text_bytes.insert(c_i + 1, pair[1])

        return bytearray(decoded_text_bytes).decode("utf-8")
