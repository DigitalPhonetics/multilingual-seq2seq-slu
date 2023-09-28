from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

from transformers import AutoTokenizer
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class HuggingFaceTokenizer(AbsTokenizer):
    def __init__(self, model: Union[Path, str]):
        assert check_argument_types()
        self.model = str(model)
        # NOTE(kamo):
        # Don't build tokenizer in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.tokenizer = None

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_tokenizer(self):
        # Build Hugging Face tokenizer lazily.
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_tokenizer()
        return self.tokenizer.tokenize(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_tokenizer()
        return self.tokenizer.convert_tokens_to_string(list(tokens))
