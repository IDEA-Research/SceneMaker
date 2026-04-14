import random
import torch
from torch import nn
import numpy as np
import re
import urllib.parse as ul
from bs4 import BeautifulSoup
from einops import rearrange
from dataclasses import dataclass
from torchvision import transforms
from diffusers.models.modeling_utils import ModelMixin

from transformers import AutoImageProcessor, AutoModel
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

import step1x3d_geometry
from step1x3d_geometry.utils.typing import *

from .base import BaseCaptionEncoder

bad_punct_regex = re.compile(
    r"["
    + "#®•©™&@·º½¾¿¡§~"
    + "\)"
    + "\("
    + "\]"
    + "\["
    + "\}"
    + "\{"
    + "\|"
    + "\\"
    + "\/"
    + "\*"
    + r"]{1,}"
)  # noqa


@step1x3d_geometry.register("t5-encoder")
class T5Encoder(BaseCaptionEncoder, ModelMixin):

    @dataclass
    class Config(BaseCaptionEncoder.Config):
        pretrained_model_name_or_path: Optional[str] = (
            None  # the pretrained model name or path for condition model
        )
        pretrained_t5_name_or_path: Optional[str] = (
            None  # the pretrained model name or path for T5
        )
        preprocessing_text: bool = False
        text_max_length: int = 77
        t5_type: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Load the T5 model and tokenizer
        if self.cfg.pretrained_t5_name_or_path is not None:
            self.cfg.t5_type = f"google-t5/{self.cfg.pretrained_t5_name_or_path.split('google-t5--')[-1].split('/')[0]}"
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.cfg.pretrained_t5_name_or_path
            )
            self.text_model = T5EncoderModel.from_pretrained(
                self.cfg.pretrained_t5_name_or_path, torch_dtype=torch.bfloat16
            )
        else:
            if (
                self.cfg.pretrained_model_name_or_path is None
            ):  # default to load t5-base model
                assert self.cfg.t5_type is not None, "The t5_type should be provided"
                print(f"Loading T5 model from {self.cfg.t5_type}")
                self.text_model = T5EncoderModel(
                    config=T5EncoderModel.config_class.from_pretrained(
                        self.cfg.t5_type,
                    )
                ).to(torch.bfloat16)
            elif "t5small" in self.cfg.pretrained_model_name_or_path:
                print("Loading Dinov2 model from google-t5/t5-small")
                self.cfg.t5_type = "google-t5/t5-small"
                self.text_model = T5EncoderModel.from_pretrained(
                    self.cfg.t5_type, torch_dtype=torch.bfloat16
                )
            elif "t5base" in self.cfg.pretrained_model_name_or_path:
                print("Loading Dinov2 model from google-t5/t5-base")
                self.cfg.t5_type = "google-t5/t5-base"
                self.text_model = T5EncoderModel.from_pretrained(
                    self.cfg.t5_type, torch_dtype=torch.bfloat16
                )
            else:
                raise ValueError(
                    f"Unknown T5 model: {self.cfg.pretrained_model_name_or_path}"
                )
            self.tokenizer = T5Tokenizer.from_pretrained(self.cfg.t5_type)

        # Set the empty image/text embeds
        if self.cfg.zero_uncond_embeds:
            self.empty_text_embeds = torch.zeros(
                (1, self.cfg.text_max_length, self.text_model.config.hidden_size)
            ).detach()
        else:
            self.empty_text_embeds = self.encode_text([""]).detach()

        # load pretrained_model_name_or_path
        if self.cfg.pretrained_model_name_or_path is not None:
            print(f"Loading ckpt from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(
                self.cfg.pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            pretrained_model_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith("caption_condition."):
                    pretrained_model_ckpt[k.replace("caption_condition.", "")] = v
            self.load_state_dict(pretrained_model_ckpt, strict=True)

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(
            bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def text_preprocessing(self, text):
        if self.cfg.preprocessing_text:
            # The exact text cleaning as was in the training stage:
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    def encode_text(self, texts: List[str]) -> torch.FloatTensor:
        texts = [self.text_preprocessing(text) for text in texts]

        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.cfg.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_tokens_and_mask["input_ids"] = text_tokens_and_mask["input_ids"]  # N x 77
        text_tokens_and_mask["attention_mask"] = text_tokens_and_mask["attention_mask"]

        with torch.no_grad():
            label_embeds = self.text_model(
                input_ids=text_tokens_and_mask["input_ids"].to(self.text_model.device),
                attention_mask=text_tokens_and_mask["attention_mask"].to(
                    self.text_model.device
                ),
            )["last_hidden_state"].detach()

        return label_embeds
