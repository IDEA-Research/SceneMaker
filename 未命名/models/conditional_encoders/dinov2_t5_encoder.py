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

from transformers import AutoImageProcessor, AutoModel
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

import craftsman
from craftsman.utils.typing import *
from .base import BaseEmbedder, ImageType
from .dinov2.modeling_dinov2 import Dinov2Model
from .dinov2.modeling_conditional_dinov2 import ConditionalDinov2Model
from .dinov2_with_registers.modeling_dinov2_with_registers import Dinov2WithRegistersModel

bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')  # noqa

class DINOEmbedOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
        
@craftsman.register("dinov2-t5-embedder")
class Dinov2T5Embedder(BaseEmbedder):

    @dataclass
    class Config(BaseEmbedder.Config):
        pretrained_model_name_or_path: Optional[str] = None # the pretrained model name or path for condition model
        pretrained_dino_name_or_path: Optional[str] = None # the pretrained model name or path for dino
        pretrained_t5_name_or_path: Optional[str] = None # the pretrained model name or path for T5
        linear_proj_init: str = "constant"
        use_text_preprocessing: bool = False
        text_max_length: int = 77
        freeze_modulation_dino: bool = False
        enable_gradient_checkpointing: bool = False
        image_size_dino: int = 224
        dino_type: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Load the DINOV2 model and processor
        if not self.cfg.encode_camera:
            if self.cfg.pretrained_dino_name_or_path is not None:
                self.dino_type = self.cfg.pretrained_dino_name_or_path
                self.dino_model: Dinov2Model = AutoModel.from_pretrained(self.cfg.pretrained_dino_name_or_path)
            else:
                if self.cfg.pretrained_model_name_or_path is None: # default to load Dinov2-base model
                    assert self.cfg.dino_type is not None, "The dino_type should be provided"
                    print(f"Loading Dinov2 model from {self.cfg.dino_type}")
                    self.dino_type = f"facebook/{self.cfg.dino_type}"
                    if "reg" in self.cfg.dino_type:
                        self.dino_model: Dinov2WithRegistersModel = Dinov2WithRegistersModel(config=Dinov2WithRegistersModel.config_class.from_pretrained(
                            self.dino_type,
                        ))
                    else:
                        self.dino_model: Dinov2Model = Dinov2Model(config=Dinov2Model.config_class.from_pretrained(
                            self.dino_type,
                        ))
                elif "dinov2base" in self.cfg.pretrained_model_name_or_path:
                    print("Loading Dinov2 model from facebook/dinov2-base")
                    self.dino_type = "facebook/dinov2-base"
                    self.dino_model: Dinov2Model = Dinov2Model(config=Dinov2Model.config_class.from_pretrained(
                        "facebook/dinov2-base",
                    ))
                elif "dinov2regbase" in self.cfg.pretrained_model_name_or_path:
                    print("Loading Dinov2 model from facebook/dinov2-with-registers-base")
                    self.dino_type = "facebook/dinov2-with-registers-base"
                    self.dino_model: Dinov2WithRegistersModel = Dinov2WithRegistersModel(config=Dinov2WithRegistersModel.config_class.from_pretrained(
                        "facebook/dinov2-with-registers-base",
                    ))
                elif "dinov2reglarge" in self.cfg.pretrained_model_name_or_path:
                    print("Loading Dinov2 model from facebook/dinov2-with-registers-large")
                    self.dino_type = "facebook/dinov2-with-registers-large"
                    self.dino_model: Dinov2WithRegistersModel = Dinov2WithRegistersModel(config=Dinov2WithRegistersModel.config_class.from_pretrained(
                        "facebook/dinov2-with-registers-large",
                    ))
                else:
                    raise ValueError(f"Unknown Dinov2 model: {self.cfg.pretrained_model_name_or_path}")
        else:
            # dino
            conditional_vit_config = ConditionalDinov2Model.config_class.from_pretrained(
                self.cfg.pretrained_dino_name_or_path,
            )                               
            conditional_vit_config.modulation_dim = self.cfg.camera_embeds_dim
            self.dino_model: ConditionalDinov2Model = ConditionalDinov2Model.from_pretrained(
                self.cfg.pretrained_dino_name_or_path,
                config=conditional_vit_config
            )
                    
        self.image_preprocess_dino = AutoImageProcessor.from_pretrained(self.dino_type)
        self.transform_dino = transforms.Compose(
            [   
                transforms.Resize(self.cfg.image_size_dino, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(self.cfg.image_size_dino),  # crop a (image_size_dino, image_size_dino) square
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if self.cfg.enable_gradient_checkpointing:
            self.dino_model.encoder.gradient_checkpointing = True

        # Load the T5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.cfg.pretrained_t5_name_or_path)
        self.text_model = T5EncoderModel.from_pretrained(self.cfg.pretrained_t5_name_or_path, torch_dtype=torch.bfloat16)
        self.linear_proj_text = nn.Linear(self.text_model.config.hidden_size, self.dino_model.config.hidden_size, bias=False)
        if self.cfg.linear_proj_init == "constant":
            nn.init.constant_(self.linear_proj_text.weight, 0)
        elif self.cfg.linear_proj_init == "xavier":
            nn.init.xavier_uniform_(self.linear_proj_text.weight)
        else:
            raise ValueError

        # Set the empty image/text embeds
        if self.cfg.zero_uncond_embeds:
            self.empty_image_embeds = torch.zeros((self.cfg.n_views, (self.cfg.image_size_dino // 14) ** 2 + 1, self.dino_model.config.hidden_size)).detach()
            self.empty_text_embeds = torch.zeros((1, self.cfg.text_max_length, self.text_model.config.hidden_size)).detach()
        else:
            if self.cfg.encode_camera:
                self.empty_image_embeds = self.encode_image_dino(torch.zeros(self.cfg.n_views, self.cfg.image_size_dino, self.cfg.image_size_dino, 3), self.cameras[:self.cfg.n_views]).detach()
            else:
                self.empty_image_embeds = self.encode_image_dino(torch.zeros(self.cfg.n_views, self.cfg.image_size_dino, self.cfg.image_size_dino, 3)).detach()
            self.empty_text_embeds = self.encode_text([""]).detach()

        # freeze the dino model parameters
        self.dino_model.eval()
        for k, p in self.dino_model.named_parameters():
            ks = k.split('.')
            if 'mod_norm1' in ks or 'mod_norm2' in ks and not self.cfg.freeze_modulation_dino:
                p.requires_grad_(not self.cfg.freeze_modulation_dino)
            else:
                p.requires_grad_(False)

        # load pretrained_model_name_or_path
        if self.cfg.pretrained_model_name_or_path is not None:
            print(f"Loading ckpt from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")['state_dict']
            pretrained_model_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('condition.'):
                    pretrained_model_ckpt[k.replace('condition.', '')] = v
            self.load_state_dict(pretrained_model_ckpt, strict=True)
 
    def encode_image_dino(self, images: Iterable[Optional[ImageType]], cameras: Optional[torch.Tensor] = None, force_none_camera_embeds: bool = False, return_dict: bool = False, **kwargs) -> torch.FloatTensor:
        camera_embeds = None
        if isinstance(images, (np.ndarray, torch.Tensor)): # for training process
            assert images.min() >= 0.0 and images.max() <= 1.0, "The pixel values should be in the range of [0, 1]"
            if self.cfg.encode_camera:
                assert cameras is not None, "The cameras should be provided"
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.transform_dino(images.permute(0, 3, 1, 2))
        else: # for inference process
            if self.cfg.encode_camera:
                if cameras is None:
                    bs = len(images) // self.cfg.n_views
                    cameras = self.cameras[:self.cfg.n_views].repeat(bs, 1, 1).to(self.dino_model.device)
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.image_preprocess_dino.preprocess(images, return_tensors='pt', \
                    do_rescale=True, do_resize=True, size=self.cfg.image_size_dino, crop_size=self.cfg.image_size_dino).pixel_values

        if force_none_camera_embeds:
            camera_embeds = None

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
            if camera_embeds is not None:
                camera_embeds = camera_embeds.unsqueeze(1)

        if self.cfg.encode_camera and camera_embeds is not None:
            vision_outputs = self.dino_model(
                rearrange(pixel_values.to(self.dino_model.device), "B N C H W -> (B N) C H W"), 
                condition=rearrange(camera_embeds, "B N C -> (B N) C"),
            )
        else:
            vision_outputs = self.dino_model(
                rearrange(pixel_values.to(self.dino_model.device), "B N C H W -> (B N) C H W"), 
            )

        if return_dict:
            # dino
            dino_embeds_dict = DINOEmbedOutput(
                last_hidden_state=vision_outputs.last_hidden_state,
                pooler_output=vision_outputs.pooler_output,
            )
            return dino_embeds_dict
        else:
            return vision_outputs.last_hidden_state

    @torch.no_grad()
    def encode_image(self, images: Iterable[Optional[ImageType]], cameras: Optional[torch.Tensor] = None, force_none_camera_embeds: bool = False, return_dict: bool = False, **kwargs) -> torch.FloatTensor:
        dino_embeds = self.encode_image_dino(images, cameras)
        if self.dino_model.__class__.__name__ == 'Dinov2WithRegistersModel': # x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens
            dino_embeds = torch.cat(
                [dino_embeds[:, :1], dino_embeds[:, self.dino_model.config.num_register_tokens + 1:]],
                dim=1
            )
        return dino_embeds

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)
        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features='html.parser').text

        # @<nickname>
        caption = re.sub(r'@[\w\d]+\b', '', caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
            '-', caption)

        # кавычки к одному стандарту
        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)

        # &quot;
        caption = re.sub(r'&quot;?', '', caption)
        # &amp
        caption = re.sub(r'&amp', '', caption)

        # ip adresses:
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

        # article ids:
        caption = re.sub(r'\d:\d\d\s+$', '', caption)

        # \n
        caption = re.sub(r'\\n', ' ', caption)

        # "#123"
        caption = re.sub(r'#\d{1,3}\b', '', caption)
        # "#12345.."
        caption = re.sub(r'#\d{5,}\b', '', caption)
        # "123456.."
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

        #
        caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

        caption = re.sub(bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)

        caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)

        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)

        return caption.strip()

    def text_preprocessing(self, text):
        if self.cfg.use_text_preprocessing:
            # The exact text cleaning as was in the training stage:
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.FloatTensor:
        texts = [self.text_preprocessing(text) for text in texts]

        text_tokens_and_mask = self.tokenizer(
            texts, 
            max_length=self.cfg.text_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        text_tokens_and_mask['input_ids'] = text_tokens_and_mask['input_ids'] # N x 77
        text_tokens_and_mask['attention_mask'] = text_tokens_and_mask['attention_mask']

        with torch.no_grad():
            text_embedding = self.text_model(
                input_ids=text_tokens_and_mask['input_ids'].to(self.text_model.device),
                attention_mask=text_tokens_and_mask['attention_mask'].to(self.text_model.device),
            )['last_hidden_state'].detach()
        
        text_embedding = self.linear_proj_text(text_embedding)
        
        return text_embedding