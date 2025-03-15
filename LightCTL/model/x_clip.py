import math
import copy
from contextlib import contextmanager
from functools import partial, wraps
from torch.nn import LayerNorm
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from .visual_ssl import SimSiam,SimCLR,SimCLR_TCR
def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# rotary positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class extracter_pMHC(nn.Module):
    def __init__(self):
        super(extracter_pMHC, self).__init__()

        self.feature1 = nn.Sequential(
            nn.Conv2d(1, 16, (3,3), stride=1, padding=1, bias=False),  # 605
            nn.BatchNorm2d(16),
            nn.MaxPool2d((1,2), stride=2),#605
            nn.ReLU(inplace=True),
        )

        self.feature2 = nn.Sequential(
            nn.Conv2d(16, 32, (3,3), stride=1, padding=1, bias=False),  # 303
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),#605
            nn.ReLU(inplace=True),
        )

    def forward(self, x,return_loss=False):
        x = self.feature1(x)
        x = self.feature2(x)
        b, c,w,h=x.shape
        if return_loss:
            out=x.reshape(b,c,-1)
            return out.transpose(-1,-2)
        # print(x.shape)
        else:
            return x
class extracter_TCR(nn.Module):
    def __init__(self):
        super(extracter_TCR, self).__init__()

        self.feature1 = nn.Sequential(
            nn.Conv2d(1, 16, (3,3), stride=1, padding=1, bias=False),  # 605
            nn.BatchNorm2d(16),
            # nn.MaxPool2d((1,2), stride=2),#605
            nn.ReLU(inplace=True),
        )

        self.feature2 = nn.Sequential(
            nn.Conv2d(16, 32, (3,3), stride=1, padding=1, bias=False),  # 303
            nn.BatchNorm2d(32),
            # nn.MaxPool2d(2, stride=2),#605
            nn.ReLU(inplace=True),
        )

    def forward(self, x,return_loss=False):
        x = self.feature1(x)
        x = self.feature2(x)
        b,c,h,w=x.shape
        if return_loss:
            # out=x.view(b,c,-1)
            # return out.transpose(-1,-2)
            # out = x.reshape(b, c, -1)
            return x
        # print(x.shape)
        else:
            return x

class extracter_Transformer_pMHC(nn.Module):
    def __init__(self):
        super(extracter_Transformer_pMHC, self).__init__()
        self.transformer_encoder =torch.nn.Transformer().encoder
        # print(self.transformer_encoder)
        self.feature1 = nn.Sequential(
            nn.Conv2d(21, 512, kernel_size=1, stride=1, bias=False),  # 605
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x,return_loss=False):
        x=self.feature1(x.permute(0, 3, 2, 1))
        # print('pMHC_x.shape:', x.shape)
        x=torch.squeeze(x.permute(0,3,2,1),dim=1)
        # print('pMHC_x.shape:', x.shape)
        e1 = self.transformer_encoder(x)
        # print('pMHC_e1.shape:',e1.shape)

        if return_loss:
            # out=x.view(b,c,-1)
            # return out.transpose(-1,-2)
            # out = x.reshape(b, c, -1)
            return x
        # print(x.shape)
        else:
            return x

class extracter_Transformer_TCR(nn.Module):
    def __init__(self):
        super(extracter_Transformer_TCR, self).__init__()
        self.transformer_encoder =torch.nn.Transformer().encoder
        # print(self.transformer_encoder)
        self.feature1 = nn.Sequential(
            nn.Conv2d(5, 512, kernel_size=1, stride=1, bias=False),  # 605
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x,return_loss=False):
        x=self.feature1(x.permute(0, 3, 2, 1))
        x=torch.squeeze(x.permute(0,3,2,1),dim=1)
        # print('TCR_x.shape:', x.shape)
        e1 = self.transformer_encoder(x)
        # print('TCR_e1.shape:',e1.shape)

        if return_loss:
            # out=x.view(b,c,-1)
            # return out.transpose(-1,-2)
            # out = x.reshape(b, c, -1)
            return x
        # print(x.shape)
        else:
            return x

class CLIP_CNN(nn.Module):
    def __init__(
        self,
        *,
        image_encoder = None,
        text_encoder = None,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = (49, 21),#(49,34)
        text_heads = 8,
        text_dim_head = 64,
        text_has_cls_token = True,
        text_pad_id = 0,
        text_rotary_pos_emb = False,
        text_causal_mask = False,
        text_eos_id = None,
        text_encode_without_mask = False,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_dim_head = 64,
        visual_image_size = (80,5),#(80,5)
        visual_patch_size = 1,
        visual_has_cls_token = True,
        channels = 32,
        use_all_token_embeds = False,
        downsample_image_embeds = False,
        decoupled_contrastive_learning = False,
        extra_latent_projection = False,
        use_mlm = True,
        text_ssl=None,
        text_ssl_type='simclr',
        text_ssl_loss_weight = 0.05,
        use_visual_ssl = True,
        visual_ssl = None,
        visual_ssl_type = 'SimCLR_TCR',
        visual_ssl_hidden_layer = -1,
        simclr_temperature = 0.1,
        image_ssl_loss_weight = 0.05,
        multiview_loss_weight = 0.1,
        checkpoint_during_training = False,
        **kwargs
    ):
        super().__init__()
        assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'

        # store some parameters for access

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        # instantiate text transformer
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = extracter_pMHC()

        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = extracter_TCR()

        # text ssl
        self.use_mlm= use_mlm or exists(text_ssl)
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        if self.use_mlm:
            if exists(text_ssl):
                self.text_ssl = text_ssl

            elif use_mlm:
                if text_ssl_type == 'simsiam':
                    ssl_type_text = partial(SimSiam, channels = channels)
                elif text_ssl_type == 'simclr':
                    ssl_type_text = partial(SimCLR, temperature = simclr_temperature, channels = channels)
                else:
                    raise ValueError(f'unknown visual_ssl_type')

                self.text_ssl = ssl_type_text(
                    self.text_transformer,
                    image_size = text_seq_len,
                    hidden_layer = visual_ssl_hidden_layer
                )


        # image ssl
        self.use_visual_ssl = use_visual_ssl or exists(visual_ssl)
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        if self.use_visual_ssl:
            if exists(visual_ssl):
                self.visual_ssl = visual_ssl

            elif use_visual_ssl:
                if visual_ssl_type == 'simsiam':
                    ssl_type = partial(SimSiam, channels = channels)
                elif visual_ssl_type == 'simclr':
                    ssl_type = partial(SimCLR, temperature = simclr_temperature, channels = channels)
                elif visual_ssl_type == 'SimCLR_TCR':
                    ssl_type = partial(SimCLR_TCR, temperature=simclr_temperature, channels=channels)
                else:
                    raise ValueError(f'unknown visual_ssl_type')

                self.visual_ssl = ssl_type(
                    self.visual_transformer,
                    image_size = visual_image_size,
                    hidden_layer = visual_ssl_hidden_layer
                )


        # image&text latent projection

        if downsample_image_embeds:
            assert use_all_token_embeds, 'must be using all token embeds for contrastive learning in order to downsampling'

            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv2d(dim_image, dim_image, 4, stride = 2, padding = 1, bias = False, groups = dim_image),
                nn.Conv2d(dim_image, dim_latent, 1),
                Rearrange('b c h w -> b (h w) c')
            )

            self.to_text_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv2d(dim_text, dim_text, 4, stride = 2, padding = 1, bias = False, groups = dim_image),
                nn.Conv2d(dim_text, dim_latent, 1),
                Rearrange('b c h w -> b (h w) c')
            )

        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)
            self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # temperature

        self.temperature = nn.Parameter(torch.tensor(1.))

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        # proposed in https://arxiv.org/abs/2110.11316 (CLOOB)
        self.extra_latent_projection = extra_latent_projection

        self.to_text_latent_extra = copy.deepcopy(self.to_text_latent)
        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

        self.multiview_loss_weight = multiview_loss_weight

    def forward(
        self,
        text,
        image,
        return_loss = False,
        return_encodings = False,
        return_latents = False,
        freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
        freeze_text_encoder = False,    # text encoder is not trained if this is set to True
        text_to_image = True,           # in the case the extra projection is turned on, would return different similarity values depending on modality directionality
        aug_text = None,                # augmented text (for multiview)
        aug_image = None                # augmented image (for multiview)
    ):
        b, device = text.shape[0], text.device
        out_text = self.text_transformer(text)
        out_image = self.visual_transformer(image)
        if out_text.ndim == 3:
            b,c,_=out_text.shape
            enc_text=out_text.transpose(-1,-2)
            enc_image=out_image.transpose(-1,-2)
        else:
            b, c, _, _ = out_text.shape
            enc_text=out_text.reshape(b,c,-1).transpose(-1,-2)
            enc_image=out_image.reshape(b,c,-1).transpose(-1,-2)

        if return_loss is False:
            return out_text,out_image
        # derive text mask

        # text_mask = text != self.text_pad_id

        # ssl
        text_ssl_loss = 0
        image_ssl_loss = 0

        if return_loss:
            text_ssl_loss = self.text_ssl(text) if self.use_mlm else 0
            image_ssl_loss = self.visual_ssl(image) if self.use_visual_ssl else 0
        # print("image_ssl_loss:",image_ssl_loss)
        # concat augmented texts and images and do some asserts

        num_batch_texts = num_batch_images = 1

        is_multiview = (num_batch_texts > 1 or num_batch_images > 1)
        # assert not (return_loss and not self.training), 'loss cannot be used if not training'
        # assert not (not return_loss and is_multiview), 'do not pass in augmented texts or images if not training'
        # assert not (self.multiview_loss_weight == 0 and is_multiview), 'multiview loss weight cannot be 0 if augmented text or images passed in'


        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        if self.use_all_token_embeds:
            assert enc_text.ndim == 3, 'encoded text must have 3 dimensions (batch, seq, features)'
            assert enc_image.ndim == 3, 'encoded image must have 3 dimensions (batch, seq [height x width], features)'
            text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text
            image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image
        else:
            text_embeds = enc_text[:, 0] if enc_text.ndim == 3 else enc_text
            image_embeds = enc_image[:, 0] if enc_image.ndim == 3 else enc_image

        # project to latents
        text_latents = self.to_text_latent(text_embeds)
        image_latents = self.to_visual_latent(image_embeds)
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # calculate another set of latents for image to text (vs text to image)
        # proposed by CLOOB

        text_latents_extra, image_latents_extra = text_latents, image_latents
        if self.extra_latent_projection:
            text_latents_extra = self.to_text_latent_extra(text_embeds)
            image_latents_extra = self.to_visual_latent_extra(image_embeds)
            text_latents_extra, image_latents_extra = map(l2norm, (text_latents_extra, image_latents_extra))

        # whether to early return latents

        if return_latents:
            if self.extra_latent_projection:
                return text_latents, image_latents, text_latents_extra, image_latents_extra

            return text_latents, image_latents

        # get temperature

        temp = self.temperature.exp()

        # early return, if needed

        if not return_loss and self.use_all_token_embeds:
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            return einsum('b t d, b i d -> b t i', *einsum_args) * temp

        if not return_loss and not self.use_all_token_embeds:
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            return einsum('b d, b d -> b', *einsum_args) * temp

        # split out multiview dimension for text and images

        text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = num_batch_texts)
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = num_batch_images)

        if self.extra_latent_projection:
            text_latents_extra = rearrange(text_latents_extra, '(m b) ... -> m b ...', m = num_batch_texts)
            image_latents_extra = rearrange(image_latents_extra, '(m b) ... -> m b ...', m = num_batch_images)

        # contrastive loss

        """
        m - num batches of text (for multiview)
        n - num batches of images (for multiview)
        x - batches of text
        y - batches of images
        t - sequence dimension along text tokens
        i - sequence dimension along image tokens
        """

        if self.use_all_token_embeds:
            # fine-grained CLIP logic
            sim_text_to_image = einsum('m x t d, n y i d -> m n x y t i', text_latents, image_latents) * temp

            sim_image_to_text = sim_text_to_image
            if self.extra_latent_projection:
                sim_image_to_text = einsum('m x t d, n y i d -> m n x y t i', text_latents_extra, image_latents_extra) * temp

            text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
            # text_to_image_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t', m = num_batch_texts)
            # text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

            # image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts)
            # masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
            image_to_text = reduce(reduce(text_to_image, '... t i -> ... i', 'max'), '... i -> ...', 'mean')
        else:
            text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp
            image_to_text = rearrange(text_to_image, '... t i -> ... i t')

            if self.extra_latent_projection:
                image_to_text = einsum('m t d, n i d -> m n i t', text_latents_extra, image_latents_extra) * temp

        # calculate loss

        text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

        # exponentiate

        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        # numerators

        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

        # denominator

        if self.decoupled_contrastive_learning:
            pos_mask = torch.eye(b, device = device, dtype = torch.bool)
            text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))

        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))

        # loss

        text_to_image_loss = -log(text_to_image_pos / text_to_image_denom).mean(dim = -1)
        image_to_text_loss = -log(image_to_text_pos / image_to_text_denom).mean(dim = -1)

        # calculate CL loss

        cl_losses = (text_to_image_loss + image_to_text_loss) / 2

        # get main CL loss vs multiview CL losses

        cl_loss, multiview_cl_loss = cl_losses[0], cl_losses[1:]

        # if no augmented text or images passed in, multiview loss weight is 0

        multiview_loss_weight = self.multiview_loss_weight if is_multiview else 0

        # calculate weights

        cl_loss_weight = 1 - (self.text_ssl_loss_weight + self.image_ssl_loss_weight + multiview_loss_weight)

        # print("text_ssl_loss",text_ssl_loss)
        # print("image_ssl_loss", image_ssl_loss)

        loss = (cl_loss * cl_loss_weight) \
            + (text_ssl_loss * self.text_ssl_loss_weight) \
            + (image_ssl_loss * self.image_ssl_loss_weight)

        # add multiview CL loss with weight

        if is_multiview:
            loss = loss + multiview_cl_loss.mean() * multiview_loss_weight

        # print("out_text:",loss)
        return loss,out_text,out_image

class Contractive_CNN_80_40(nn.Module):
    def __init__(self, use_mlm= False,use_visual_ssl= False,num_classes=2):
        super(Contractive_CNN_80_40, self).__init__()

        self.extract_feature= CLIP_CNN(
        dim_image = 32,
        dim_text = 32,
        channels=1,
        use_mlm = use_mlm,
        use_visual_ssl=use_visual_ssl,
        use_all_token_embeds = False,
        extra_latent_projection = False,
        mlm_random_token_prob = 0.1
    )
        self.feature3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),  # 605
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),  # 605
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # 605
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.feature4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),  # 303
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),  # 303
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # 303
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


        # SE layers
        self.att3_1 = DoubleConv(64,64//16)
        self.att3_2 = DoubleConv(64//16, 64)
        self.att4_1 = DoubleConv(128, 128 //16)
        self.att4_2 = DoubleConv(128 //16, 128)

        self.classifier1 = nn.Sequential(
            # nn.Linear(33*128, 1024),
            nn.Linear(2*25* 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, TCR,pMHC,returnloss=True):
        # print("TCR.shape",TCR.shape)
        # print("pMHC.shape", pMHC.shape)
        loss_cl,e_tcr,e_pMHC=self.extract_feature(pMHC,TCR,return_loss=returnloss)



        # print("e_tcr.shape", e_tcr.shape)
        # print("e_pMHC.shape", e_pMHC.shape)
        # b,c,w,h=e_tcr.shape
        fusion = torch.cat((e_tcr, e_pMHC), dim=2)
        # fusion=torch.cat((e_tcr.reshape(b,c,-1),e_pMHC.reshape(b,c,-1)),dim=2)
        # print('fusion.shape', fusion.shape)
        x = self.feature3(fusion)
        # print('x.shape',x.shape)
        # print('x.size(2)',x.size(2))
        # print('x.size', x.size)

        # Squeeze
        # w3 = self.att_w3(x)
        w3 = F.avg_pool2d(x, (x.size(2),x.size(3)))
        # print('w3.shape', w3.shape)
        w3 = F.relu(self.att3_1(w3))
        w3 = torch.sigmoid(self.att3_2(w3))
        x = x * w3
        # print(x.shape)

        x = self.feature4(x)
        # print(x.shape)
        # Squeeze
        # w4 = self.att_w4(x)
        w4 = F.avg_pool2d(x, (x.size(2),x.size(3)))
        w4 = F.relu(self.att4_1(w4))
        w4 = torch.sigmoid(self.att4_2(w4))
        x = x * w4
        # print("x.shape",x.shape)

        # x = x.view(x.size(0), 33*128)
        x = x.view(x.size(0), -1)
        # print("x.shape",x.shape)
        x = self.classifier1(x)
        # out=F.log_softmax(x, dim=1)
        # x = self.classifier2(x)
        # x =torch.sigmoid(x)
        # return x,loss
        return x,loss_cl

