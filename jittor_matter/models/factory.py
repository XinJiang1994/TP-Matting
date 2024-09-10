
from jittor_matter.utils import PolynomialLR
from .vit import VisionTransformer
from .decoder_trans import MaskTransformer
from .matter_trans import TgtFilter
from .matter_cnn import MattingNetwork
from .model import TgtFilterMatterWrapper
from collections import defaultdict
import sys
import jittor as jt

_module_to_models = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module = {}  # mapping of model names to module names
_model_entrypoints = {}  # mapping of model names to entrypoint fns
_model_has_pretrained = set()  # set of model names that have pretrained weight url present
_model_pretrained_cfgs = dict()  # central repo for model default_cfgs


def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_valid_pretrained = False  # check if model has a pretrained url to allow filtering on this
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        cfg = mod.default_cfgs[model_name]
        has_valid_pretrained = (
            ('url' in cfg and 'http' in cfg['url']) or
            ('file' in cfg and cfg['file']) or
            ('hf_hub_id' in cfg and cfg['hf_hub_id'])
        )
        _model_pretrained_cfgs[model_name] = mod.default_cfgs[model_name]
    if has_valid_pretrained:
        _model_has_pretrained.add(model_name)
    return fn

def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model_cfg.pop('normalization')
    model_cfg.pop('backbone')
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]
    model = VisionTransformer(**model_cfg)
    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_filter(model_cfg):
    model_cfg = model_cfg.copy()
    # model_cfg.pop("backbone")
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = TgtFilter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model

def create_model(tgt_filter,filter_ckpt,matting_model_ckpt,load_matter=False):
    matting_model = MattingNetwork('mobilenetv3', pretrained_backbone=True)
    if load_matter:
        matting_model.load(matting_model_ckpt)
    model =TgtFilterMatterWrapper(tgt_filter,matting_model)
    # ckpt=jt.load('ckpt_jittor/checkpoint.pkl')
    # model.load_state_dict(ckpt)
    return model

def create_scheduler(opt_args, optimizer):
    lr_scheduler = PolynomialLR(
            optimizer,
            opt_args.poly_step_size,
            opt_args.iter_warmup,
            opt_args.iter_max,
            opt_args.poly_power,
            opt_args.min_lr,
        )
    return lr_scheduler

# def load_model(model_path):
#     variant_path = Path(model_path).parent / "variant.yml"
#     with open(variant_path, "r") as f:
#         variant = yaml.load(f, Loader=yaml.FullLoader)
#     net_kwargs = variant["net_kwargs"]

#     model = create_filter(net_kwargs)
#     data = torch.load(model_path, map_location=ptu.device)
#     checkpoint = data["model"]

#     model.load_state_dict(checkpoint, strict=True)

#     return model, variant
