from enum import IntEnum, Enum
from backend.sampling import SAMPLER_NAMES
from backend.schedulers import SCHEDULER_NAMES

disabled = 'Disabled'
enabled = 'Enabled'

uov_list = [disabled, 'Upscale', 'Super-Upscale']

remove_bg = 'remove_bg'
remove_obj = 'remove_obj'


CIVITAI_NO_KARRAS = ["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]

# fooocus: a1111 (Civitai)
KSAMPLER = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "",
    "dpmpp_3m_sde_gpu": "",
    "ddpm": "",
    "lcm": "LCM",
    "tcd": "TCD",
    "restart": "Restart"
}

SAMPLER_EXTRA = {
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": ""
}

SAMPLERS = KSAMPLER | SAMPLER_EXTRA

KSAMPLER_NAMES = list(KSAMPLER.keys())

sampler_list = SAMPLER_NAMES
scheduler_list = SCHEDULER_NAMES

clip_skip_max = 12

default_vae = 'Default (model)'


default_input_image_tab = 'uov_tab'
input_image_tab_ids = ['uov_tab', 'ip_tab', 'inpaint_tab', 'metadata_tab']

cn_structural = 'Structural'
cn_contextual = 'Contextual'
cn_channels = [cn_structural, cn_contextual]

cn_ip = 'ImagePrompt'
cn_faceid = 'FaceID V2'
cn_pulid = 'PuLID'
cn_ip_face = 'FaceSwap'
cn_canny = 'PyraCanny'
cn_mistoline = 'MistoLine'
cn_depth = 'Depth'
cn_cpds = 'CPDS'
cn_mlsd = 'MLSD'

cn_structural_types = [cn_canny, cn_mistoline, cn_depth, cn_cpds, cn_mlsd]
cn_contextual_types = [cn_ip, cn_faceid, cn_pulid]
cn_type_aliases = {
    cn_ip_face: cn_faceid,
}
cn_all_types = cn_structural_types + cn_contextual_types
cn_type_to_channel = {guidance_type: cn_structural for guidance_type in cn_structural_types}
cn_type_to_channel.update({guidance_type: cn_contextual for guidance_type in cn_contextual_types})
cn_type_to_channel.update({legacy_type: cn_contextual for legacy_type in cn_type_aliases})

# Legacy compatibility list. New UI surfaces the explicit channel lists above.
ip_list = cn_all_types + list(cn_type_aliases.keys())
default_ip = cn_ip

default_parameters = {
    cn_ip: (0.9, 0.9),
    cn_faceid: (0.9, 0.9),
    cn_pulid: (0.9, 0.9),
    cn_ip_face: (0.9, 0.9),
    cn_canny: (0.5, 0.9),
    cn_mistoline: (0.5, 0.9),
    cn_depth: (0.5, 0.9),
    cn_cpds: (0.5, 0.9),
    cn_mlsd: (0.5, 0.9),
}  # stop, weight


def normalize_cn_type(value):
    if value is None:
        return None
    return cn_type_aliases.get(value, value)


def resolve_cn_type(value, default=default_ip):
    normalized = normalize_cn_type(value)
    if normalized in cn_all_types:
        return normalized
    return default


def get_cn_channel(value):
    return cn_type_to_channel.get(normalize_cn_type(value))


def get_default_cn_type_for_channel(channel):
    choices = cn_structural_types if channel == cn_structural else cn_contextual_types
    return choices[0]


def get_default_cn_parameters_for_type(cn_type):
    normalized_type = resolve_cn_type(cn_type)
    return default_parameters.get(normalized_type, default_parameters[default_ip])

output_formats = ['png', 'jpeg', 'webp']

inpaint_mask_models = ['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime']
inpaint_mask_cloth_category = ['full', 'upper', 'lower']

inpaint_engine_versions = ['None', 'v1', 'v2.5', 'v2.6']
inpaint_option_default = 'Outpaint (2-Step)'
inpaint_option_detail = 'Improve Detail (face, hand, eyes, etc.)'
inpaint_option_modify = 'Modify Content (add objects, change background, etc.)'
inpaint_options = [inpaint_option_default, inpaint_option_detail, inpaint_option_modify]


sdxl_aspect_ratios = [
    '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
    '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
    '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
    '1664*576', '1728*576'
]

sd15_aspect_ratios = [
    '512*768', '512*704', '576*704', '576*640', '640*640', '640*576',
    '704*576', '704*512', '768*512', '512*512', '768*768', '896*512',
    '512*896', '1024*512', '512*1024'
]


class MetadataScheme(Enum):
    FOOOCUS = 'fooocus'
    FOOOCUS_NEX = 'fooocus_nex'


metadata_scheme = [
    (f'{MetadataScheme.FOOOCUS_NEX.value} (json)', MetadataScheme.FOOOCUS_NEX.value),
    (f'{MetadataScheme.FOOOCUS.value} (json)', MetadataScheme.FOOOCUS.value),
]


class OutputFormat(Enum):
    PNG = 'png'
    JPEG = 'jpeg'
    WEBP = 'webp'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: c.value, cls))


