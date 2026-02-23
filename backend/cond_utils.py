import torch
import collections
from typing import Any, Callable, List, Dict, Optional, Union, Tuple, NamedTuple

def add_area_dims(area: List[int], num_dims: int) -> List[int]:
    while (len(area) // 2) < num_dims:
        area = [2147483648] + area[:len(area) // 2] + [0] + area[len(area) // 2:]
    return area

def get_area_and_mult(conds: Dict[str, Any], x_in: torch.Tensor, timestep_in: torch.Tensor) -> Optional[collections.namedtuple]:
    dims = tuple(x_in.shape[2:])
    area = None
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    
    if 'area' in conds:
        area = list(conds['area'])
        area = add_area_dims(area, len(dims))
        if (len(area) // 2) > len(dims):
            area = area[:len(dims)] + area[len(area) // 2:(len(area) // 2) + len(dims)]

    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in
    if area is not None:
        for i in range(len(dims)):
            area[i] = min(input_x.shape[i + 2] - area[len(dims) + i], area[i])
            input_x = input_x.narrow(i + 2, area[len(dims) + i], area[i])

    if 'mask' in conds:
        mask_strength = conds.get("mask_strength", 1.0)
        mask = conds['mask']
        assert (mask.shape[1:] == x_in.shape[2:])

        mask = mask[:input_x.shape[0]]
        if area is not None:
            for i in range(len(dims)):
                mask = mask.narrow(i + 1, area[len(dims) + i], area[i])

        mask = mask * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    
    mult = mask * strength

    # Area fuzzing for smooth transitions
    if 'mask' not in conds and area is not None:
        fuzz = 8
        for i in range(len(dims)):
            rr = min(fuzz, mult.shape[2 + i] // 4)
            if area[len(dims) + i] != 0:
                for t in range(rr):
                    m = mult.narrow(i + 2, t, 1)
                    m *= ((1.0 / rr) * (t + 1))
            if (area[i] + area[len(dims) + i]) < x_in.shape[i + 2]:
                for t in range(rr):
                    m = mult.narrow(i + 2, area[i] - 1 - t, 1)
                    m *= ((1.0 / rr) * (t + 1))

    conditioning = {}
    if 'cross_attn' in conds:
        conditioning['c_crossattn'] = conds['cross_attn'].to(device=x_in.device, dtype=x_in.dtype)
    if 'concat' in conds:
        conditioning['c_concat'] = conds['concat'].to(device=x_in.device, dtype=x_in.dtype)

    model_conds = conds.get("model_conds", {})
    for c in model_conds:
        if hasattr(model_conds[c], "process_cond"):
            conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)
        else:
            conditioning[c] = model_conds[c].to(device=x_in.device, dtype=x_in.dtype)

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'uuid'])
    return cond_obj(input_x, mult, conditioning, area, conds.get('uuid'))

def cond_equal_size(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
    if c1 is c2: return True
    if c1.keys() != c2.keys(): return False
    for k in c1:
        if not hasattr(c1[k], 'can_concat') or not c1[k].can_concat(c2[k]):
            return False
    return True

def can_concat_cond(c1: Any, c2: Any) -> bool:
    if c1.input_x.shape != c2.input_x.shape:
        return False
    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        if hasattr(conds[0], 'concat'):
            out[k] = conds[0].concat(conds[1:])
        else:
            out[k] = torch.cat(conds, dim=0)
    return out

def calc_cond_batch(model: Any, conds: List[List[Dict[str, Any]]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any]) -> List[torch.Tensor]:
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue
                to_run.append((p, i))

    while len(to_run) > 0:
        first = to_run[0]
        to_batch = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch.append(x)
        
        items_to_pop = [to_run[i] for i in to_batch]
        for i in sorted(to_batch, reverse=True):
            to_run.pop(i)
            
        batch_input_x = []
        batch_mult = []
        batch_c = []
        batch_cond_indices = []
        batch_areas = []
        
        for p, cond_index in items_to_pop:
            batch_input_x.append(p.input_x)
            batch_mult.append(p.mult)
            batch_c.append(p.conditioning)
            batch_areas.append(p.area)
            batch_cond_indices.append(cond_index)

        batch_chunks = len(batch_cond_indices)
        input_x = torch.cat(batch_input_x)
        c = cond_cat(batch_c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_idx = batch_cond_indices[o]
            a = batch_areas[o]
            if a is None:
                out_conds[cond_idx] += output[o] * batch_mult[o]
                out_counts[cond_idx] += batch_mult[o]
            else:
                out_c = out_conds[cond_idx]
                out_cts = out_counts[cond_idx]
                dims = len(a) // 2
                for i in range(dims):
                    out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                    out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                out_c += output[o] * batch_mult[o]
                out_cts += batch_mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds

def resolve_areas_and_cond_masks_multidim(conditions: List[Dict[str, Any]], dims: Tuple[int, ...], device: torch.device):
    for i in range(len(conditions)):
        c = conditions[i]
        if 'area' in c:
            area = c['area']
            if area[0] == "percentage":
                modified = c.copy()
                a = area[1:]
                a_len = len(a) // 2
                new_area = []
                for d in range(len(dims)):
                    new_area.append(max(1, round(a[d] * dims[d])))
                for d in range(len(dims)):
                    new_area.append(round(a[d + a_len] * dims[d]))
                modified['area'] = tuple(new_area)
                conditions[i] = modified
                c = modified

        if 'mask' in c:
            mask = c['mask'].to(device=device)
            modified = c.copy()
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=dims, mode='bilinear', align_corners=False).squeeze(1)
            
            modified['mask'] = mask
            conditions[i] = modified

def calculate_start_end_timesteps(model: Any, conds: List[Dict[str, Any]]):
    s = getattr(model, "model_sampling", None)
    if s is None: return

    for t in range(len(conds)):
        x = conds[t]
        timestep_start = None
        timestep_end = None
        
        if 'start_percent' in x:
            timestep_start = s.percent_to_sigma(x['start_percent'])
        if 'end_percent' in x:
            timestep_end = s.percent_to_sigma(x['end_percent'])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if timestep_start is not None: n['timestep_start'] = timestep_start
            if timestep_end is not None: n['timestep_end'] = timestep_end
            conds[t] = n

def encode_model_conds(model_function: Callable, conds: List[Dict[str, Any]], noise: torch.Tensor, device: torch.device, prompt_type: str, **kwargs) -> List[Dict[str, Any]]:
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        
        if len(noise.shape) >= 4:
            params["width"] = params.get("width", noise.shape[3] * 8)
            params["height"] = params.get("height", noise.shape[2] * 8)
        
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k, v in kwargs.items():
            if k not in params:
                params[k] = v

        out = model_function(**params)
        x = x.copy()
        model_conds = x.get('model_conds', {}).copy()
        for k, v in out.items():
            model_conds[k] = v
        x['model_conds'] = model_conds
        conds[t] = x
    return conds

def process_conds(model: Any, noise: torch.Tensor, conds: Dict[str, Any], device: torch.device, latent_image: Optional[torch.Tensor] = None, denoise_mask: Optional[torch.Tensor] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, 'extra_conds'):
        for k in conds:
            conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    return conds
