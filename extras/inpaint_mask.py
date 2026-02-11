import numpy as np
import torch
import modules.config


def generate_mask_from_image(image: np.ndarray, mask_model: str = 'sam', extras=None,
                             sam_options: None = None) -> tuple[np.ndarray | None, int | None, int | None, int | None]:
    """
    Nex Refactor: Removed SAM and GroundingDINO bloat.
    Only keeps rembg (lazy-loaded) for basic background removal if requested.
    """
    dino_detection_count = 0
    sam_detection_count = 0
    sam_detection_on_mask_count = 0

    if image is None:
        return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    if extras is None:
        extras = {}

    if 'image' in image:
        image = image['image']

    if mask_model == 'sam':
        print("[Nex] SAM/GroundingDINO auto-masking has been removed to reduce bloat.")
        return None, 0, 0, 0

    # Lazy load rembg only if used
    from rembg import remove, new_session
    
    result = remove(
        image,
        session=new_session(mask_model, **extras),
        only_mask=True,
        **extras
    )

    return result, dino_detection_count, sam_detection_count, sam_detection_on_mask_count
