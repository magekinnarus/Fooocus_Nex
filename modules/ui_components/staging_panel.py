import gradio as gr

def build_staging_tab(num_slots=4):
    """
    Builds the Staging tab: a side-panel for temporary image storage.
    
    Args:
        num_slots (int): Number of image slots to create. Default is 4 for stability.
    
    Returns:
        dict: Gradio components (staging_images).
    """
    staging_images = []
    
    with gr.Column() as staging_col:
        for i in range(num_slots):
            with gr.Row():
                img = gr.Image(
                    label=f"Slot {i}",
                    sources=['upload'],
                    type='filepath',
                    elem_id=f"staging_img_{i}",
                    show_label=False,
                    height=200
                )
                staging_images.append(img)
    
    return {
        'staging_images': staging_images,
        'staging_col': staging_col
    }
