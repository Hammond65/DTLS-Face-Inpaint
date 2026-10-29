import gradio as gr
import torch
from PIL import Image
import os.path as osp
from util.yaml import load_yaml
from util.unet import Unet
from util.dtls_model.dtls_fast import Trainer, DTLS
import cv2
from face_crop_plus import Cropper
import time
import numpy as np

example_data_dir = "assets/"
cropper = Cropper(face_factor=0.7, strategy="largest")
enhancer = Cropper(face_factor=0.65, enh_threshold=0.01, strategy="largest")

# Global variable to track current example path
current_example_path = None

css = """
.center-gallery .gr-gallery {
    display: flex;
    justify-content: center;
    align-items: center;
}
"""

# Function to resize the image to 256x256
def align_face(image, use_predefined_mask=False, desired_size=256):
    global current_example_path
    image = Image.fromarray(cropper.process_single(image)).resize((desired_size, desired_size))
    
    if use_predefined_mask and current_example_path:
        base_path, ext = osp.splitext(current_example_path)
        mask_path = f"{base_path}_mask{ext}"
        print("Loading mask from:", mask_path)
        
        # Check if mask file exists
        if osp.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)

            # Create RGBA array with inverted alpha
            rgba_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
            rgba_array[:, :, :3] = 0  # RGB channels (black)
            rgba_array[:, :, 3] = 255 - mask_array  # Inverted alpha: white->transparent, black->opaque

            mask_layer = Image.fromarray(rgba_array, 'RGBA')
        else:
            print(f"Mask file not found: {mask_path}")
            mask_layer = Image.new("RGBA", (desired_size, desired_size), (0, 0, 0, 0))
    else:
        mask_layer = Image.new("RGBA", (desired_size, desired_size), (0, 0, 0, 0))
    
    return {"composite": image,
        "background": image,
        "layers": [
            mask_layer,
        ],
        "active_layer": 1
    }

def update_example_path(image):
    global current_example_path
    # This function will be called when an example is selected
    # We need to determine which example was selected based on the image
    example_files = [
        osp.join(example_data_dir, "1.jpg"),
        osp.join(example_data_dir, "2.jpg"),
        osp.join(example_data_dir, "3.jpg"),
        osp.join(example_data_dir, "4.jpg"),
        osp.join(example_data_dir, "5.jpeg"),
    ]
    
    # Try to match the uploaded image with one of the examples
    if image is not None:
        for example_path in example_files:
            if osp.exists(example_path):
                example_image = Image.open(example_path)
                example_array = np.array(example_image)
                # Simple comparison - you might want to make this more robust
                if image.shape == example_array.shape and np.array_equal(image, example_array):
                    current_example_path = example_path
                    print(f"Matched example: {example_path}")
                    return
    
    # If no match found, clear the current example path
    current_example_path = None

def loop_images(images):
    """Loop through images and update the gallery."""
    for img in images:
        yield [img]
        time.sleep(0.5)
        
# Function to process the edited image and mask
def process_input(ims):
    input_image = Image.fromarray(cv2.resize(ims["composite"], (256,256), interpolation=cv2.INTER_NEAREST))
    mask = cv2.resize((255-ims["layers"][0][:,:,3]), (256,256), interpolation=cv2.INTER_NEAREST)
    mask[mask < 128] = 0
    mask[mask >= 128] = 255
    mask = Image.fromarray(mask)

    output, intermediates = trainer.test(input_image, mask)
    intermediates.append(output)
    return intermediates

# Creating the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]), css=css) as demo:
    gr.HTML('')
    gr.HTML('''
            <div style="display: flex; justify-content: flex-end; align-items: flex-start; padding: 10px;">
            <img src="/gradio_api/file=logo/SFU-99.svg" style="height: 25px; margin-left: 10px;" />
            <img src="/gradio_api/file=logo/polyu.png" style="height: 30px; margin-left: 10px;" />
            </div>
            <div style='display: flex; overflow: hidden; width: 100%;'>
                <div style='flex: 1; text-align: center;'>
                    <h1 style='font-size: 4em; line-height: 1.2;'> DTLS-Face-Inpaint </h1>
                    <h2>Domain Transfer in Latent Space for Face Inpainting 🎨🖌</h2>
                    <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <a href="https://github.com/Hammond65/DTLS" target="_blank">
                        <img src="https://badgen.net/badge/GitHub/To%20Be%20Announced/blue?icon=github">
                    </a>

                    &nbsp;
                    <a href="https://www.sfu.edu.hk/en/schools-and-offices/schools-and-departments/school-of-computing-and-information-sciences/introduction/index.html">
                        <img src="https://img.shields.io/badge/CIS-Saint%20\Francis%20University-red">
                    </a>
                    &nbsp;
                    <a href="https://www.polyu.edu.hk/eee/">
                        <img src="https://img.shields.io/badge/EEE-PolyU-yellow">
                    </a>
                </div>
            </div>
                <div style='flex: 1; text-align: left; padding-left: 12px;'>
                    <h3 style='color: #dc2626; font-weight: bold;'>Instructions</h3>
                    <ol style='color: #dc2626; font-weight: bold;'>
                        <li>Click on any examples</li>
                        <li>Press "Process Masked Image"</li>
                        <li>Wait for a few seconds, this is done by real time</li>
                        <li>(Optional) Self defined mask, you may define your mask by unchecking "Use Predefined Mask"</li>
                        <li>(Optional) Click the pen icon in the Draw Box to draw your own mask</li>
                        <li>(Optional) Press "Process Masked Image"</li>
                    </ol>
            </div>
            ''')
    
    with gr.Row():
        with gr.Accordion(open=True, label="Source Image"):
            image_state = gr.State()
            input_image = gr.Image(type="numpy", label="Upload Image", interactive=True, render=False)

            gr.Examples(
                examples=[
                    [osp.join(example_data_dir, "1.jpg")],
                    [osp.join(example_data_dir, "2.jpg")],
                    [osp.join(example_data_dir, "3.jpg")],
                    [osp.join(example_data_dir, "4.jpg")],
                    [osp.join(example_data_dir, "5.jpeg")],
                ],
                inputs=[input_image],
                cache_examples=False,
                examples_per_page=20,
            )
            
            process_button = gr.Button("Process Masked Image")

            with gr.Row():
                input_image.render()
                paint_image = gr.ImageMask(type="numpy", layers=True, label="Draw Box")

            use_predefined_checkbox = gr.Checkbox(label="Use Predefined Mask", value=True)

        with gr.Accordion(open=True, label="Output Images"):
            output_image = gr.Gallery(label="Prediction", interactive=False)
            gr.Markdown(
                """
                <div style='text-align: left; padding-top: 80px; padding-left: 10px; font-size: 10px;'>
                    <h5>Module Leader: Prof. Wan-Chi Siu, PhD, DIC, Life-FIEEE, Emeritus Professor (PolyU) and Research Professor (SFU)<br>Module Contributor: Mr. Hammond Lee, RA(SFU) and PhD Student (PolyU)<br>School Dean: Prof. Anthony Chan, PhD, Life-FIEEE, Dean(SCIS), SFU</h5>
                    <h5>Key references:
                    <br>    1. Hon Man Hammond Lee, Wan-Chi Siu, Ivan Wang-Hei Ho, and H. Anthony Chan, "DTLS-Face-Inpaint: A Novel Inpainting Framework with Latent Space Domain Transfer" Submitted to IEEE Transactions on Image Processing (TIP), 2025. (Under Review)
                    <br><br>Conference paper, an early version with much simplified approach:
                    <br>    2. Hon Man Hammond Lee, Wan-Chi Siu, "DTLS-Inpaint: Yet Another Efficient Image Inpainting with Domain Transfer," Proceedings, IEE International Conference on Image Processing (ICIP 2025). 14-17 September 2025, Anchorage, Alaska.</h5>
                </div>
                """
            )
    
    # Event handlers
    def handle_image_change(image, use_predefined_mask):
        update_example_path(image)
        return align_face(image, use_predefined_mask)
    
    input_image.change(
        fn=handle_image_change,
        inputs=[input_image, use_predefined_checkbox], 
        outputs=paint_image
    )
    
    # Update when checkbox changes
    use_predefined_checkbox.change(
        fn=lambda checkbox_val, img: align_face(img, checkbox_val) if img is not None else None,
        inputs=[use_predefined_checkbox, input_image],
        outputs=[paint_image]
    )
    
    process_button.click(fn=process_input, inputs=[paint_image], outputs=output_image).then(fn=loop_images, inputs=[output_image], outputs=[output_image])

if __name__ == '__main__':
    from pathlib import Path
    gr.set_static_paths(paths=[Path.cwd().absolute()/"logo"])
    config = load_yaml('./config/trainer/app.yaml')
    device = config.device if torch.cuda.is_available() else "cpu"
    print(device)
    model = Unet(**config.networks.refine_unet).to(device)
    print("Loading models")
    dtls = DTLS(
        model,
        **config.networks.refine_dtls,
        device=device,
    ).to(device)

    trainer = Trainer(
        dtls,
        **config.trainer,
        device=device,
    )
    demo.launch()
