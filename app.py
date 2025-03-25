import gradio as gr
import cv2
import numpy as np
from image_processing import segment_person, load_stereo_images, insert_person, create_anaglyph


def process_image(person_img, background_image, depth_level, size_percentage, horizontal_position, vertical_position):
    # Convert PIL images to OpenCV format np array
    person_np = cv2.cvtColor(np.array(person_img), cv2.COLOR_RGB2BGRA)

    # Segment person
    segmented_person_image = segment_person(person_np)

    # Load stereo images
    left_bg_image, right_bg_image = load_stereo_images(background_image)

    # Insert person into stereo images
    stereo_left, stereo_right = insert_person(left_bg_image, right_bg_image, segmented_person_image, depth_level, size_percentage, horizontal_position, vertical_position)

    # Create anaglyph
    anaglyph = create_anaglyph(stereo_left, stereo_right)

    return anaglyph


demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Person Image"),
        gr.Dropdown(["Steam clock", "Building", "Downtown", "Office", "Path", "Yard"], label="Background image", value="Steam clock"),
        gr.Radio(["Close", "Medium", "Far"], label="Depth Level", value="Medium"),
        gr.Slider(1, 100, value=50, step=1, label="Person Size Percentage",
                  info="Adjust the size percentage of the person in the image."),
        gr.Slider(0, 100, value=50, label="Horizontal Position Percentage",
                  info="Adjust the horizontal position percentage of the person in the image."),
        gr.Slider(0, 100, value=50, label="Vertical Position Percentage",
                  info="Adjust the vertical position percentage of the person in the image.")
    ],
    outputs=gr.Image(label="Generated Anaglyph Image"),
    title="3D Image Composer",
    description="Upload images to generate anaglyph 3D images."
)

demo.launch()
