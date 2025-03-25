import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image


background_images = {
    "Steam clock": "background_images/steam_clock.jpg",
    "Building": "background_images/building.jpg",
    "Downtown": "background_images/downtown.jpg",
    "Office": "background_images/office.jpg",
    "Path": "background_images/path.jpg",
    "Yard": "background_images/yard.jpg"
}


def segment_person(image_np):
    # Load DeepLabV3 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    # Transformation for input image
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert NumPy array (OpenCV format) to PIL Image
    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)['out'][0]

    # Generate mask
    mask = output.argmax(0).byte().cpu().numpy()
    person_mask = (mask == 15).astype(np.uint8) * 255

    # Apply mask to image
    masked_img = cv2.bitwise_and(image_np, image_np, mask=person_mask)

    # Add alpha channel
    b, g, r, a = cv2.split(masked_img)
    alpha = person_mask
    segmented_person_image = cv2.merge([b, g, r, alpha])
    segmented_person_image = cv2.cvtColor(segmented_person_image, cv2.COLOR_BGRA2RGBA)

    return segmented_person_image


def load_stereo_images(image):
    # Load the full stereoscopic image
    image_path = background_images[image]
    stereo_image = cv2.imread(image_path)

    # Split the stereo image into left and right images
    height, width, _ = stereo_image.shape
    if width % 2 != 0:
        raise ValueError("The image width must be even to split it into left and right images.")
    mid = width // 2
    left_image = stereo_image[:, :mid]
    right_image = stereo_image[:, mid:]
    left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

    return left_image_rgb, right_image_rgb


def insert_person(stereo_left, stereo_right, person_img, depth_level, size_percentage, horizontal_position, vertical_position):
    # Get the depth level
    disparities = {"Close": 30, "Medium": 15, "Far": 5}
    disparity = disparities[depth_level]

    # Convert stereo images to RGBA
    stereo_left = cv2.cvtColor(stereo_left, cv2.COLOR_RGB2RGBA)
    stereo_right = cv2.cvtColor(stereo_right, cv2.COLOR_RGB2RGBA)
    stereo_left[:, :, 3] = 255
    stereo_right[:, :, 3] = 255

    # Get dimensions of the person image
    ph, pw, _ = person_img.shape

    # Get dimensions of the background image
    bh, bw, _ = stereo_left.shape

    # # Resize person image based on percentage
    p_ratio = pw / ph
    bg_ratio = bw / bh
    if p_ratio > bg_ratio:
        resize_pw = int(bw * size_percentage / 100)
        resize_ph = int(resize_pw / p_ratio)
    else:
        resize_ph = int(bh * size_percentage / 100)
        resize_pw = int(resize_ph * p_ratio)
    resize_person_img = cv2.resize(person_img, (resize_pw, resize_ph))

    # Compute dynamic x, y positions based on percentage
    x = int((bw-resize_pw-disparity) * horizontal_position/100)
    y = int((bh-resize_ph) * vertical_position/100)

    # Place person with offset in right image
    stereo_left = overlay_person_on_background(stereo_left, resize_person_img, x, y)
    stereo_right = overlay_person_on_background(stereo_right, resize_person_img, x+disparity, y)

    return stereo_left, stereo_right


def overlay_person_on_background(background_img, person_img, x, y):
    # Get dimensions of the person image
    h, w, _ = person_img.shape

    # Get the overlay region
    roi = background_img[y:y+h, x:x+w]

    # Separate the person image into RGBA channels
    person_rgb = person_img[:, :, :3]
    person_alpha = person_img[:, :, 3]

    # Separate the overlay region of background image into RGB channels
    background_rgb = roi[:, :, :3]
    background_alpha = roi[:, :, 3]

    # Normalize alpha to range [0, 1]
    alpha = person_alpha / 255.0

    # Blend images
    blended_rgb = (alpha[:, :, np.newaxis] * person_rgb) + ((1 - alpha[:, :, np.newaxis]) * background_rgb)
    blended_alpha = np.maximum(person_alpha, background_alpha)

    # Combine the blended RGB and alpha
    blended_image = np.dstack((blended_rgb.astype(np.uint8), blended_alpha))

    # Place the blended image back onto the background
    background_img[y:y+h, x:x+w] = blended_image

    return background_img


def create_anaglyph(stereo_left, stereo_right):
    # Extract Red channel from the left image
    left_r = stereo_left[:, :, 0]

    # Extract Green and Blue channels from the right image
    right_g = stereo_right[:, :, 1]
    right_b = stereo_right[:, :, 2]

    # Create an empty anaglyph image
    anaglyph = np.zeros_like(stereo_left)

    # Assign Red from left, Green and Blue from right
    anaglyph[:, :, 0] = left_r
    anaglyph[:, :, 1] = right_g
    anaglyph[:, :, 2] = right_b
    anaglyph[:, :, 3] = stereo_left[:, :, 3]

    return anaglyph
