---
title: 3D_Image_Composer
app_file: app.py
sdk: gradio
sdk_version: 4.44.1
---
# 3D Image Composer

## Overview
This project generates stereoscopic 3D images by overlaying a segmented person onto left and right background images with adjustable position, scale, and depth-based disparity. It creates an anaglyph image for red-cyan 3D visualization.

## Demo
[Gradio link](https://huggingface.co/spaces/zoeyziz/3D_Image_Composer)

## Features
1. **Select Background Image**

The project provides a list of stereoscopic background images split into left and right views for display.

2. **Adjust Depth Level**

Control the disparity between left and right images to simulate different distances (close, medium, far).

3. **Person Image Insertion**

Overlay a segmented person onto the background while maintaining transparency (RGBA format).

4. **Adjust Person Position**

Set the horizontal and vertical placement using percentage-based coordinates.

5. **Scale Person Image**

Resize the person relative to the background using a percentage-based scaling factor.