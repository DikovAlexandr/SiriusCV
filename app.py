import streamlit as st
from PIL import Image
import numpy as np
import os
import io

from utils import load_image, transform_image, process_single_image, create_blurred_background, replace_background
from utils import generate_caption, correct_text, translate_to_russian, modify_description

import warnings
warnings.filterwarnings("ignore")

def main():
    st.title("Image Background Replacement")

    if 'final_image' not in st.session_state:
        st.session_state['final_image'] = None

    uploaded_file = st.file_uploader("Load image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        transformed_image = transform_image(original_image)
        st.image(original_image, caption='Original Image', use_column_width=True)

        st.subheader("Choose background color tone")
        color_palette = {
            "White": (255, 255, 255),
            "Gray": (180, 180, 180),
            "Red": (255, 0, 0),
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "Violet": (255, 0, 255),
            "Orange": (255, 165, 0),
            "Yellow": (255, 255, 0),
        }
        selected_color = st.selectbox("Choose a color:", list(color_palette.keys()))
        custom_color = st.color_picker("Or enter a custom color", "#FFFFFF")
        final_color = color_palette[selected_color] if selected_color else custom_color

        if st.button("Replace background"):
            _, object_image, background_image = process_single_image(transformed_image, 
                                                                     model_name="u2net", 
                                                                     threshold_cutoff=0.90)
            new_background = create_blurred_background(background_image, 
                                                       color=final_color, 
                                                       blur_strength=25)
            final_image = replace_background(object_image, new_background)

            st.session_state['final_image'] = final_image

    if st.session_state['final_image'] is not None:
        st.image(st.session_state['final_image'], caption="Final Image", use_column_width=True)

        img_buffer = io.BytesIO()
        st.session_state['final_image'].save(img_buffer, format="PNG")
        img_buffer.seek(0)

        st.download_button(
            label="Download Image",
            data=img_buffer,
            file_name="final_image.png",
            mime="image/png"
        )

    if st.session_state['final_image'] is not None:
        st.subheader("Generate description")
        
        if st.button("Short description"):
            description = generate_caption(original_image, 'default')
            ru_description = translate_to_russian(description)
            st.session_state['description'] = description
            st.session_state['ru_description'] = ru_description
            st.write(f"Description: {description}")
            st.write(f"Russian description: {ru_description}")

        if st.button("Advertising description"):
            description = generate_caption(original_image, 'advertising')
            ru_description = translate_to_russian(description)
            st.session_state['description'] = description
            st.session_state['ru_description'] = ru_description
            st.write(f"Advertising description: {description}")
            st.write(f"Russian advertising description: {ru_description}")

        if st.button("Detailed description"):
            description = generate_caption(original_image, 'detailed')
            ru_description = translate_to_russian(description)
            st.session_state['description'] = description
            st.session_state['ru_description'] = ru_description
            st.write(f"Detailed description: {description}")
            st.write(f"Russian detailed description: {ru_description}")

if __name__ == "__main__":
    main()
