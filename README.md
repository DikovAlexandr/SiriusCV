# SiriusCV
 
## Image Background Replacement and Description Generation

This project provides a simple web interface using **Streamlit** for background replacement in images and generation of detailed, advertising-style, or concise descriptions of the images. The tool allows users to upload an image, replace its background with custom colors, generate descriptions in English and Russian, and correct the generated text using a pretrained **T5** model.

### Features:
- Upload an image (JPG/PNG/JPEG formats).
- Remove the background and replace it with a custom or predefined color.
- Blur the background for a studio-like effect.
- Generate short, detailed, or advertising descriptions of the image.
- Translate descriptions to Russian.
- Correct the generated text using T5-based language model.
- Download the final image with the replaced background.

---

### Installation Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/image-background-replacement.git
   cd image-background-replacement
   ```

2. **Set up a Python environment**:

   Make sure you have Python 3.8 or higher installed. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   Install the necessary libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support**:

   For GPU support, make sure you have CUDA installed. Install PyTorch with CUDA by running the following command (replace `cu117` with your CUDA version):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

   For other versions, visit [PyTorch official website](https://pytorch.org/get-started/locally/).

5. **Download U2Net model weights**:

   The U2Net model is used for background removal. Download the pretrained weights from the following link:

   - [U2Net Pretrained Weights](https://drive.google.com/uc?export=download&id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy)

   After downloading the weights, place the file in the following directory:

   ```
   saved_models/u2net/u2net.pth
   ```

   Create the directories if they do not exist.

6. **Run the Streamlit app**:

   Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

   This will launch the application in your default web browser.

---

### Usage

1. **Upload Image**: Click "Load image" and upload an image from your local files.
2. **Replace Background**: Choose a color for the background (either from a palette or a custom color) and click "Replace background."
3. **Download Image**: Once the image is processed, click "Download Image" to save the image with the replaced background.
4. **Generate Descriptions**: Click on the buttons to generate short, advertising, or detailed descriptions of the image.
5. **Text Correction**: Once a description is generated, click the "Исправить текст" button to correct any errors or repetitions in the text.

---

### Requirements

- Python 3.8+
- PyTorch (with CUDA support for GPU acceleration)
- Streamlit
- PIL (Python Imaging Library)
- U2Net model weights
- Transformers (for T5 and BLIP models)
  
---

### Project Structure

```
image-background-replacement/
│
├── app.py                    # Main Streamlit application
├── requirements.txt           # List of required Python packages
├── utils.py                   # Utility functions for image processing and text generation
├── saved_models/              # Directory to store model weights
│   └── u2net/                 # Directory for U2Net model weights
│       └── u2net.pth          # Pretrained U2Net weights for background removal
└── README.md                  # This README file
```

---

### License

This project is open-source and available under the [MIT License](LICENSE).

---

### Troubleshooting

- **Torch CUDA support**: If you encounter issues with GPU acceleration, ensure that CUDA is installed correctly and that you're using a compatible version of PyTorch.
- **Missing U2Net weights**: Ensure that the U2Net model weights are downloaded and placed in the `saved_models/u2net/` directory.

For further assistance, feel free to open an issue on the project's GitHub page.

--- 

### Credits

This project utilizes the following open-source models and libraries:
- [U2Net](https://github.com/xuebinqin/U-2-Net) for background removal.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for text generation with T5 and BLIP.
