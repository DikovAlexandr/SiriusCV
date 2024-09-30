import os
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torchvision import transforms

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from model import U2NET
from model import U2NETP 

def normPRED(d):
    """
    Normalize the predicted SOD probability map.
    """
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def show_output(pred):
    """
    Show the predicted SOD probability map.
    """
    pred = pred.squeeze()
    pred_np = pred.cpu().data.numpy()

    plt.imshow(pred_np, cmap='gray')
    plt.axis('off')
    plt.show()

def load_image(image_path,
               transform=None):
    """
    Load image from path.
    """
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image).unsqueeze(0)
    return image

def transform_image(image):
    """
    Transform image to tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def process_single_image(input_image, 
                         model_name='u2net', 
                         threshold_cutoff=0.90):
    """
    Process single image using U2Net or U2NetP and return the object and background images.
    Return the predicted SOD probability map, the object and background images.
    """
    model_dir = f'saved_models/{model_name}/{model_name}.pth'

    if model_name == 'u2net':
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        net = U2NETP(3, 1)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    net.eval()

    input_image = input_image.type(torch.FloatTensor)
    if torch.cuda.is_available():
        input_image = Variable(input_image.cuda())
    else:
        input_image = Variable(input_image)

    d1, _, _, _, _, _, _ = net(input_image)

    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    pred_np = pred.detach().cpu().numpy()[0]
    pred_np = np.expand_dims(pred_np, axis=-1)

    original_img_np = input_image.squeeze().cpu().data.numpy().transpose(1, 2, 0)

    object_img = np.concatenate((original_img_np, pred_np), axis=-1).clip(0, 1)
    object_img = Image.fromarray((object_img * 255).astype(np.uint8), 'RGBA')

    background_img = (original_img_np * (1 - pred_np)).clip(0, 1)
    background_img = Image.fromarray((background_img * 255).astype(np.uint8))

    return pred, object_img, background_img

def create_blurred_background(original_image,
                              color=(240, 240, 240),
                              blur_strength=25):
    """
    Create a blurred background image using the original image and a solid color layer.
    """
    original_np = np.array(original_image)
    color_layer = np.ones_like(original_np) * np.array(color, dtype=np.uint8)
    blended_image = cv2.addWeighted(original_np, 0.3, color_layer, 0.7, 0)
    blurred_background = cv2.GaussianBlur(blended_image, 
                                          ksize=(blur_strength, blur_strength),
                                          sigmaX=0)

    return Image.fromarray(blurred_background)

def replace_background(object_img,
                       background_img):
    """
    Replace the object image with the background image.
    """
    object_img = object_img.resize(background_img.size)

    if object_img.mode != 'RGBA':
        object_img = object_img.convert('RGBA')

    final_image = Image.new("RGBA", background_img.size)
    final_image.paste(background_img.convert('RGBA'), (0, 0))

    mask = object_img.split()[-1]
    final_image.paste(object_img, (0, 0), mask)

    return final_image.convert("RGB")

def generate_caption(raw_image, 
                     style="default"):
    """
    Generate caption for an image using the Blip model.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    inputs = processor(raw_image, return_tensors="pt")

    if style == "default":
        prompt = "a photo of"
    elif style == "advertising":
        prompt = "a beautiful and stylish product"
    elif style == "detailed":
        prompt = "a highly detailed description of the product"
    else:
        prompt = "a photo of"
    
    text_input = processor.tokenizer(prompt, 
                                     return_tensors="pt").input_ids
    
    output = model.generate(**inputs, 
                            input_ids=text_input, 
                            max_length=100,
                            num_beams=5,
                            no_repeat_ngram_size=3,
                            early_stopping=True)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def correct_text(text):
    """
    Correct the text from BLIP using the T5 model.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    t5_model_name = "t5-base"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device) 

    input_ids = t5_tokenizer("Correct the description, make it sound nice, remove repetitions and errors: " + text, 
                             return_tensors="pt").input_ids.to(device)
    outputs = t5_model.generate(input_ids, max_length=100)
    corrected_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def translate_to_russian(text):
    """
    Translate the text from English to Russian.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    translation_model_name = "Helsinki-NLP/opus-mt-en-ru"
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

    inputs = translation_tokenizer(text, return_tensors="pt", padding=True).to(device)
    translated_tokens = translation_model.generate(**inputs, max_length=100)
    translated_text = translation_tokenizer.decode(translated_tokens[0], 
                                                   skip_special_tokens=True)
    return translated_text

def modify_description(simple_description, 
                          style="detailed"):
    """
    Modify the description using the T5 model.
    """
    if style == "detailed":
        prompt = f"Expand and provide more detailed information about this image, focusing on important aspects: {simple_description}"
    elif style == "advertising":
        prompt = f"Write a compelling and catchy advertisement for this product, highlighting its uniqueness and value: {simple_description}"
    else:
        raise ValueError("Unsupported style. Choose either 'detailed' or 'advertising'.")

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        inputs,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text