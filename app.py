__all__ = ["is_cat", "learn", "categories", "classify_images", "image", "label", "examples", "intf"]

from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn  = load_learner('model.pkl')

categories = ('Grizzly', 'Teddy', 'Polar', 'Black')

def classify_image(img):
    predict,index,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['grizzly.jpg', 'black.jpg', 'polar.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)