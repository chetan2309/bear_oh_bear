__all__ = ["is_cat", "learn", "categories", "classify_images", "image", "label", "examples", "intf"]

from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn  = load_learner('model.pkl')

#categories = ('Grizzly', 'Teddy', 'Polar', 'Black')

labels = learn.dls.vocab
def classify_image(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label(num_top_classes=4)
#examples = ['grizzly.jpeg', 'teddy.jpeg', 'polar.jpeg', 'black.jpeg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(inline=False)