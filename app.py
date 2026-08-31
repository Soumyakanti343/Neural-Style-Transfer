import os
import uuid

import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

VGG_PATH = os.path.join(MODEL_DIR, "vgg_normalised.pth")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_final.pth")


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg",}

Bootstrap(app)
os.makedirs(UPLOAD_DIR, exist_ok = True)


class UploadForm(FlaskForm) :
    content = FileField("Content Image", validators = [InputRequired()])
    style = FileField("Style Image", validators = [InputRequired()])
    alpha = FloatField("Alpha", default = 1.0)
    submit = SubmitField("Transfer Style")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")


if not os.path.exists(VGG_PATH) :
    raise FileNotFoundError(
        f"VGG model not found : {VGG_PATH}"
    )

if not os.path.exists(DECODER_PATH) :
    raise FileNotFoundError(
        f"Decoder model not found : {DECODER_PATH}"
    )


print("Loading VGG encoder...")

encoder = VGGEncoder(VGG_PATH).to(device)

print("Loading decoder...")

decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location = device, weights_only = True))

encoder.eval()
decoder.eval()

print("Models loaded successfully.")


def allowed_file(filename) :
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def create_unique_filename(filename) :
    filename = secure_filename(filename)
    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return f"{uuid.uuid4().hex}.{extension}"



def style_transfer(content_image, style_image, alpha) :
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad() :
        content_features = encoder(content_tensor, is_test = True)
        style_features = encoder(style_tensor, is_test = True)
        target_features = adaptive_instance_normalization(content_features, style_features)
        target_features = (alpha * target_features + (1 - alpha) * content_features)
        output = decoder(target_features)

    return output


def save_tensor_as_image(tensor, path) :
    image = tensor.detach().cpu()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)

    image.save(path)



@app.route("/", methods = ["GET", "POST"])
def index() :
    form = UploadForm()
    result_image = None
    content_image = None
    style_image = None
    error = None

    if form.validate_on_submit() :
        content_file = form.content.data
        style_file = form.style.data

        if not allowed_file(content_file.filename) :
            error = "Invalid content image format."

        elif not allowed_file(style_file.filename) :
            error = "Invalid style image format."

        else :
            try :
                content_filename = (create_unique_filename(content_file.filename))
                style_filename = (create_unique_filename(style_file.filename))

                content_path = os.path.join(app.config["UPLOAD_FOLDER"], content_filename)
                style_path = os.path.join(app.config["UPLOAD_FOLDER"], style_filename)
                content_file.save(content_path)
                style_file.save(style_path)
                
                content = Image.open(content_path).convert("RGB")
                style = Image.open(style_path).convert("RGB")

                alpha = float(form.alpha.data)
                alpha = max(0.0, min(1.0, alpha))
                output = style_transfer(content, style, alpha)
                result_filename = ("stylized_" + content_filename)
                result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
                save_tensor_as_image(output, result_path)

                content_image = content_filename
                style_image = style_filename
                result_image = result_filename

            except Exception as e :
                print(f"Style transfer error: {e}")
                error = (
                    "An error occurred while "
                    "processing the images."
                )

    return render_template(
        "index.html",
        form = form,
        result_image = result_image,
        content_image = content_image,
        style_image = style_image,
        error = error
    )



@app.route(
    "/uploads/<filename>"
)
def send_image(filename) :
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )



@app.route("/examples/<path:filename>")
def send_example(filename) :
    return send_from_directory(EXAMPLES_DIR, filename)



@app.route("/health")
def health() :
    return {"status" : "healthy", "device" : str(device)}


if __name__ == "__main__" :
    app.run(host = "0.0.0.0", port = int(os.environ.get("PORT", 5000)), debug = False)




    
