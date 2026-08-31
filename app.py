import os
import uuid

import torch
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, FloatField, HiddenField
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# =========================================================
# Flask Application
# =========================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "neural-style-transfer-secret-key"
)


# =========================================================
# Project Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# Hugging Face Model Repository
# =========================================================

# Replace this with your actual Hugging Face repository.
HF_REPO_ID = "YOUR_USERNAME/neural-style-transfer-models"


VGG_FILENAME = "vgg_normalised.pth"
DECODER_FILENAME = "decoder_final.pth"


# =========================================================
# Model Paths
# =========================================================

VGG_PATH = os.path.join(MODEL_DIR, VGG_FILENAME)
DECODER_PATH = os.path.join(MODEL_DIR, DECODER_FILENAME)


def download_models():
    """
    Download model files from Hugging Face if they are not
    already available locally.
    """

    global VGG_PATH, DECODER_PATH

    # Download VGG model
    if not os.path.exists(VGG_PATH):

        print("Downloading VGG model from Hugging Face...")

        VGG_PATH = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=VGG_FILENAME,
            local_dir=MODEL_DIR
        )

        print("VGG model downloaded.")


    # Download Decoder model
    if not os.path.exists(DECODER_PATH):

        print("Downloading decoder model from Hugging Face...")

        DECODER_PATH = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=DECODER_FILENAME,
            local_dir=MODEL_DIR
        )

        print("Decoder model downloaded.")


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# =========================================================
# Download Models
# =========================================================

download_models()


# =========================================================
# Load Models
# =========================================================

print("Loading VGG encoder...")

encoder = VGGEncoder(VGG_PATH).to(device)

print("Loading decoder...")

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device
    )
)

encoder.eval()
decoder.eval()

print("Models loaded successfully.")


# =========================================================
# Upload Form
# =========================================================

class UploadForm(FlaskForm):

    content = FileField("Content Image")

    style = FileField("Style Image")

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField(
        "Transfer Style"
    )


# =========================================================
# Allowed File Extensions
# =========================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower() in ALLOWED_EXTENSIONS
    )


# =========================================================
# Style Transfer
# =========================================================

def style_transfer(
    content_image,
    style_image,
    alpha
):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    content_tensor = transform(
        content_image
    ).unsqueeze(0).to(device)

    style_tensor = transform(
        style_image
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        # Extract VGG features
        content_features = encoder(
            content_tensor,
            is_test=True
        )

        style_features = encoder(
            style_tensor,
            is_test=True
        )


        # Adaptive Instance Normalization
        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )


        # Blend original content and stylized features
        stylized_features = (
            alpha * stylized_features
            + (1 - alpha) * content_features
        )


        # Decode image
        stylized_image = decoder(
            stylized_features
        )


    return stylized_image


# =========================================================
# Save Image
# =========================================================

def save_image(image, path):

    image = image.detach().cpu()

    image = image.squeeze(0)

    image = image.clamp(
        0,
        1
    )

    image = transforms.ToPILImage()(
        image
    )

    image.save(path)


# =========================================================
# Home / Style Transfer Route
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    form = UploadForm()

    result_image = None

    content_filename = None

    style_filename = None

    error = None


    if form.validate_on_submit():

        # -------------------------------------------------
        # Content Image
        # -------------------------------------------------

        if (
            form.content.data
            and form.content.data.filename
        ):

            if not allowed_file(
                form.content.data.filename
            ):

                error = "Invalid content image format."

            else:

                original_name = secure_filename(
                    form.content.data.filename
                )

                # Unique filename prevents collisions
                content_filename = (
                    f"{uuid.uuid4().hex}_"
                    f"{original_name}"
                )

                content_path = os.path.join(
                    UPLOAD_DIR,
                    content_filename
                )

                form.content.data.save(
                    content_path
                )

                form.content_path.data = (
                    content_filename
                )

        else:

            content_filename = (
                form.content_path.data
            )


        # -------------------------------------------------
        # Style Image
        # -------------------------------------------------

        if not error:

            if (
                form.style.data
                and form.style.data.filename
            ):

                if not allowed_file(
                    form.style.data.filename
                ):

                    error = "Invalid style image format."

                else:

                    original_name = secure_filename(
                        form.style.data.filename
                    )

                    style_filename = (
                        f"{uuid.uuid4().hex}_"
                        f"{original_name}"
                    )

                    style_path = os.path.join(
                        UPLOAD_DIR,
                        style_filename
                    )

                    form.style.data.save(
                        style_path
                    )

                    form.style_path.data = (
                        style_filename
                    )

            else:

                style_filename = (
                    form.style_path.data
                )


        # -------------------------------------------------
        # Perform Style Transfer
        # -------------------------------------------------

        if (
            not error
            and content_filename
            and style_filename
        ):

            content_path = os.path.join(
                UPLOAD_DIR,
                content_filename
            )

            style_path = os.path.join(
                UPLOAD_DIR,
                style_filename
            )

            try:

                # Open images
                content_image = Image.open(
                    content_path
                ).convert("RGB")

                style_image = Image.open(
                    style_path
                ).convert("RGB")


                # Get alpha
                alpha = float(
                    form.alpha.data
                    if form.alpha.data is not None
                    else 1.0
                )


                # Keep alpha between 0 and 1
                alpha = max(
                    0.0,
                    min(
                        1.0,
                        alpha
                    )
                )


                # Generate stylized image
                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )


                # Result filename
                result_filename = (
                    "stylized_"
                    + content_filename
                )

                result_path = os.path.join(
                    UPLOAD_DIR,
                    result_filename
                )


                # Save result
                save_image(
                    stylized_image,
                    result_path
                )


                result_image = result_filename


            except Exception as e:

                print(
                    "Style transfer error:",
                    e
                )

                error = (
                    "An error occurred while "
                    "processing the images."
                )


        elif not error:

            if not content_filename:

                error = (
                    "Please upload a content image."
                )

            elif not style_filename:

                error = (
                    "Please upload a style image."
                )


    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# =========================================================
# Uploaded Images
# =========================================================

@app.route(
    "/uploads/<filename>"
)
def send_image(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )


# =========================================================
# Example Images
# =========================================================

@app.route(
    "/examples/<path:filename>"
)
def send_example(filename):

    return send_from_directory(
        EXAMPLES_DIR,
        filename
    )


# =========================================================
# Run Application
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
