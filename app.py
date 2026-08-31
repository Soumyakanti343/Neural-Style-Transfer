import os
import requests
import torch

from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# =========================================================
# BASE DIRECTORIES
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# MODEL PATHS
# =========================================================

VGG_PATH = os.path.join(MODEL_DIR, "vgg_normalised.pth")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_final.pth")


# =========================================================
# HUGGING FACE MODEL URLS
# =========================================================

HF_BASE_URL = (
    "https://huggingface.co/"
    "soumyacodes16/neural-style-transfer-model/"
    "resolve/main/"
)

VGG_URL = HF_BASE_URL + "vgg_normalised.pth"
DECODER_URL = HF_BASE_URL + "decoder_final.pth"


# =========================================================
# DOWNLOAD MODEL
# =========================================================

def download_model(url, destination):
    if os.path.exists(destination):
        print(f"Model already exists: {destination}")
        return

    print(f"Downloading model from Hugging Face...")
    print(url)

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(destination, "wb") as file:

        for chunk in response.iter_content(chunk_size=1024 * 1024):

            if chunk:

                file.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    percentage = downloaded * 100 / total_size
                    print(
                        f"\rDownloading: {percentage:.1f}%",
                        end=""
                    )

    print("\nDownload completed.")


# =========================================================
# DOWNLOAD REQUIRED MODELS
# =========================================================

download_model(VGG_URL, VGG_PATH)
download_model(DECODER_URL, DECODER_PATH)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg"
}

Bootstrap(app)


# =========================================================
# FORM
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
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading VGG encoder...")

encoder = VGGEncoder(
    VGG_PATH
).to(device)

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
# ALLOWED FILE
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


# =========================================================
# STYLE TRANSFER
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

        content_features = encoder(
            content_tensor,
            is_test=True
        )

        style_features = encoder(
            style_tensor,
            is_test=True
        )

        stylized_features = (
            adaptive_instance_normalization(
                content_features,
                style_features
            )
        )

        stylized_features = (
            alpha * stylized_features
            +
            (1 - alpha) * content_features
        )

        output = decoder(
            stylized_features
        )

    return output


# =========================================================
# SAVE IMAGE
# =========================================================

def save_image(image, path):

    image = image.cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)


# =========================================================
# HOME
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        # ---------------------------------------------
        # CONTENT IMAGE
        # ---------------------------------------------

        if (
            form.content.data
            and
            form.content.data.filename
        ):

            if allowed_file(
                form.content.data.filename
            ):

                content_filename = secure_filename(
                    form.content.data.filename
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


        # ---------------------------------------------
        # STYLE IMAGE
        # ---------------------------------------------

        if (
            form.style.data
            and
            form.style.data.filename
        ):

            if allowed_file(
                form.style.data.filename
            ):

                style_filename = secure_filename(
                    form.style.data.filename
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


        # ---------------------------------------------
        # STYLE TRANSFER
        # ---------------------------------------------

        if content_filename and style_filename:

            content_path = os.path.join(
                UPLOAD_DIR,
                content_filename
            )

            style_path = os.path.join(
                UPLOAD_DIR,
                style_filename
            )

            try:

                content_image = Image.open(
                    content_path
                ).convert("RGB")

                style_image = Image.open(
                    style_path
                ).convert("RGB")

                alpha = float(
                    form.alpha.data
                )

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )

                result_filename = (
                    "stylized_" + content_filename
                )

                result_path = os.path.join(
                    UPLOAD_DIR,
                    result_filename
                )

                save_image(
                    stylized_image,
                    result_path
                )

                result_image = result_filename

            except Exception as e:

                error = str(e)

        else:

            error = (
                "Please upload both content "
                "and style images."
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
# UPLOADED IMAGES
# =========================================================

@app.route("/uploads/<filename>")
def send_image(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )


# =========================================================
# EXAMPLE IMAGES
# =========================================================

@app.route("/examples/<path:filename>")
def send_example(filename):

    return send_from_directory(
        EXAMPLES_DIR,
        filename
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )
