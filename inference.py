import sys
import os
import numpy as np
from PIL import Image
import onnxruntime as ort

from food_ai.config import CATEGORIES, FRESHNESS_CLASSES
from food_ai.data import build_transforms

ROUTER = "models/router.onnx"


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")

    _, eval_transform = build_transforms()
    tensor = eval_transform(img).unsqueeze(0)

    return tensor.numpy().astype(np.float32)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_model(model, input_data):

    session = ort.InferenceSession(model)

    inp = session.get_inputs()[0]

    result = session.run(
        None,
        {
            inp.name: input_data
        }
    )

    return result[0][0]


def main(image):

    x = preprocess(image)

    print("INPUT SHAPE:", x.shape)
    print("INPUT MIN:", x.min())
    print("INPUT MAX:", x.max())
    print("INPUT MEAN:", x.mean())

    # router
    router_out = run_model(
    ROUTER,
    x
)

    print("Router logits:", router_out)

    router_prob = softmax(router_out)

    print("Router probs:", router_prob)

    router_prob = softmax(router_out)

    category_id = np.argmax(router_prob)

    category = CATEGORIES[category_id]


    print("Category:", category)
    print(
        "Router confidence:",
        round(float(router_prob[category_id])*100,2),
        "%"
    )


    # freshness
    model = f"models/freshness/{category}.onnx"

    fresh_out = run_model(
        model,
        x
    )

    fresh_prob = softmax(fresh_out)

    freshness_id = np.argmax(fresh_prob)


    print(
        "Freshness:",
        FRESHNESS_CLASSES[freshness_id]
    )

    print(
        "Confidence:",
        round(float(fresh_prob[freshness_id])*100,2),
        "%"
    )


if __name__ == "__main__":

    main(sys.argv[1])