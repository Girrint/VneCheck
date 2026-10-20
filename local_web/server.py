import argparse
import csv
import json
import math
import mimetypes
import random
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from torchvision.datasets.folder import IMG_EXTENSIONS

from food_ai.config import (
    ALLOWED_FRESHNESS_BY_CATEGORY,
    CATEGORIES,
    CATEGORY_LABELS,
    DATA_DIR,
    EVAL_RESIZE_SIZE,
    FOOD_STATUS_CLASSES,
    FRESHNESS_ANCHORS,
    FRESHNESS_CLASSES,
    FRESHNESS_LABELS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    UNIFIED_ONNX_MODEL,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_TEST_ROOT = PROJECT_ROOT / DATA_DIR / "test"
DEFAULT_NEGATIVE_TEST_ROOT = PROJECT_ROOT / "data_negative" / "test"
DEFAULT_MODEL = PROJECT_ROOT / UNIFIED_ONNX_MODEL
DEFAULT_UNIFIED_ROOT = PROJECT_ROOT / "models" / "unified"
DEFAULT_LEGACY_ROOT = PROJECT_ROOT / "models" / "legacy" / "original"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "web_evaluations"
MODEL_MODES = ("both", "unified", "legacy")

REPORT_FIELDS = [
    "image_path", "truth_food_status", "truth_category", "truth_freshness",
    "truth_product",
    "unified_available", "unified_model_id", "unified_model_name",
    "unified_architecture", "unified_food_status", "unified_food_confidence",
    "unified_category", "unified_category_confidence", "unified_freshness",
    "unified_freshness_confidence", "unified_score",
    "unified_food_correct", "unified_category_correct", "unified_freshness_correct",
    "unified_joint_correct",
    *[f"unified_food_probability_{name}" for name in FOOD_STATUS_CLASSES],
    *[f"unified_category_probability_{name}" for name in CATEGORIES],
    *[f"unified_freshness_probability_{name}" for name in FRESHNESS_CLASSES],
    "legacy_available", "legacy_router_category",
    "legacy_router_confidence", "legacy_pipeline_freshness",
    "legacy_pipeline_freshness_confidence", "legacy_pipeline_score",
    "legacy_pipeline_expert_classes", "legacy_category_correct",
    "legacy_freshness_correct", "legacy_joint_correct",
    *[f"legacy_router_probability_{name}" for name in CATEGORIES],
    *[f"legacy_pipeline_probability_{name}" for name in FRESHNESS_CLASSES],
    "legacy_oracle_category", "legacy_oracle_freshness",
    "legacy_oracle_confidence", "legacy_oracle_score",
    "legacy_oracle_expert_classes", "legacy_oracle_correct",
    *[f"legacy_oracle_probability_{name}" for name in FRESHNESS_CLASSES],
]


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def _resize_shorter_side(image, size):
    width, height = image.size
    if width <= height:
        new_width = size
        new_height = int(height * size / width)
    else:
        new_height = size
        new_width = int(width * size / height)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def preprocess_pil_image(source):
    image = ImageOps.exif_transpose(source).convert("RGB")
    image = _resize_shorter_side(image, EVAL_RESIZE_SIZE)
    left = (image.width - IMAGE_SIZE) // 2
    top = (image.height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def preprocess_image(path):
    with Image.open(path) as source:
        return preprocess_pil_image(source)


def sample_metadata(relative):
    relative = PurePosixPath(relative)
    if len(relative.parts) < 4:
        return None
    category, freshness, product = relative.parts[:3]
    if category not in CATEGORIES or freshness not in FRESHNESS_CLASSES:
        return None
    return {
        "path": relative.as_posix(),
        "dataset": "supported",
        "food_status": "SUPPORTED",
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "freshness": freshness,
        "freshness_label": FRESHNESS_LABELS[freshness],
        "product": product,
        "filename": relative.name,
    }


def negative_sample_metadata(relative):
    relative = PurePosixPath(relative)
    if len(relative.parts) < 2:
        return None
    food_status = relative.parts[0]
    if food_status not in {"UNSUPPORTED", "NON_FOOD"}:
        return None
    return {
        "path": relative.as_posix(),
        "dataset": "negative",
        "food_status": food_status,
        "category": "",
        "category_label": "",
        "freshness": "",
        "freshness_label": "",
        "product": "",
        "filename": relative.name,
    }


def index_test_images(test_root, negative_test_root=None):
    test_root = Path(test_root).resolve()
    samples = []
    extensions = {suffix.lower() for suffix in IMG_EXTENSIONS}
    sources = [(test_root, sample_metadata)]
    if negative_test_root is not None:
        sources.append((Path(negative_test_root).resolve(), negative_sample_metadata))
    for root, metadata_parser in sources:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            relative = path.relative_to(root)
            metadata = metadata_parser(relative.as_posix())
            if metadata:
                samples.append(metadata)
    return samples


class OnnxRuntimeModel:
    def __init__(self, model_path, manifest_path=None, model_id=None):
        self.model_path = Path(model_path).resolve()
        self.manifest_path = (
            False
            if manifest_path is False
            else (Path(manifest_path).resolve() if manifest_path else None)
        )
        self.model_id = model_id or self.model_path.stem
        self.display_name = self.model_path.stem
        self.architecture = "unknown"
        self.metadata = {}
        self.session = None
        self.error = None
        self.food_gate_ready = False
        self._load_manifest()
        self._load_session()

    def _load_manifest(self):
        if self.manifest_path is False:
            return
        manifest_path = self.manifest_path
        if manifest_path is None:
            sidecar = self.model_path.with_suffix(".manifest.json")
            manifest_path = (
                sidecar if sidecar.is_file() else self.model_path.parent / "manifest.json"
            )
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.metadata = manifest
            self.model_id = str(manifest.get("artifact_id") or self.model_id)
            self.display_name = str(
                manifest.get("display_name")
                or manifest.get("run_name")
                or f"{manifest.get('architecture', 'unified')} · {self.model_path.stem}"
            )
            self.architecture = str(manifest.get("architecture") or "unknown")
            self.food_gate_ready = bool(manifest.get("food_gate_ready", False))
        except (OSError, ValueError) as error:
            self.error = f"Không đọc được manifest: {error}"

    def _load_session(self):
        if not self.model_path.is_file():
            self.error = f"Chưa có ONNX: {self.model_path}"
            return
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            if len(self.session.get_outputs()) != 3:
                raise ValueError("Unified ONNX phải có đúng ba output")
        except Exception as error:  # ONNX Runtime exposes several exception types.
            self.session = None
            self.error = f"Không tải được ONNX: {error}"

    @property
    def ready(self):
        return self.session is not None

    def descriptor(self):
        return {
            "id": self.model_id,
            "display_name": self.display_name,
            "architecture": self.architecture,
            "path": str(self.model_path),
            "ready": self.ready,
            "error": self.error,
            "food_gate_ready": self.food_gate_ready,
            "run_name": self.metadata.get("run_name"),
            "training_method": self.metadata.get("training_method"),
            "selection_metric": self.metadata.get("selection_metric"),
            "epoch": self.metadata.get("epoch"),
            "best_epoch": self.metadata.get("best_epoch"),
        }

    def predict_batch(self, paths):
        if not self.ready:
            raise RuntimeError(self.error or "ONNX chưa sẵn sàng")
        inputs = np.stack([preprocess_image(path) for path in paths])
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: inputs})
        return [
            self._decode(food, category, freshness)
            for food, category, freshness in zip(*outputs)
        ]

    def _decode(self, food_logits, category_logits, freshness_logits):
        food_values = _softmax(food_logits)
        food_index = int(np.argmax(food_values))
        food_status = FOOD_STATUS_CLASSES[food_index]

        category_values = _softmax(category_logits)
        category_index = int(np.argmax(category_values))
        category = CATEGORIES[category_index]

        allowed = set(ALLOWED_FRESHNESS_BY_CATEGORY[category])
        masked = np.asarray(freshness_logits, dtype=np.float64).copy()
        for index, name in enumerate(FRESHNESS_CLASSES):
            if name not in allowed:
                masked[index] = -math.inf
        freshness_values = _softmax(masked)
        freshness_index = int(np.argmax(freshness_values))
        freshness = FRESHNESS_CLASSES[freshness_index]
        score = sum(
            freshness_values[index] * FRESHNESS_ANCHORS[name]
            for index, name in enumerate(FRESHNESS_CLASSES)
            if name in allowed
        )
        return {
            "food_status": food_status if self.food_gate_ready else "NOT_TRAINED",
            "food_confidence": (
                round(float(food_values[food_index]), 4)
                if self.food_gate_ready
                else None
            ),
            "food_gate_ready": self.food_gate_ready,
            "category": category,
            "category_label": CATEGORY_LABELS[category],
            "category_confidence": round(float(category_values[category_index]), 4),
            "freshness": freshness,
            "freshness_label": FRESHNESS_LABELS[freshness],
            "freshness_confidence": round(
                float(freshness_values[freshness_index]), 4
            ),
            "estimated_freshness_score": round(float(score), 2),
            "probabilities": {
                "food": {
                    name: round(float(food_values[index]), 4)
                    for index, name in enumerate(FOOD_STATUS_CLASSES)
                },
                "category": {
                    name: round(float(category_values[index]), 4)
                    for index, name in enumerate(CATEGORIES)
                },
                "freshness": {
                    name: round(float(freshness_values[index]), 4)
                    for index, name in enumerate(FRESHNESS_CLASSES)
                    if name in allowed
                },
            },
        }


class LegacyOnnxRuntimeModel:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.router_path = self.root / "router.onnx"
        self.manifest_path = self.root / "manifest.json"
        self.sessions = {}
        self.error = None
        self.category_classes = list(CATEGORIES)
        self.freshness_classes = {
            category: list(ALLOWED_FRESHNESS_BY_CATEGORY[category])
            for category in CATEGORIES
        }
        self._load_manifest()
        self._validate_files()

    def _load_manifest(self):
        if not self.manifest_path.is_file():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            checkpoints = manifest["checkpoints"]
            self.category_classes = list(checkpoints["router"]["classes"])
            for category in CATEGORIES:
                self.freshness_classes[category] = list(
                    checkpoints[f"freshness/{category}"]["classes"]
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.error = f"Legacy manifest không hợp lệ: {error}"

    def _validate_files(self):
        missing = []
        if not self.router_path.is_file():
            missing.append(str(self.router_path))
        for category in CATEGORIES:
            path = self.root / "freshness" / f"{category}.onnx"
            if not path.is_file():
                missing.append(str(path))
        if missing:
            self.error = "Thiếu legacy ONNX: " + ", ".join(missing)

    @property
    def ready(self):
        return self.error is None

    def _session(self, key, path):
        if key not in self.sessions:
            self.sessions[key] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        return self.sessions[key]

    @staticmethod
    def _run(session, inputs):
        input_name = session.get_inputs()[0].name
        return session.run(None, {input_name: inputs})[0]

    def predict_batch(self, paths, truth_categories):
        if not self.ready:
            raise RuntimeError(self.error or "Legacy ONNX chưa sẵn sàng")
        inputs = np.stack([preprocess_image(path) for path in paths])
        router = self._session("router", self.router_path)
        router_logits = self._run(router, inputs)
        router_probabilities = [_softmax(logits) for logits in router_logits]
        predicted_categories = [
            self.category_classes[int(np.argmax(values))]
            for values in router_probabilities
        ]

        required = {category: set() for category in CATEGORIES}
        for index, (predicted, truth) in enumerate(
            zip(predicted_categories, truth_categories)
        ):
            required[predicted].add(index)
            required[truth].add(index)

        expert_outputs = {}
        for category, indexes in required.items():
            if not indexes:
                continue
            ordered = sorted(indexes)
            expert = self._session(
                f"freshness/{category}",
                self.root / "freshness" / f"{category}.onnx",
            )
            logits = self._run(expert, inputs[ordered])
            for index, values in zip(ordered, logits):
                expert_outputs[(category, index)] = self._decode_freshness(
                    category, values
                )

        results = []
        for index, (predicted, truth, category_values) in enumerate(
            zip(predicted_categories, truth_categories, router_probabilities)
        ):
            category_index = self.category_classes.index(predicted)
            pipeline = {
                "category": predicted,
                "category_label": CATEGORY_LABELS[predicted],
                "category_confidence": round(
                    float(category_values[category_index]), 4
                ),
                "category_probabilities": {
                    name: round(float(category_values[position]), 4)
                    for position, name in enumerate(self.category_classes)
                },
                **expert_outputs[(predicted, index)],
            }
            oracle = {
                "category": truth,
                "category_label": CATEGORY_LABELS[truth],
                **expert_outputs[(truth, index)],
            }
            results.append({"pipeline": pipeline, "oracle": oracle})
        return results

    def _decode_freshness(self, category, logits):
        classes = self.freshness_classes[category]
        values = _softmax(logits)
        index = int(np.argmax(values))
        freshness = classes[index]
        score = sum(
            values[position] * FRESHNESS_ANCHORS[name]
            for position, name in enumerate(classes)
        )
        return {
            "freshness": freshness,
            "freshness_label": FRESHNESS_LABELS[freshness],
            "freshness_confidence": round(float(values[index]), 4),
            "estimated_freshness_score": round(float(score), 2),
            "freshness_probabilities": {
                name: round(float(values[position]), 4)
                for position, name in enumerate(classes)
            },
            "class_count": len(classes),
        }


def manifest_for_onnx(path, preferred_path):
    sidecar = path.with_suffix(".manifest.json")
    if sidecar.is_file():
        return sidecar
    legacy = path.parent / "manifest.json"
    if not legacy.is_file():
        return None
    if path == preferred_path:
        return legacy
    try:
        payload = json.loads(legacy.read_text(encoding="utf-8"))
        declared = Path(payload.get("onnx", {}).get("path", "")).name
        return legacy if declared == path.name else None
    except (OSError, ValueError, TypeError):
        return None


def discover_unified_models(root, preferred_path):
    root = Path(root).resolve()
    preferred_path = Path(preferred_path).resolve()
    paths = set(root.rglob("*.onnx")) if root.is_dir() else set()
    if preferred_path.is_file():
        paths.add(preferred_path)

    models = {}
    default_id = None
    for path in sorted(paths, key=lambda item: str(item).lower()):
        try:
            relative_id = path.relative_to(root).with_suffix("").as_posix()
        except ValueError:
            relative_id = path.stem
        model = OnnxRuntimeModel(
            path,
            manifest_for_onnx(path, preferred_path) or False,
            relative_id,
        )
        model_id = model.model_id
        if model_id in models:
            model_id = relative_id
            model.model_id = model_id
        if model_id in models:
            raise ValueError(f"Trùng model id trong ONNX registry: {model_id}")
        models[model_id] = model
        if path == preferred_path:
            default_id = model_id

    if default_id is None and models:
        default_id = next(iter(models))
    return models, default_id


class LocalEvaluationApp:
    def __init__(
        self,
        test_root,
        model_path,
        legacy_root=DEFAULT_LEGACY_ROOT,
        report_root=DEFAULT_REPORT_ROOT,
        negative_test_root=DEFAULT_NEGATIVE_TEST_ROOT,
        unified_root=DEFAULT_UNIFIED_ROOT,
    ):
        self.test_root = Path(test_root).resolve()
        self.negative_test_root = Path(negative_test_root).resolve()
        self.sample_roots = {
            "supported": self.test_root,
            "negative": self.negative_test_root,
        }
        self.samples = index_test_images(
            self.test_root, self.negative_test_root
        )
        self.by_path = {sample["path"]: sample for sample in self.samples}
        self.unified_root = Path(unified_root).resolve()
        self.unified_models, self.default_unified_model_id = discover_unified_models(
            self.unified_root, model_path
        )
        if not self.unified_models:
            fallback = OnnxRuntimeModel(model_path)
            self.unified_models[fallback.model_id] = fallback
            self.default_unified_model_id = fallback.model_id
        self.model = self.unified_models[self.default_unified_model_id]
        self.legacy_model = LegacyOnnxRuntimeModel(legacy_root)
        self.report_root = Path(report_root).resolve()

    def resolve_sample_path(self, relative_path):
        normalized = PurePosixPath(str(relative_path)).as_posix()
        sample = self.by_path.get(normalized)
        if sample is None:
            raise FileNotFoundError("Ảnh không thuộc tập test đã lập chỉ mục")
        root = self.sample_roots[sample["dataset"]]
        target = (root / Path(*PurePosixPath(normalized).parts)).resolve()
        if root not in target.parents or not target.is_file():
            raise FileNotFoundError("Không tìm thấy ảnh test")
        return target

    def resolve_report_path(self, filename):
        if not filename or Path(filename).name != filename:
            raise FileNotFoundError("Tên report không hợp lệ")
        if Path(filename).suffix.lower() not in {".csv", ".json"}:
            raise FileNotFoundError("Định dạng report không hợp lệ")
        target = (self.report_root / filename).resolve()
        if self.report_root not in target.parents or not target.is_file():
            raise FileNotFoundError("Không tìm thấy report")
        return target

    @staticmethod
    def validate_model_mode(model_mode):
        if model_mode not in MODEL_MODES:
            raise ValueError(
                f"Model mode không hợp lệ: {model_mode}; chọn {MODEL_MODES}"
            )

    def unified_model(self, model_id=None):
        selected_id = model_id or self.default_unified_model_id
        model = self.unified_models.get(selected_id)
        if model is None:
            raise ValueError(f"Không tìm thấy unified ONNX model: {selected_id}")
        return model

    def report_rows(self, filename, offset=0, limit=20):
        path = self.resolve_report_path(filename)
        if path.suffix.lower() != ".csv":
            raise ValueError("Chỉ có thể duyệt từng dòng từ report CSV")
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 100))
        rows = []
        total = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for total, row in enumerate(csv.DictReader(stream), start=1):
                index = total - 1
                if offset <= index < offset + limit:
                    rows.append(row)
        return {
            "filename": path.name,
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_previous": offset > 0,
            "has_next": offset + len(rows) < total,
            "rows": rows,
        }

    def _report_paths(self):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        token = uuid.uuid4().hex[:8]
        stem = f"evaluation_{timestamp}_{token}"
        self.report_root.mkdir(parents=True, exist_ok=True)
        return self.report_root / f"{stem}.csv", self.report_root / f"{stem}.json"

    @staticmethod
    def _base_report_row(sample):
        row = {field: "" for field in REPORT_FIELDS}
        row.update({
            "image_path": sample["path"],
            "truth_food_status": sample["food_status"],
            "truth_category": sample["category"],
            "truth_freshness": sample["freshness"],
            "truth_product": sample["product"],
            "unified_available": False,
            "legacy_available": False,
        })
        return row

    def filter_samples(self, food_status=None, category=None, freshness=None):
        return [
            sample
            for sample in self.samples
            if (not food_status or sample["food_status"] == food_status)
            and (not category or sample["category"] == category)
            and (not freshness or sample["freshness"] == freshness)
        ]

    def status(self):
        food_statuses = Counter(sample["food_status"] for sample in self.samples)
        categories = Counter(
            sample["category"] for sample in self.samples if sample["category"]
        )
        freshness = Counter(
            sample["freshness"] for sample in self.samples if sample["freshness"]
        )
        return {
            "model_ready": self.model.ready,
            "model_error": self.model.error,
            "model_path": str(self.model.model_path),
            "legacy_ready": self.legacy_model.ready,
            "legacy_error": self.legacy_model.error,
            "legacy_root": str(self.legacy_model.root),
            "food_gate_ready": self.model.food_gate_ready,
            "default_unified_model_id": self.default_unified_model_id,
            "unified_models": [
                model.descriptor() for model in self.unified_models.values()
            ],
            "test_root": str(self.test_root),
            "negative_test_root": str(self.negative_test_root),
            "test_images": len(self.samples),
            "counts": {
                "food_status": dict(food_statuses),
                "category": dict(categories),
                "freshness": dict(freshness),
            },
            "labels": {
                "food_statuses": [
                    {"value": "SUPPORTED", "label": "Thực phẩm được hỗ trợ"},
                    {"value": "UNSUPPORTED", "label": "Thực phẩm ngoài phạm vi"},
                    {"value": "NON_FOOD", "label": "Không phải thực phẩm"},
                ],
                "categories": [
                    {"value": name, "label": CATEGORY_LABELS[name]}
                    for name in CATEGORIES
                ],
                "freshness": [
                    {"value": name, "label": FRESHNESS_LABELS[name]}
                    for name in FRESHNESS_CLASSES
                ],
            },
        }

    def predict(self, relative_path, model_mode="both", unified_model_id=None):
        self.validate_model_mode(model_mode)
        sample = self.by_path.get(relative_path)
        path = self.resolve_sample_path(relative_path)
        result = {"ground_truth": sample}
        use_unified = model_mode in {"both", "unified"}
        use_legacy = model_mode in {"both", "legacy"}
        model = self.unified_model(unified_model_id) if use_unified else None
        if model is not None and model.ready:
            prediction = model.predict_batch([path])[0]
            prediction["model"] = model.descriptor()
            prediction["correct"] = {}
            if prediction["food_gate_ready"]:
                prediction["correct"]["food"] = (
                    prediction["food_status"] == sample["food_status"]
                )
            if sample["food_status"] == "SUPPORTED":
                prediction["correct"].update({
                    "category": prediction["category"] == sample["category"],
                    "freshness": prediction["freshness"] == sample["freshness"],
                })
            evaluable = list(prediction["correct"].values())
            prediction["correct"]["joint"] = bool(evaluable) and all(evaluable)
            result["unified"] = {"ready": True, "prediction": prediction}
        else:
            result["unified"] = {
                "ready": False,
                "error": model.error if model is not None else "Unified không được chọn",
            }

        if use_legacy and self.legacy_model.ready and sample["food_status"] == "SUPPORTED":
            legacy = self.legacy_model.predict_batch(
                [path], [sample["category"]]
            )[0]
            pipeline = legacy["pipeline"]
            oracle = legacy["oracle"]
            pipeline["correct"] = {
                "category": pipeline["category"] == sample["category"],
                "freshness": pipeline["freshness"] == sample["freshness"],
            }
            pipeline["correct"]["joint"] = all(
                pipeline["correct"].values()
            )
            oracle["correct"] = {
                "freshness": oracle["freshness"] == sample["freshness"]
            }
            result["legacy"] = {
                "ready": True,
                "pipeline": pipeline,
                "oracle": oracle,
            }
        else:
            result["legacy"] = {
                "ready": False,
                "applicable": sample["food_status"] == "SUPPORTED",
                "error": (
                    (self.legacy_model.error if use_legacy else "Legacy không được chọn")
                    if sample["food_status"] == "SUPPORTED"
                    else "Legacy không có food gate nên không chấm ảnh negative"
                ),
            }
        return result

    def evaluate(
        self,
        food_status,
        category,
        freshness,
        limit,
        seed,
        model_mode="both",
        unified_model_id=None,
    ):
        self.validate_model_mode(model_mode)
        candidates = self.filter_samples(food_status, category, freshness)
        if not candidates:
            raise ValueError("Không có ảnh test phù hợp bộ lọc")
        requested = int(limit)
        if requested <= 0 or requested >= len(candidates):
            selected = list(candidates)
        else:
            selected = random.Random(seed).sample(candidates, requested)

        unified_counts = Counter()
        legacy_counts = Counter()
        food_truth_counts = Counter()
        food_correct_counts = Counter()
        food_confusion = Counter()
        supported_total = sum(
            sample["food_status"] == "SUPPORTED" for sample in selected
        )
        use_unified = model_mode in {"both", "unified"}
        use_legacy = model_mode in {"both", "legacy"}
        model = self.unified_model(unified_model_id) if use_unified else None
        if use_legacy and not use_unified and not supported_total:
            raise ValueError("Legacy không có food gate nên không thể chấm ảnh negative")
        csv_path, summary_path = self._report_paths()
        try:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
                writer.writeheader()
                batch_size = 32
                for start in range(0, len(selected), batch_size):
                    batch = selected[start:start + batch_size]
                    paths = [
                        self.resolve_sample_path(sample["path"])
                        for sample in batch
                    ]
                    rows = [self._base_report_row(sample) for sample in batch]
                    if model is not None and model.ready:
                        outputs = model.predict_batch(paths)
                        for sample, output, row in zip(batch, outputs, rows):
                            is_supported = sample["food_status"] == "SUPPORTED"
                            category_correct = None
                            freshness_correct = None
                            if is_supported:
                                category_correct = (
                                    output["category"] == sample["category"]
                                )
                                freshness_correct = (
                                    output["freshness"] == sample["freshness"]
                                )
                                unified_counts["category"] += category_correct
                                unified_counts["freshness"] += freshness_correct

                            food_correct = None
                            if output["food_gate_ready"]:
                                truth_status = sample["food_status"]
                                predicted_status = output["food_status"]
                                food_correct = predicted_status == truth_status
                                unified_counts["food"] += food_correct
                                food_truth_counts[truth_status] += 1
                                food_correct_counts[truth_status] += food_correct
                                food_confusion[(truth_status, predicted_status)] += 1

                            if is_supported:
                                joint_correct = category_correct and freshness_correct
                                if food_correct is not None:
                                    joint_correct = joint_correct and food_correct
                            else:
                                joint_correct = bool(food_correct)
                            unified_counts["joint"] += joint_correct
                            row.update({
                                "unified_available": True,
                                "unified_model_id": model.model_id,
                                "unified_model_name": model.display_name,
                                "unified_architecture": model.architecture,
                                "unified_food_status": output["food_status"],
                                "unified_food_confidence": output["food_confidence"],
                                "unified_category": output["category"],
                                "unified_category_confidence": output["category_confidence"],
                                "unified_freshness": output["freshness"],
                                "unified_freshness_confidence": output["freshness_confidence"],
                                "unified_score": output["estimated_freshness_score"],
                                "unified_food_correct": food_correct,
                                "unified_category_correct": category_correct,
                                "unified_freshness_correct": freshness_correct,
                                "unified_joint_correct": joint_correct,
                            })
                            for name, value in output["probabilities"]["food"].items():
                                row[f"unified_food_probability_{name}"] = value
                            for name, value in output["probabilities"]["category"].items():
                                row[f"unified_category_probability_{name}"] = value
                            for name, value in output["probabilities"]["freshness"].items():
                                row[f"unified_freshness_probability_{name}"] = value
                    if use_legacy and self.legacy_model.ready:
                        supported_items = [
                            (sample, path, row)
                            for sample, path, row in zip(batch, paths, rows)
                            if sample["food_status"] == "SUPPORTED"
                        ]
                        outputs = self.legacy_model.predict_batch(
                            [item[1] for item in supported_items],
                            [item[0]["category"] for item in supported_items],
                        ) if supported_items else []
                        for (sample, _, row), output in zip(supported_items, outputs):
                            pipeline = output["pipeline"]
                            oracle = output["oracle"]
                            category_correct = (
                                pipeline["category"] == sample["category"]
                            )
                            freshness_correct = (
                                pipeline["freshness"] == sample["freshness"]
                            )
                            oracle_correct = (
                                oracle["freshness"] == sample["freshness"]
                            )
                            joint_correct = category_correct and freshness_correct
                            legacy_counts["category"] += category_correct
                            legacy_counts["freshness"] += freshness_correct
                            legacy_counts["joint"] += joint_correct
                            legacy_counts["oracle"] += oracle_correct
                            row.update({
                                "legacy_available": True,
                                "legacy_router_category": pipeline["category"],
                                "legacy_router_confidence": pipeline["category_confidence"],
                                "legacy_pipeline_freshness": pipeline["freshness"],
                                "legacy_pipeline_freshness_confidence": pipeline["freshness_confidence"],
                                "legacy_pipeline_score": pipeline["estimated_freshness_score"],
                                "legacy_pipeline_expert_classes": pipeline["class_count"],
                                "legacy_category_correct": category_correct,
                                "legacy_freshness_correct": freshness_correct,
                                "legacy_joint_correct": joint_correct,
                                "legacy_oracle_category": oracle["category"],
                                "legacy_oracle_freshness": oracle["freshness"],
                                "legacy_oracle_confidence": oracle["freshness_confidence"],
                                "legacy_oracle_score": oracle["estimated_freshness_score"],
                                "legacy_oracle_expert_classes": oracle["class_count"],
                                "legacy_oracle_correct": oracle_correct,
                            })
                            for name, value in pipeline["category_probabilities"].items():
                                row[f"legacy_router_probability_{name}"] = value
                            for name, value in pipeline["freshness_probabilities"].items():
                                row[f"legacy_pipeline_probability_{name}"] = value
                            for name, value in oracle["freshness_probabilities"].items():
                                row[f"legacy_oracle_probability_{name}"] = value
                    writer.writerows(rows)
        except Exception:
            csv_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
            raise
        total = len(selected)
        unified_metrics = None
        if model is not None and model.ready:
            food_accuracy = (
                unified_counts["food"] / total
                if model.food_gate_ready
                else None
            )
            unified_metrics = {
                "food_accuracy": food_accuracy,
                "food_recall": {
                    name: (
                        food_correct_counts[name] / food_truth_counts[name]
                        if food_truth_counts[name]
                        else None
                    )
                    for name in FOOD_STATUS_CLASSES
                },
                "food_confusion_matrix": [
                    [food_confusion[(truth, predicted)] for predicted in FOOD_STATUS_CLASSES]
                    for truth in FOOD_STATUS_CLASSES
                ],
                "category_accuracy": (
                    unified_counts["category"] / supported_total
                    if supported_total else None
                ),
                "freshness_accuracy": (
                    unified_counts["freshness"] / supported_total
                    if supported_total else None
                ),
                "joint_accuracy": unified_counts["joint"] / total,
                "supported_count": supported_total,
            }
        legacy_metrics = None
        if use_legacy and self.legacy_model.ready and supported_total:
            legacy_metrics = {
                "pipeline_category_accuracy": legacy_counts["category"] / supported_total,
                "pipeline_freshness_accuracy": legacy_counts["freshness"] / supported_total,
                "pipeline_joint_accuracy": legacy_counts["joint"] / supported_total,
                "oracle_freshness_accuracy": legacy_counts["oracle"] / supported_total,
                "applicable_count": supported_total,
            }
        result = {
            "sample_count": total,
            "candidate_count": len(candidates),
            "evaluated_all": total == len(candidates),
            "filters": {
                "model_mode": model_mode,
                "unified_model_id": model.model_id if model is not None else None,
                "food_status": food_status,
                "category": category,
                "freshness": freshness,
            },
            "models": {
                "unified": {
                    "ready": unified_metrics is not None,
                    "error": model.error if model is not None else "Unified không được chọn",
                    "model": model.descriptor() if model is not None else None,
                    "metrics": unified_metrics,
                },
                "legacy": {
                    "ready": legacy_metrics is not None,
                    "error": (
                        (self.legacy_model.error if use_legacy else "Legacy không được chọn")
                        if not (use_legacy and self.legacy_model.ready)
                        else (
                            None if supported_total
                            else "Legacy không áp dụng cho ảnh negative"
                        )
                    ),
                    "metrics": legacy_metrics,
                },
            },
            "reports": {
                "csv": {
                    "filename": csv_path.name,
                    "url": f"/api/report?file={csv_path.name}",
                },
                "summary_json": {
                    "filename": summary_path.name,
                    "url": f"/api/report?file={summary_path.name}",
                },
            },
        }
        summary_path.write_text(
            json.dumps({
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "csv_report": csv_path.name,
                **{key: value for key, value in result.items() if key != "reports"},
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result


class RequestHandler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, format_string, *args):
        sys.stdout.write(
            f"[{self.log_date_time_string()}] {format_string % args}\n"
        )

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({"error": message}, status)

    def _serve_file(self, path, cache=False, download_name=None):
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "Không tìm thấy tài nguyên")
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        if download_name:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{download_name}"'
            )
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/status":
                self._json(self.app.status())
                return
            if parsed.path == "/api/random":
                query = parse_qs(parsed.query)
                candidates = self.app.filter_samples(
                    query.get("food_status", [None])[0] or None,
                    query.get("category", [None])[0] or None,
                    query.get("freshness", [None])[0] or None,
                )
                if not candidates:
                    self._error(HTTPStatus.NOT_FOUND, "Không có ảnh phù hợp bộ lọc")
                    return
                self._json(random.choice(candidates))
                return
            if parsed.path == "/api/image":
                query = parse_qs(parsed.query)
                relative = unquote(query.get("path", [""])[0])
                self._serve_file(self.app.resolve_sample_path(relative), cache=True)
                return
            if parsed.path == "/api/report":
                query = parse_qs(parsed.query)
                filename = unquote(query.get("file", [""])[0])
                self._serve_file(
                    self.app.resolve_report_path(filename),
                    download_name=filename,
                )
                return
            if parsed.path == "/api/report_rows":
                query = parse_qs(parsed.query)
                filename = unquote(query.get("file", [""])[0])
                self._json(self.app.report_rows(
                    filename,
                    query.get("offset", [0])[0],
                    query.get("limit", [20])[0],
                ))
                return
            if parsed.path == "/":
                self._serve_file(STATIC_ROOT / "index.html")
                return
            relative = parsed.path.lstrip("/")
            static_path = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT.resolve() not in static_path.parents:
                self._error(HTTPStatus.NOT_FOUND, "Đường dẫn không hợp lệ")
                return
            self._serve_file(static_path)
        except (FileNotFoundError, ValueError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 64 * 1024:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request quá lớn")
                return
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            if parsed.path == "/api/predict":
                self._json(self.app.predict(
                    payload.get("path", ""),
                    payload.get("model_mode", "both"),
                    payload.get("unified_model_id") or None,
                ))
                return
            if parsed.path == "/api/evaluate":
                self._json(self.app.evaluate(
                    payload.get("food_status") or None,
                    payload.get("category") or None,
                    payload.get("freshness") or None,
                    payload.get("limit", 100),
                    payload.get("seed", 42),
                    payload.get("model_mode", "both"),
                    payload.get("unified_model_id") or None,
                ))
                return
            self._error(HTTPStatus.NOT_FOUND, "API không tồn tại")
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))


def main():
    parser = argparse.ArgumentParser(description="Web local đánh giá unified ONNX")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument(
        "--negative-test-root",
        type=Path,
        default=DEFAULT_NEGATIVE_TEST_ROOT,
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--onnx-root",
        type=Path,
        default=DEFAULT_UNIFIED_ROOT,
        help="Thư mục quét đệ quy các unified ONNX",
    )
    parser.add_argument(
        "--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT
    )
    parser.add_argument(
        "--report-root", type=Path, default=DEFAULT_REPORT_ROOT
    )
    args = parser.parse_args()

    app = LocalEvaluationApp(
        args.test_root,
        args.onnx,
        args.legacy_root,
        args.report_root,
        args.negative_test_root,
        args.onnx_root,
    )
    RequestHandler.app = app
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print("=" * 68)
    print("WEB SO SÁNH UNIFIED / LEGACY ONNX")
    print(f"  URL       : http://{args.host}:{args.port}")
    print(f"  Test      : {app.test_root} ({len(app.samples)} ảnh)")
    print(f"  Negative  : {app.negative_test_root}")
    print(f"  ONNX root : {app.unified_root}")
    ready_models = sum(model.ready for model in app.unified_models.values())
    print(f"  Unified   : {ready_models}/{len(app.unified_models)} model sẵn sàng")
    print(f"  Legacy    : {'sẵn sàng' if app.legacy_model.ready else 'chưa có'}")
    print(f"  Reports   : {app.report_root}")
    for model in app.unified_models.values():
        state = "sẵn sàng" if model.ready else model.error
        print(f"    - {model.display_name} [{model.model_id}]: {state}")
    if app.legacy_model.error:
        print(f"  Legacy    : {app.legacy_model.error}")
    print("  Dừng      : Ctrl+C")
    print("=" * 68)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng web local.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
