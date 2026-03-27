from __future__ import annotations

from time import perf_counter
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from PIL import Image, UnidentifiedImageError

from core.detector_data import DATASET_DIR, format_label_name, get_overview_table, sample_records
from core.model_loader import DEFAULT_MODEL_PATH, predict_image

ROOT_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT_DIR / "web" / "templates"
STATIC_DIR = ROOT_DIR / "web" / "static"


def _format_int(value: int) -> str:
    return f"{value:,}"


def _build_gallery_samples() -> list[dict[str, str]]:
    gallery_frame = sample_records("test_v2", sample_size=6, existing_only=True, seed=17)
    gallery_samples = []
    for row in gallery_frame.itertuples():
        gallery_samples.append(
            {
                "relative_path": row.relative_path,
                "label_display": row.label_display,
                "asset_url": url_for("dataset_asset", relative_path=row.relative_path),
            }
        )
    return gallery_samples


def _resolve_dataset_asset(relative_path: str) -> Path:
    absolute_path = (DATASET_DIR / Path(relative_path)).resolve()
    dataset_root = DATASET_DIR.resolve()

    if not absolute_path.is_relative_to(dataset_root):
        raise FileNotFoundError(relative_path)

    if not absolute_path.exists() or not absolute_path.is_file():
        raise FileNotFoundError(relative_path)

    return absolute_path


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

    @app.route("/")
    def detector_ui() -> str:
        overview = get_overview_table()
        split_lookup = {row.split: row for row in overview.itertuples()}
        model_ready = DEFAULT_MODEL_PATH.exists()

        metrics = {
            "ui_mode": "AI vs Foto Asli",
            "ready_assets": _format_int(int(overview["present_files"].sum())),
            "labeled_eval": _format_int(int(split_lookup["test_v2"].labeled_rows)),
            "missing_assets": _format_int(int(overview["missing_files"].sum())),
        }

        readiness_rows = [
            {
                "label": "Upload preview",
                "status": "Ready",
                "detail": "Preview image sudah aktif via browser tanpa backend inference.",
            },
            {
                "label": "Model loader",
                "status": "Ready" if model_ready else "Pending",
                "detail": (
                    "Checkpoint model tersedia dan loader sudah ada di core/model_loader.py."
                    if model_ready
                    else "Checkpoint model belum ditemukan di folder model."
                ),
            },
            {
                "label": "Inference wiring",
                "status": "Ready" if model_ready else "Pending",
                "detail": (
                    "Tombol analisis sudah terhubung ke model klasifikasi biner."
                    if model_ready
                    else "Prediksi model belum dihubungkan ke request web atau aksi tombol analisis."
                ),
            },
            {
                "label": "Dataset samples",
                "status": "Ready",
                "detail": "UI bisa pakai sampel dari folder dataset buat demo alur.",
            },
        ]

        overview_rows = [
            {
                "title": row.title,
                "rows": _format_int(int(row.rows)),
                "present_files": _format_int(int(row.present_files)),
                "missing_files": _format_int(int(row.missing_files)),
            }
            for row in overview.itertuples()
        ]

        return render_template(
            "index.html",
            metrics=metrics,
            model_ready=model_ready,
            gallery_samples=_build_gallery_samples(),
            readiness_rows=readiness_rows,
            overview_rows=overview_rows,
        )

    @app.route("/asset/<path:relative_path>")
    def dataset_asset(relative_path: str):
        try:
            absolute_path = _resolve_dataset_asset(relative_path)
        except FileNotFoundError:
            abort(404)

        return send_file(absolute_path)

    @app.post("/predict")
    def predict():
        image_file = request.files.get("image")
        relative_path = request.form.get("relative_path", "").strip()

        if image_file is None and not relative_path:
            return jsonify({"error": "Pilih gambar dulu sebelum memulai analisis."}), 400

        if not DEFAULT_MODEL_PATH.exists():
            return jsonify({"error": "Checkpoint model belum tersedia di server."}), 503

        try:
            started_at = perf_counter()

            if image_file is not None and image_file.filename:
                with Image.open(image_file.stream) as uploaded_image:
                    result = predict_image(uploaded_image.copy())
            elif relative_path:
                result = predict_image(_resolve_dataset_asset(relative_path))
            else:
                return jsonify({"error": "File gambar tidak valid."}), 400

            elapsed_ms = round((perf_counter() - started_at) * 1000)
        except UnidentifiedImageError:
            return jsonify({"error": "File ini bukan gambar yang valid."}), 400
        except FileNotFoundError:
            return jsonify({"error": "Gambar sample tidak ditemukan."}), 404
        except RuntimeError as exc:
            return jsonify({"error": f"Gagal memuat model: {exc}"}), 500
        except Exception as exc:
            return jsonify({"error": f"Terjadi error saat inferensi: {exc}"}), 500

        confidence = result.probability if result.predicted_label == 1 else 1 - result.probability
        predicted_label_name = format_label_name(result.predicted_label)

        return jsonify(
            {
                "predicted_label": predicted_label_name,
                "predicted_label_id": result.predicted_label,
                "confidence_percent": round(confidence * 100, 2),
                "ai_probability_percent": round(result.probability * 100, 2),
                "inference_time_ms": elapsed_ms,
                "analysis_message": (
                    f"Model menilai gambar ini sebagai {predicted_label_name} "
                    f"dengan confidence {confidence * 100:.2f}%."
                ),
            }
        )

    return app
