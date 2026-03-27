document.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("imageInput");
    const dropzone = document.getElementById("dropzone");
    const previewImage = document.getElementById("previewImage");
    const placeholder = document.getElementById("placeholder");
    const triggerUpload = document.getElementById("triggerUpload");
    const clearButton = document.getElementById("clearButton");
    const analyzeButton = document.getElementById("analyzeButton");
    const fileMeta = document.getElementById("fileMeta");
    const predictedClass = document.getElementById("predictedClass");
    const confidenceValue = document.getElementById("confidenceValue");
    const aiProbability = document.getElementById("aiProbability");
    const inferenceTime = document.getElementById("inferenceTime");
    const analysisMessage = document.getElementById("analysisMessage");
    const sampleButtons = document.querySelectorAll("[data-image-url]");
    const analyzeButtonLabel = analyzeButton.textContent;

    const defaultState = {
        fileMeta: "Belum ada file dipilih.",
        predictedClass: "Belum dicek",
        confidenceValue: "--",
        aiProbability: "--",
        inferenceTime: "--",
        analysisMessage: "Pilih gambar dulu. Setelah itu, model akan menilai apakah gambar ini buatan AI atau foto asli.",
    };

    let localObjectUrl = null;
    let selectedFile = null;
    let selectedSamplePath = "";

    function revokeLocalObjectUrl() {
        if (localObjectUrl) {
            URL.revokeObjectURL(localObjectUrl);
            localObjectUrl = null;
        }
    }

    function resetResultState() {
        predictedClass.textContent = defaultState.predictedClass;
        confidenceValue.textContent = defaultState.confidenceValue;
        aiProbability.textContent = defaultState.aiProbability;
        inferenceTime.textContent = defaultState.inferenceTime;
    }

    function setPreview(src, labelText) {
        previewImage.src = src;
        previewImage.style.display = "block";
        placeholder.style.display = "none";
        fileMeta.textContent = labelText;
        resetResultState();
        predictedClass.textContent = "Siap dianalisis";
        analysisMessage.textContent =
            "Gambar sudah masuk. Tekan tombol analisis untuk menjalankan model.";
    }

    function clearPreview() {
        revokeLocalObjectUrl();
        imageInput.value = "";
        selectedFile = null;
        selectedSamplePath = "";
        previewImage.removeAttribute("src");
        previewImage.style.display = "none";
        placeholder.style.display = "block";
        fileMeta.textContent = defaultState.fileMeta;
        resetResultState();
        analysisMessage.textContent = defaultState.analysisMessage;
    }

    function readFile(file) {
        if (!file) {
            return;
        }

        if (!file.type.startsWith("image/")) {
            analysisMessage.textContent = "File ini bukan gambar. Coba gunakan JPG, PNG, atau WEBP.";
            return;
        }

        revokeLocalObjectUrl();
        selectedFile = file;
        selectedSamplePath = "";
        localObjectUrl = URL.createObjectURL(file);
        setPreview(localObjectUrl, `${file.name} · ${(file.size / 1024).toFixed(1)} KB`);
    }

    function formatPercent(value) {
        return `${Number(value).toFixed(2)}%`;
    }

    async function runPrediction() {
        if (!selectedFile && !selectedSamplePath) {
            analysisMessage.textContent = "Pilih gambar dulu sebelum memulai analisis.";
            return;
        }

        const formData = new FormData();
        if (selectedFile) {
            formData.append("image", selectedFile);
        } else {
            formData.append("relative_path", selectedSamplePath);
        }

        analyzeButton.disabled = true;
        analyzeButton.textContent = "Sedang menganalisis...";
        analysisMessage.textContent = "Model sedang memproses gambar Anda...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });
            const payload = await response.json();

            if (!response.ok) {
                throw new Error(payload.error || "Pengecekan gagal dijalankan.");
            }

            predictedClass.textContent = payload.predicted_label;
            confidenceValue.textContent = formatPercent(payload.confidence_percent);
            aiProbability.textContent = formatPercent(payload.ai_probability_percent);
            inferenceTime.textContent = `${payload.inference_time_ms} ms`;
            analysisMessage.textContent = payload.analysis_message;
        } catch (error) {
            resetResultState();
            predictedClass.textContent = "Gagal";
            analysisMessage.textContent =
                error instanceof Error ? error.message : "Terjadi masalah saat menjalankan model.";
        } finally {
            analyzeButton.disabled = false;
            analyzeButton.textContent = analyzeButtonLabel;
        }
    }

    triggerUpload.addEventListener("click", () => imageInput.click());
    clearButton.addEventListener("click", clearPreview);

    imageInput.addEventListener("change", (event) => {
        const [file] = event.target.files;
        readFile(file);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        const [file] = event.dataTransfer.files;
        readFile(file);
    });

    analyzeButton.addEventListener("click", runPrediction);

    sampleButtons.forEach((button) => {
        button.addEventListener("click", () => {
            revokeLocalObjectUrl();
            selectedFile = null;
            selectedSamplePath = button.dataset.relativePath || "";
            setPreview(button.dataset.imageUrl, `Foto contoh · ${button.dataset.imageLabel}`);
        });
    });

    window.addEventListener("beforeunload", revokeLocalObjectUrl);
});
