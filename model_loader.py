from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

from utils import log_info, log_warn, log_error
from config import MODELS_DIR, SUPPORTED_LANGUAGES, MODEL_TYPE, RETRIEVAL_TOP_K, USE_TRANSFORMERS


@dataclass
class RetrievalModel:
    pipeline: Any  # Pipeline sklearn
    responses: List[str]  # respons pelatihan yang diselaraskan


class ChatbotModels:
    def __init__(self):
        self.type = MODEL_TYPE
        self.retrieval: Dict[str, RetrievalModel] = {}
        self.generators: Dict[str, Any] = {}  # pipeline transformers per bahasa, opsional

    def model_path(self, lang: str) -> Path:
        return MODELS_DIR / lang

    def load(self):
        if self.type == "sklearn":
            self._load_sklearn()
        else:
            self._load_sklearn()  # selalu simpan retrieval sebagai fallback
            if USE_TRANSFORMERS:
                self._load_transformers()

    def _load_sklearn(self):
        import joblib

        for lang in SUPPORTED_LANGUAGES:
            lang_dir = self.model_path(lang) / "sklearn"
            pipe_fp = lang_dir / "pipeline.joblib"
            resp_fp = lang_dir / "responses.joblib"
            if pipe_fp.exists() and resp_fp.exists():
                try:
                    pipeline = joblib.load(pipe_fp)
                    responses = joblib.load(resp_fp)
                    self.retrieval[lang] = RetrievalModel(pipeline=pipeline, responses=responses)
                    log_info("Loaded sklearn model", lang=lang, items=len(responses))
                except Exception as e:
                    log_warn("Failed to load sklearn model", lang=lang, error=str(e))
            else:
                log_warn("Sklearn artifacts missing", lang=lang, dir=str(lang_dir))

    def _load_transformers(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
        from config import HF_MODEL_NAME

        for lang in SUPPORTED_LANGUAGES:
            # Untuk kesederhanaan, gunakan model multibahasa yang sama; bisa diganti per folder bahasa jika fine-tuned secara terpisah
            try:
                gen = hf_pipeline(
                    "text2text-generation",
                    model=HF_MODEL_NAME,
                    tokenizer=HF_MODEL_NAME,
                    device=0 if USE_TRANSFORMERS else -1,
                )
                self.generators[lang] = gen
                log_info("Loaded transformers generator", lang=lang, model=HF_MODEL_NAME)
            except Exception as e:
                log_warn("Failed to load transformers generator", lang=lang, error=str(e))

    def infer(self, lang: str, text: str, tone: str = "neutral") -> str:
        """
        - Jika generator transformers ada → hasilkan respons (opsional seed dengan exemplar yang diambil)
        - Jika tidak gunakan respons kecocokan terbaik retrieval
        """
        text = text.strip()
        # Retrieval sebagai dasar
        base_resp = self._retrieve(lang, text)

        # Jika generator tersedia, opsional kondisikan dengan base_resp
        if lang in self.generators:
            prompt = f"User: {text}\nContext: {base_resp}\nTone: {tone}\nAssistant:"
            try:
                out = self.generators[lang](prompt, max_new_tokens=128, num_return_sequences=1)
                gen_text = out[0]["generated_text"]
                return gen_text.strip()
            except Exception as e:
                log_warn("Generation failed, falling back to retrieval", lang=lang, error=str(e))

        # Fallback
        return base_resp

    def _retrieve(self, lang: str, text: str) -> str:
        model = self.retrieval.get(lang)
        if not model:
            # fallback ke bahasa lain yang tersedia
            for m in self.retrieval.values():
                model = m
                break
        if not model:
            return "Maaf, model belum dimuat."

        try:
            vec = model.pipeline.transform([text])
            # Gunakan kneighbors dari NearestNeighbors jika tersedia di dalam pipeline
            # Jika tidak, gunakan dot-product kesamaan cosinus
            knn = getattr(model.pipeline[-1], "kneighbors", None)
            if callable(knn):
                distances, indices = knn(vec, n_neighbors=min(RETRIEVAL_TOP_K, len(model.responses)))
                idx = indices[0][0]
                return model.responses[idx]
            else:
                # kesamaan cosinus
                sim = model.pipeline[-1].transform(vec)  # jalur yang tidak mungkin; simpan untuk kompatibilitas
                idx = 0
                return model.responses[idx]
        except Exception as e:
            log_warn("Retrieval failed", error=str(e))
            return "Saya kesulitan mengambil jawaban saat ini."