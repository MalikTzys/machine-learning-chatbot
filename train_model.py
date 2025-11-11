from __future__ import annotations
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import joblib

from config import (
    DATA_FILE,
    MODELS_DIR,
    SUPPORTED_LANGUAGES,
    MIN_SAMPLES_PER_LANG,
    TFIDF_MAX_FEATURES,
    NGRAM_RANGE,
    MODEL_TYPE,
    USE_TRANSFORMERS,
    RANDOM_SEED,
)
from preprocessing import parse_data_file
from utils import log_info, log_warn, log_error, timed


def _build_vectorizer(lang: str) -> TfidfVectorizer:
    # Bahasa Jepang mendapat manfaat dari character n-grams; bahasa lain menggunakan word n-grams
    if lang == "JP":
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            max_features=TFIDF_MAX_FEATURES,
        )
    else:
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=NGRAM_RANGE,
            min_df=1,
            max_features=TFIDF_MAX_FEATURES,
        )


def train_sklearn_per_language(rows: List[Dict]):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Kelompokkan berdasarkan bahasa
    by_lang: Dict[str, List[Dict]] = {lang: [] for lang in SUPPORTED_LANGUAGES}
    for r in rows:
        lang = r["lang"].upper()
        if lang in by_lang:
            by_lang[lang].append(r)

    for lang, items in by_lang.items():
        if len(items) < MIN_SAMPLES_PER_LANG:
            log_warn("Melewati bahasa - sampel tidak cukup", lang=lang, samples=len(items))
            continue

        X = [r["input"] for r in items]
        y = [r["response"] for r in items]

        with timed(f"Train sklearn model for {lang}"):
            vectorizer = _build_vectorizer(lang)
            knn = NearestNeighbors(n_neighbors=5, metric="cosine")
            pipe = Pipeline([("tfidf", vectorizer), ("knn", knn)])
            # Fit TF-IDF
            pipe.named_steps["tfidf"].fit(X)
            X_mat = pipe.named_steps["tfidf"].transform(X)
            pipe.named_steps["knn"].fit(X_mat)

        # Simpan artifacts
        out_dir = MODELS_DIR / lang / "sklearn"
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / "pipeline.joblib")
        joblib.dump(y, out_dir / "responses.joblib")
        log_info("Saved sklearn artifacts", lang=lang, dir=str(out_dir))


def train_transformers_per_language(rows: List[Dict]):
    """
    Opsional: fine-tune model multilingual T5-style.
    Dengan GPU 6GB, jaga batch tetap kecil dan epochs rendah.
    Langkah ini opsional dan dikontrol oleh flag USE_TRANSFORMERS.
    """
    if not USE_TRANSFORMERS:
        log_warn("Pelatihan Transformers dilewati (USE_TRANSFORMERS=0)")
        return

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    import random
    import numpy as np
    import torch

    from config import (
        HF_MODEL_NAME,
        HF_MAX_INPUT_LENGTH,
        HF_MAX_OUTPUT_LENGTH,
        HF_TRAIN_EPOCHS,
        HF_BATCH_SIZE,
        HF_LR,
    )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    # Kelompokkan baris berdasarkan bahasa
    by_lang: Dict[str, List[Dict]] = {lang: [] for lang in SUPPORTED_LANGUAGES}
    for r in rows:
        lang = r["lang"].upper()
        if lang in by_lang:
            by_lang[lang].append(r)

    for lang, items in by_lang.items():
        if len(items) < MIN_SAMPLES_PER_LANG:
            log_warn("Melewati transformers - sampel tidak cukup", lang=lang, samples=len(items))
            continue

        model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Siapkan dataset
        def preprocess(example):
            src = f"User: {example['input']}\nAssistant:"
            model_inputs = tokenizer(
                src,
                max_length=HF_MAX_INPUT_LENGTH,
                truncation=True,
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    example["response"],
                    max_length=HF_MAX_OUTPUT_LENGTH,
                    truncation=True,
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds_items = [preprocess(ex) for ex in items]

        import torch.utils.data as tud

        class SimpleDS(tud.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self): return len(self.data)

            def __getitem__(self, idx): return self.data[idx]

        ds = SimpleDS(ds_items)

        args = TrainingArguments(
            output_dir=str(MODELS_DIR / lang / "transformers"),
            per_device_train_batch_size=HF_BATCH_SIZE,
            num_train_epochs=HF_TRAIN_EPOCHS,
            learning_rate=HF_LR,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            report_to=[],
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=data_collator)
        with timed(f"Fine-tune transformers for {lang}"):
            trainer.train()

        # Simpan model yang sudah di-fine-tune
        out_dir = MODELS_DIR / lang / "transformers"
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        log_info("Saved transformers model", lang=lang, dir=str(out_dir))


def run_training(data_path: Path | None = None):
    data_path = data_path or DATA_FILE
    rows = parse_data_file(data_path)
    log_info("Loaded training rows", total=len(rows))
    if not rows:
        raise RuntimeError("No training data found.")

    train_sklearn_per_language(rows)
    # Opsional transformers
    train_transformers_per_language(rows)
    log_info("Training completed.")


if __name__ == "__main__":
    run_training()