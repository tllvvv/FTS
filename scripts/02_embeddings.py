import time
import numpy as np
import pandas as pd

from common import DATA_PREP, ROOT, EMB_DIM, MAX_TOKENS

MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
BATCH = 128


def load_corpus():
    p = DATA_PREP / "corpus.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(DATA_PREP / "corpus.csv")


def main():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_TOKENS)
    tok.enable_padding(length=None, pad_id=0, pad_token="[PAD]")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(MODEL_DIR / "onnx" / "model.onnx"),
                                so, providers=["CPUExecutionProvider"])
    inputs = {i.name for i in sess.get_inputs()}

    df = load_corpus()
    texts = df["text"].tolist()
    n = len(texts)
    out = np.lib.format.open_memmap(DATA_PREP / "embeddings.npy", mode="w+",
                                    dtype=np.float32, shape=(n, EMB_DIM))

    t0 = time.perf_counter()
    for start in range(0, n, BATCH):
        chunk = texts[start:start + BATCH]
        enc = tok.encode_batch(chunk)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(None, feed)[0]                       # (B, L, 384)
        m = mask[:, :, None].astype(np.float32)
        pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        out[start:start + len(chunk)] = pooled.astype(np.float32)

        if (start // BATCH) % 20 == 0 and start:
            el = time.perf_counter() - t0
            rate = start / el
            print("  {:>7}/{} | {:.1f} док/с | осталось ~{:.1f} мин".format(
                start, n, rate, (n - start) / rate / 60), flush=True)

    out.flush()
    el = time.perf_counter() - t0
    print("Готово: {} векторов за {:.1f} мин ({:.1f} док/с)".format(n, el / 60, n / el))

    with open(DATA_PREP / "embeddings_meta.txt", "w", encoding="utf-8") as f:
        f.write("model=sentence-transformers/all-MiniLM-L6-v2 (ONNX)\n")
        f.write("dim={}\nmax_tokens={}\npooling=mean\nnormalize=L2\n".format(EMB_DIM, MAX_TOKENS))
        f.write("rows={}\nseconds={:.1f}\n".format(n, el))


if __name__ == "__main__":
    main()
