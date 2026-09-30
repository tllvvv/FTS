import sys
import pandas as pd

from common import DATA_RAW, DATA_PREP, SIZES, INSERT_BATCH, SEED

MIN_WORDS = 10


def main():
    frames = []
    for name in ("DBPEDIA_train.csv", "DBPEDIA_val.csv", "DBPEDIA_test.csv"):
        path = DATA_RAW / name
        if not path.exists():
            sys.exit("Не найден файл {}".format(path))
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    print("Исходный объём датасета:", len(df))

    df["text"] = df["text"].astype(str).str.strip()
    before = len(df)
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    print("Отброшено статей короче {} слов: {}".format(MIN_WORDS, before - len(df)))

    before = len(df)
    df = df.drop_duplicates(subset="text")
    print("Отброшено дубликатов:", before - len(df))

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    need = SIZES["large"] + INSERT_BATCH
    if len(df) < need:
        sys.exit("Недостаточно записей: есть {}, требуется {}".format(len(df), need))

    corpus = df.iloc[:need].copy()
    corpus.insert(0, "id", range(1, len(corpus) + 1))
    # split: id <= 300000 -- основной корпус, остальное -- выборка для INSERT
    corpus["part"] = ["corpus"] * SIZES["large"] + ["insert"] * INSERT_BATCH
    corpus["n_words"] = corpus["text"].str.split().str.len()
    corpus["n_chars"] = corpus["text"].str.len()

    out = DATA_PREP / "corpus.parquet"
    try:
        corpus.to_parquet(out, index=False)
    except Exception:                       # нет pyarrow -- сохраняем в csv
        out = DATA_PREP / "corpus.csv"
        corpus.to_csv(out, index=False)
    print("Сохранено:", out, len(corpus), "записей")

    stats = corpus[corpus.part == "corpus"]
    print("\nХарактеристики корпуса (300 000 статей):")
    print("  слов:   среднее {:.1f}, медиана {:.0f}, мин {}, макс {}".format(
        stats.n_words.mean(), stats.n_words.median(),
        stats.n_words.min(), stats.n_words.max()))
    print("  знаков: среднее {:.1f}, медиана {:.0f}".format(
        stats.n_chars.mean(), stats.n_chars.median()))
    print("  суммарный объём текста, МБ: {:.1f}".format(stats.n_chars.sum() / 1024 / 1024))

    summary = pd.DataFrame([{
        "articles": len(stats),
        "words_mean": round(stats.n_words.mean(), 1),
        "words_median": int(stats.n_words.median()),
        "words_min": int(stats.n_words.min()),
        "words_max": int(stats.n_words.max()),
        "chars_mean": round(stats.n_chars.mean(), 1),
        "text_mb": round(stats.n_chars.sum() / 1024 / 1024, 1),
    }])
    summary.to_csv(DATA_PREP / "corpus_stats.csv", index=False)


if __name__ == "__main__":
    main()
