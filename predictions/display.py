import pandas as pd
import unicodedata


def get_display_width(text):
    """全角・半角を考慮した文字幅を取得します"""
    return sum(
        2 if unicodedata.east_asian_width(c) in "FWA" else 1
        for c in str(text)
    )


def pad_str(text, width, align="left"):
    """文字幅を考慮して空白でパディングします"""
    text_str = str(text)
    pad_len = max(0, width - get_display_width(text_str))

    if align == "left":
        return text_str + " " * pad_len
    else:
        return " " * pad_len + text_str


def main():
    csv_file = "_all_predictions.csv"

    try:
        # BOM付きUTF-8に対応して読み込み
        df = pd.read_csv(csv_file, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"エラー: '{csv_file}' が見つかりません。同じディレクトリに配置してください。")
        return

    # 「1R」「10R」などのレース番号を数値化してソート用カラムを作成
    df["レース番号_num"] = (
        df["レース番号"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    # レース番号: 昇順 → 競馬場: 昇順 → 1着確率: 降順
    df = df.sort_values(
        by=["レース番号_num", "競馬場", "1着確率"],
        ascending=[True, True, False]
    )

    # コンソール出力用のANSIカラーコード
    COLOR_RED = "\033[91m"
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    COLOR_RESET = "\033[0m"

    # ヘッダーの表示
    header = (
        f"{pad_str('競馬場', 8)} | "
        f"{pad_str('レース', 6)} | "
        f"{pad_str('馬番', 4)} | "
        f"{pad_str('馬名', 22)} | "
        f"{pad_str('騎手', 14)} | "
        f"{pad_str('1着確率', 10, 'right')} | "
        f"{pad_str('2着以内確率', 12, 'right')} | "
        f"{pad_str('3着以内確率', 12, 'right')}"
    )

    separator = "-" * 110

    print(header)
    print(separator)

    # 直前に出力したレース（競馬場・レース番号）を記録
    previous_race_key = None

    # 各行の判定と出力
    for _, row in df.iterrows():
        keiba = row["競馬場"]
        race = row["レース番号"]
        umaban = row["馬番"]
        umamei = row["馬名"]
        kishu = row["騎手"]

        p1 = row["1着確率"]
        p2 = row["2着以内確率"]
        p3 = row["3着以内確率"]

        # 競馬場またはレース番号が変わったら区切り線を表示
        current_race_key = (keiba, race)

        if previous_race_key is not None and current_race_key != previous_race_key:
            print(separator)

        previous_race_key = current_race_key

        # 確率を小数点以下4桁でフォーマット
        s_p1 = f"{p1:.4f}".rjust(10)
        s_p2 = f"{p2:.4f}".rjust(12)
        s_p3 = f"{p3:.4f}".rjust(12)

        # 条件による色付け処理
        if p1 >= 0.2:
            s_p1 = s_p1.replace(
                f"{p1:.4f}",
                f"{COLOR_RED}{p1:.4f}{COLOR_RESET}"
            )

        if p2 >= 0.3:
            s_p2 = s_p2.replace(
                f"{p2:.4f}",
                f"{COLOR_GREEN}{p2:.4f}{COLOR_RESET}"
            )

        if p3 >= 0.4:
            s_p3 = s_p3.replace(
                f"{p3:.4f}",
                f"{COLOR_YELLOW}{p3:.4f}{COLOR_RESET}"
            )

        # 整形して1行ずつ出力
        line = (
            f"{pad_str(keiba, 8)} | "
            f"{pad_str(race, 6)} | "
            f"{pad_str(umaban, 4)} | "
            f"{pad_str(umamei, 22)} | "
            f"{pad_str(kishu, 14)} | "
            f"{s_p1} | "
            f"{s_p2} | "
            f"{s_p3}"
        )

        print(line)


if __name__ == "__main__":
    main()