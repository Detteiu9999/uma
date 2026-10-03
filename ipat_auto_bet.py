# -*- coding: utf-8 -*-
"""ipat_auto_bet.py
JRA ネット投票（IPAT）スマートフォン版で、指定した買い目を自動購入するモジュール。

schedule_bets.py から --auto-bet 指定時に呼び出される。
ログイン情報は ipat_login_info.py（git 管理外）から読み込む。

通信仕様はブラウザ実機（pw_740 通常投票画面）の挙動を解析して再現したもの:
  1. index.cgi          : ログイン画面（uh トークン取得）
  2. pw_732_i.cgi       : ログイン実行（u=利用者番号+暗証番号+P-ARS番号）
  3. pw_740_i.cgi       : 通常投票トップ（場コード Mg / レース状態 Jg を取得）
  4. pw_743_i.cgi (POST): 馬情報取得（tr=<場ID><レース番号2桁>）※オッズ・出走馬確認用
  5. pw_741_i.cgi       : 投票一覧への登録（Nb コードをスロット名 001.. で送信）
  6. pw_742_i.cgi       : 投票実行（確認ページの FORM0 をそのまま送信:
                          受理済み Nb を 001.. スロットに格納 + 合計金額 l=s）
                          ※この POST で購入が確定する

Nb コード（27文字）:
  [0]    '1'（通常）
  [1]    方式（0=通常 ※単勝/複勝/枠連/馬連/ワイド/馬単/3連複/3連単の通常買いのみ対応）
  [2:4]  スロット番号 16進2桁（送信時は常に '00' でよい）
  [4]    場コード（Mg の1文字目に対応: 1=札幌,2=函館,...,9=阪神,A=小倉）
  [5]    レース番号 16進1桁（1..C）
  [6]    曜日コード（Mg の2文字目。開催日の曜日）
  [7:9]  方式+式別（'01'=単勝, '02'=複勝, '03'=枠連, '04'=馬連, '05'=ワイド,
                    '06'=馬単, '07'=3連複, '08'=3連単）
  [9:23] 馬番ビットマップ（式別ごとの ConvBT 形式）
  [23:27] 金額の16進4桁。**単位は100円**（100円="0001"。js740: `金額<INPUT/>00円` の
          入力欄に100円単位で入力し `mml="000"+hex(mm)` として詰められる。
          誤って円で入れると100倍の金額になるので注意）
"""

import re
import threading
import time

import requests

from ipat_login_info import IPAT_PASSCODE, IPAT_PARS_ID, IPAT_USER_ID

IPAT_BASE = "https://www.ipat.jra.go.jp/sp/"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

MAX_BET_SLOTS = 255
UNIT_YEN = 100  # 1点あたりの購入金額
SESSION_MAX_IDLE_SECONDS = 15 * 60  # 共有セッションを使い回す上限（これを超えたら再ログイン）

# 場コード（race_id の4-5桁目）→ IPAT 場コード文字
JYO_CODE_CHARS = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
                  6: "6", 7: "7", 8: "8", 9: "9", 10: "A"}

# 式別 → Nb の式別コード
BET_TYPE_CODES = {"単勝": "1", "複勝": "2", "枠連": "3", "馬連": "4",
                  "ワイド": "5", "馬単": "6", "3連複": "7", "3連単": "8"}


class IpatError(RuntimeError):
    pass


class IpatSessionExpired(IpatError):
    """セッション切れと判断できるエラー（再ログインで回復可能）。
    _form_fields は pw_742（購入確定）送信前の処理でのみ使われるため、
    このエラーの再試行で二重購入が発生することはない。"""


def _form_fields(html, form_name):
    match = re.search(rf'<FORM[^>]*NAME="{form_name}"[^>]*>(.*?)</FORM>',
                      html, re.S | re.I)
    if not match:
        raise IpatSessionExpired(f"フォーム {form_name} が見つかりません（セッション切れの可能性）")
    return dict(re.findall(r'NAME="?([A-Za-z0-9_]+)"?\s+VALUE="([^"]*)"',
                           match.group(1), re.I))


def _hidden_value(html, name):
    match = re.search(rf'NAME={name} VALUE="([^"]+)"', html)
    if not match:
        raise IpatError(f"隠しフィールド {name} が見つかりません")
    return match.group(1)


# ---------------------------------------------------------------------------
# Nb コード生成
# ---------------------------------------------------------------------------

def _cbt_horse_bit(num):
    """馬番(1-18) → CBT('3') 相当のビット値（0x100000 基準）。"""
    if not 1 <= num <= 18:
        raise ValueError(f"馬番が範囲外です: {num}")
    return 0x100000 + (0x80000 >> (num - 1))


def _cbt4_bit(num):
    return 0x1000000000 + (0x800000000 >> (num - 1))


def _cbt5_bit(num):
    return 0x10000 + (0x8000 >> (num - 1))


def _cbt1_waku(num):
    return 0x100 + (0x80 >> (num - 1))


def _cbt2_waku(num):
    return 0x1000 + (0x200 >> (num - 1))


def build_combi_code(bet_type, combo):
    """式別と馬番・枠番のタプルから Nb[9:23]（14文字）を生成する。
    通常買い（方式0）のみ対応。ボックス・ながし・マルチは未対応。"""
    if bet_type in ("単勝", "複勝"):
        (num,) = combo
        return format(_cbt_horse_bit(num), "x")[-5:] + "000000000"
    if bet_type == "枠連":
        w1, w2 = combo
        dx = _cbt1_waku(w1)
        dx1 = _cbt2_waku(w2)
        return format(dx, "x")[-2:] + "00" + format(dx1, "x")[-3:] + "0000000"
    if bet_type in ("馬連", "ワイド", "馬単"):
        # 通常買いは馬連・ワイド・馬単とも「上位ビット=1着目, 下位ビット=2着目」。
        # 馬連・ワイドは昇順、馬単は着順どおり（parse_bet_combo で正規化済み）。
        # 2着目ビットは 0x20000 >> (n-1)（ブラウザ実機の生成値で確認済み）。
        a, b = combo
        dx = 0x1000000000 + (0x800000000 >> (a - 1)) + (0x20000 >> (b - 1))
        return format(dx, "x")[-9:] + "00000"
    if bet_type == "3連複":
        a, b, c = combo
        dx = 0x1000000000 + (0x800000000 >> (a - 1)) + (0x20000 >> (b - 1))
        dx1 = 0x100000 + (0x80000 >> (c - 1))
        return format(dx, "x")[-9:] + format(dx1, "x")[-5:]
    if bet_type == "3連単":
        a, b, c = combo
        dx = 0x1000000000 + (0x800000000 >> (a - 1)) + (0x20000 >> (b - 1))
        dx1 = 0x100000 + (0x80000 >> (c - 1))
        return format(dx, "x")[-9:] + format(dx1, "x")[-5:]
    raise ValueError(f"未対応の式別です: {bet_type}")


def build_nb(jyo_char, week_char, race_number, bet_type, combo, yen=UNIT_YEN):
    """投票コード Nb（27文字）を生成する。
    金額フィールドは 100円単位の16進4桁（100円 -> "0001"）。"""
    siki = BET_TYPE_CODES[bet_type]
    if not 1 <= race_number <= 15:
        raise ValueError(f"レース番号が範囲外です: {race_number}")
    if yen % 100 or yen <= 0:
        raise ValueError(f"金額は100円単位である必要があります: {yen}")
    return ("1" + "0" + "00" + jyo_char + format(race_number, "X") + week_char
            + "0" + siki + build_combi_code(bet_type, combo) + format(yen // 100, "04X"))


# ---------------------------------------------------------------------------
# セッション
# ---------------------------------------------------------------------------

class IpatSession:
    """IPAT ログイン〜投票実行を行うセッション。"""

    def __init__(self, sleep=time.sleep):
        self.http = requests.Session()
        self.http.headers["User-Agent"] = USER_AGENT
        self.sleep = sleep
        self.meetings = []  # [{letter, jyo_char, week_char, kai, day, jyo_index}, ...]
        self._s_fields = None
        self.logged_in_at = None  # 最終ログイン時刻（time.monotonic）
        # 共有セッション時に pw_741→pw_742 の投票シーケンスをレース間で直列化するロック
        # （投票一覧はセッション単位でサーバー側に保持されるため、同時実行すると混ざる）
        self.lock = threading.Lock()

    # -- ページ遷移 ----------------------------------------------------------

    def _get_text(self, url, **kwargs):
        response = self.http.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        response.encoding = "euc-jp"
        return response.text

    def _post_text(self, url, data, **kwargs):
        response = self.http.post(url, data=data, timeout=30, **kwargs)
        response.raise_for_status()
        response.encoding = "euc-jp"
        return response.text

    def login(self):
        """ログインし、通常投票トップ（pw_740）の情報を保持する。"""
        html = self._get_text(IPAT_BASE + "index.cgi")
        uh = _hidden_value(html, "uh")
        g = _hidden_value(html, "g")
        html = self._post_text(IPAT_BASE + "pw_732_i.cgi", data={
            "uh": uh, "g": g,
            "u": IPAT_USER_ID + IPAT_PASSCODE + IPAT_PARS_ID,
            "i": IPAT_USER_ID, "p": IPAT_PASSCODE, "r": IPAT_PARS_ID,
            "lf": "0",
        })
        form0 = _form_fields(html, "FORM0")
        if not form0.get("inetid"):
            raise IpatError("ログインに失敗しました（ログイン情報を確認してください）")
        html = self._post_text(IPAT_BASE + "pw_740_i.cgi", data=form0)
        self._parse_vote_top(html)
        self._s_fields = _form_fields(html, "s")
        self.logged_in_at = time.monotonic()

    def stale(self):
        """ログインから一定時間（SESSION_MAX_IDLE_SECONDS）経過していれば True。
        長時間アイドル後の投票はセッション切れの可能性が高いため、
        事前にログインし直すかの判定に使う。"""
        return (self.logged_in_at is None
                or time.monotonic() - self.logged_in_at > SESSION_MAX_IDLE_SECONDS)

    def _parse_vote_top(self, html):
        mg = re.findall(r'"([^"]*)"', re.search(
            r'Mg\s*=\s*new Array\s*\((.*?)\);', html, re.S).group(1))
        jg = {}
        for j, i, code in re.findall(r'Jg\[(\d+)\]\[(\d+)\]\s*=\s*"([^"]*)"', html):
            jg.setdefault(int(j), {})[int(i)] = code
        self.meetings = []
        for j in sorted(jg):
            races = [jg[j][i] for i in sorted(jg[j])]
            code = mg[j]
            self.meetings.append({
                "letter": code[0],
                "jyo_char": code[1],
                "week_char": code[2],
                "kai": int(code[3:5]),
                "day": int(code[5:7]),
                "jyo_index": j,
                "races": races,  # 各26文字: [0:2]HH [2:4]MM [4:6]頭数 [6]発売状態
            })

    # -- レース解決 ----------------------------------------------------------

    def race_state(self, jyo_char, week_char, race_number):
        """指定レースの発売状態を返す。見つからなければ None。"""
        for meeting in self.meetings:
            if meeting["jyo_char"] != jyo_char or meeting["week_char"] != week_char:
                continue
            if race_number > len(meeting["races"]):
                return None
            return meeting["races"][race_number - 1]
        return None

    def find_meeting(self, place_code, kai, day):
        """netkeiba の場コード・回・日から IPAT の開催を特定する。
        回・日が一致する開催が無ければ場コードのみで特定する（当日開催は1場につき1つ）。"""
        jyo_char = JYO_CODE_CHARS[place_code]
        candidates = [m for m in self.meetings if m["jyo_char"] == jyo_char]
        for meeting in candidates:
            if meeting["kai"] == kai and meeting["day"] == day:
                return meeting
        if len(candidates) == 1:
            return candidates[0]
        raise IpatError(
            f"IPAT で対象の開催を特定できません（場={place_code} 回={kai} 日={day}）")

    # -- 投票 ----------------------------------------------------------------

    def place_bets(self, meeting, race_number, combos, yen=UNIT_YEN):
        """1レース分の買い目を投票する。
        combos: [(式別, 番号タプル), ...]（番号は馬番 or 枠番）
        戻り値: 確定した点数。"""
        if not combos:
            return 0
        if len(combos) > MAX_BET_SLOTS:
            raise IpatError(f"投票件数が上限（{MAX_BET_SLOTS}件）を超えています: {len(combos)}件")
        state = self.race_state(meeting["jyo_char"], meeting["week_char"], race_number)
        if state is None:
            raise IpatError(f"レースが見つかりません（{race_number}R）")
        if state[4:6] == "00":
            raise IpatError(f"レース情報がありません（{race_number}R）")
        if state[6] != "0":
            raise IpatError(f"このレースは発売中ではありません（{race_number}R, 状態={state[6]}）")

        nbs = [build_nb(meeting["jyo_char"], meeting["week_char"], race_number,
                        bet_type, combo, yen) for bet_type, combo in combos]

        data = {f"{i:03d}": "0" for i in range(1, MAX_BET_SLOTS + 1)}
        for i, nb in enumerate(nbs):
            data[f"{i + 1:03d}"] = nb
        for key in ("uh", "inetid", "g", "u", "nm", "zj", "mli", "uk"):
            data[key] = self._s_fields.get(key, "")
        html = self._post_text(IPAT_BASE + "pw_741_i.cgi", data=data)
        accepted = [value for _, value in
                    sorted((int(i), v) for i, v in
                           re.findall(r'Nb\[(\d+)\]\s*=\s*"([^"]*)"', html)) if value != "0"]
        if len(accepted) != len(nbs):
            # 確認ページ以外（トップページ g=730 やエラーページ g=742 等）が返った場合は
            # セッション切れとみなす。まだ pw_742 を送っていないため再ログインは安全。
            g = re.search(r'NAME="?g"?\s+VALUE="([^"]*)"', html, re.I)
            if not g or g.group(1) != "741":
                raise IpatSessionExpired(
                    "投票確認ページが返りませんでした（セッション切れの可能性）")
            raise IpatError(
                f"投票一覧への登録に失敗しました（{len(accepted)}/{len(nbs)}件のみ受理）")
        # 受領コードと送信コードの一致確認（スロット番号部分はサーバー側で付け替わる）
        for sent, got in zip(nbs, accepted):
            if sent[:2] != got[:2] or sent[4:] != got[4:]:
                raise IpatError(f"受理内容が送信内容と一致しません: {sent} -> {got}")

        # 合計金額確認 → 投票実行（この POST で購入が確定する）
        # ブラウザの挙動（js741 の ToSend()）: 確認ページの FORM0 をそのまま
        # pw_742_i.cgi へ submit する。FORM0 には 001..255 のスロット input があり、
        # JS が受理済み Nb コード（未使用は "0"）を詰めてから l=s=合計金額(円) を設定する。
        form0 = _form_fields(html, "FORM0")
        total = yen * len(nbs)
        data = {f"{i:03d}": "0" for i in range(1, MAX_BET_SLOTS + 1)}
        for i, nb in enumerate(accepted):
            data[f"{i + 1:03d}"] = nb
        data.update(form0)
        data["l"] = str(total)
        data["s"] = str(total)
        html = self._post_text(IPAT_BASE + "pw_742_i.cgi", data=data)
        # エラーページは <H1>エラー</H1> を持つ（通常ページも文字列「エラー」を
        # 含みうるため H1 タグで判定する。誤検知すると再試行で二重購入になり得る）
        if re.search(r"<H1>エラー</H1>", html):
            raise IpatError("投票実行に失敗しました（pw_742 がエラーを返しました）")
        if not re.search(r'Nb\[0\]\s*=\s*"[^0]', html):
            raise IpatError("投票結果を確認できません（結果ページに投票内容がありません）")
        return len(nbs)

    def close(self):
        self.http.close()


def race_id_to_keys(race_id):
    """netkeiba/JRA-VAN 形式のレースID(12桁) → (場コード, 回, 日, レース番号)。"""
    race_id = str(race_id)
    return (int(race_id[4:6]), int(race_id[6:8]), int(race_id[8:10]), int(race_id[10:12]))


def parse_bet_combo(bet_type, combo_text):
    """schedule_bets の買い目文字列 → 番号タプル（自動購入可能な形式のみ）。
    単勝/複勝: '3 馬名' / 枠連: '枠1-枠2' / 馬連・ワイド・3連複: '1-2[-3]' /
    馬単・3連単: '1→2[→3]'"""
    text = str(combo_text).strip()
    if bet_type in ("単勝", "複勝"):
        match = re.fullmatch(r"(\d+).*", text)
    elif bet_type == "枠連":
        match = re.fullmatch(r"枠(\d+)-枠(\d+)", text)
    elif bet_type in ("馬連", "ワイド", "3連複"):
        match = re.fullmatch(r"(\d+)-(\d+)(?:-(\d+))?", text)
    elif bet_type in ("馬単", "3連単"):
        match = re.fullmatch(r"(\d+)→(\d+)(?:→(\d+))?", text)
    else:
        return None
    if not match:
        return None
    nums = tuple(int(n) for n in match.groups() if n is not None)
    expected = {"単勝": 1, "複勝": 1, "枠連": 2, "馬連": 2, "ワイド": 2,
                "馬単": 2, "3連複": 3, "3連単": 3}
    if len(nums) != expected[bet_type] or len(set(nums)) != len(nums):
        return None
    if bet_type == "枠連":
        if not all(1 <= n <= 8 for n in nums):
            return None
        return tuple(sorted(nums))
    if not all(1 <= n <= 18 for n in nums):
        return None
    if bet_type in ("馬連", "ワイド", "3連複"):
        return tuple(sorted(nums))
    return nums


def buy_race_bets(race_id, bets, yen=UNIT_YEN, session=None, sleep=time.sleep):
    """1レース分の買い目（schedule_bets の suggestions 形式）を購入する。
    bets: [{"式別": ..., "買い目": ...}, ...]
    session に共有セッション（ログイン済み）を渡すとログインを省略する。
    共有セッションが切れていた場合は再ログインして1回だけ再試行する。
    戻り値: 確定した点数。"""
    place_code, kai, day, race_number = race_id_to_keys(race_id)
    combos = []
    for bet in bets:
        bet_type = bet.get("式別")
        if bet_type not in BET_TYPE_CODES:
            raise IpatError(f"自動購入に未対応の式別です: {bet_type}")
        combo = parse_bet_combo(bet_type, bet.get("買い目"))
        if combo is None:
            raise IpatError(f"買い目を解析できません: {bet_type} {bet.get('買い目')}")
        combos.append((bet_type, combo))

    own_session = session is None
    if own_session:
        session = IpatSession(sleep=sleep)
    try:
        with session.lock:
            if own_session:
                session.login()
            elif session.stale():
                # 共有セッションが長時間アイドルなら先にログインし直す
                session.login()
            for attempt in range(2):
                try:
                    meeting = session.find_meeting(place_code, kai, day)
                    return session.place_bets(meeting, race_number, combos, yen)
                except IpatSessionExpired:
                    if own_session or attempt:
                        raise
                    # 共有セッションは長時間のアイドルで切れることがあるため、
                    # 再ログインして1回だけ再試行する
                    session.login()
    finally:
        if own_session:
            session.close()
