"use strict";

const $ = id => document.getElementById(id);
const yen = value => `${new Intl.NumberFormat("ja-JP").format(value)}円`;
const rate = value => value === null ? "—" : `${value.toFixed(1)}%`;
const signed = value => `${value > 0 ? "+" : ""}${yen(value)}`;
const tone = value => value > 0 ? "positive" : value < 0 ? "negative" : "";
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const filters = ["mode", "stake", "place", "type"];
let currentRequest;
let availableYears = [];

function currentMonth() {
  const parts = new Intl.DateTimeFormat("en-US", {timeZone: "Asia/Tokyo", year: "numeric", month: "numeric"}).formatToParts(new Date());
  return {year: Number(parts.find(p => p.type === "year").value), month: Number(parts.find(p => p.type === "month").value)};
}

let period = {unit: "month", ...currentMonth()};

function periodLabel(selection) {
  if (selection.unit === "all") return "全期間";
  return `${selection.year}年${selection.unit === "month" ? `${selection.month}月` : ""}`;
}

function periodRange(selection) {
  if (selection.unit === "all") return {};
  const year = String(selection.year).padStart(4, "0");
  if (selection.unit === "year") return {from: `${year}-01-01`, to: `${year}-12-31`};
  const month = String(selection.month).padStart(2, "0");
  const leap = selection.year % 4 === 0 && (selection.year % 100 !== 0 || selection.year % 400 === 0);
  const lastDay = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][selection.month - 1];
  return {from: `${year}-${month}-01`, to: `${year}-${month}-${lastDay}`};
}

function updatePeriodControls() {
  const all = period.unit === "all";
  document.querySelectorAll("[data-period]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.period === period.unit));
  });
  $("period-selection").textContent = periodLabel(period);
  $("year-label").hidden = all;
  $("months").hidden = period.unit !== "month";
  const years = [...new Set([...availableYears, currentMonth().year, period.year])].sort((a, b) => b - a);
  $("year").innerHTML = years.map(year => `<option value="${year}">${year}年</option>`).join("");
  $("year").value = String(period.year);
  for (const [id, direction] of [["previous-period", "前"], ["next-period", "次"]]) {
    $(id).hidden = all;
    $(id).setAttribute("aria-label", `${direction}の${period.unit === "month" ? "月" : "年"}`);
  }
  $("previous-period").disabled = period.year <= 1 && (period.unit === "year" || period.month === 1);
  $("next-period").disabled = period.year >= 9999 && (period.unit === "year" || period.month === 12);
  document.querySelectorAll("[data-month]").forEach(button => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.month) === period.month));
  });
}

function selectPeriod(changes) {
  period = {...period, ...changes};
  updatePeriodControls();
  refresh();
}

function movePeriod(step) {
  if (period.unit === "all") return;
  let {year, month} = period;
  if (period.unit === "year") year += step;
  else {
    month += step;
    if (month < 1) { month = 12; year--; }
    if (month > 12) { month = 1; year++; }
  }
  if (year >= 1 && year <= 9999) selectPeriod({year, month});
}

function periodBreakdown(days, selection) {
  if (selection.unit === "month") return days;
  const groups = new Map();
  const fields = ["points", "settled_points", "hits", "purchased", "bet", "pending_bet", "return", "profit"];
  function group(date) {
    if (!groups.has(date)) groups.set(date, {date, ...Object.fromEntries(fields.map(key => [key, 0]))});
    return groups.get(date);
  }
  if (selection.unit === "year") {
    for (let month = 1; month <= 12; month++) group(`${String(selection.year).padStart(4, "0")}-${String(month).padStart(2, "0")}`);
  }
  for (const day of days) {
    const bucket = group(day.date.slice(0, selection.unit === "year" ? 7 : 4));
    fields.forEach(key => { bucket[key] += day[key]; });
  }
  let cumulative = 0;
  return [...groups.values()].sort((a, b) => a.date.localeCompare(b.date)).map(bucket => {
    cumulative += bucket.profit;
    return {...bucket, roi: bucket.bet ? bucket.return / bucket.bet * 100 : null, cumulative_profit: cumulative};
  });
}

function emptyRow(columns) {
  return `<tr><td class="empty" colspan="${columns}">対象データがありません</td></tr>`;
}

function renderChart(days) {
  const points = days.filter(day => day.settled_points > 0);
  if (!points.length) {
    $("chart").innerHTML = '<p class="empty">結果が取得されると、累計収支が表示されます。</p>';
    return;
  }
  const values = [0, ...points.map(day => day.cumulative_profit)];
  let low = Math.min(...values), high = Math.max(...values);
  if (low === high) { low -= 100; high += 100; }
  const pad = (high - low) * .15;
  low -= pad; high += pad;
  const x = i => 105 + i * 830 / Math.max(points.length - 1, 1);
  const y = value => 170 - (value - low) / (high - low) * 145;
  const grid = [low + pad, 0, high - pad].filter((v, i, a) => a.indexOf(v) === i).map(v =>
    `<line x1="100" y1="${y(v)}" x2="945" y2="${y(v)}" stroke="${v === 0 ? "#b2c6c9" : "#edf1f3"}" stroke-dasharray="4 4"/><text x="90" y="${y(v) + 4}" text-anchor="end">${escapeHTML(yen(Math.round(v)))}</text>`).join("");
  const line = points.map((d, i) => `${x(i)},${y(d.cumulative_profit)}`).join(" ");
  const dots = points.map((d, i) => `<circle cx="${x(i)}" cy="${y(d.cumulative_profit)}" r="4" fill="#087e78"><title>${escapeHTML(`${d.date}: ${signed(d.cumulative_profit)}`)}</title></circle>`).join("");
  const labels = [...new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])].map(i =>
    `<text x="${x(i)}" y="205" text-anchor="middle">${escapeHTML(points[i].date)}</text>`).join("");
  $("chart").innerHTML = `<svg viewBox="0 0 1000 225" role="img" aria-label="選択期間の累計収支。詳細は成績表を参照。">${grid}<polyline points="${line}" fill="none" stroke="#087e78" stroke-width="3"/>${dots}${labels}</svg>`;
}

function render(data, selection) {
  availableYears = data.years;
  updatePeriodControls();
  $("report-period").textContent = `${periodLabel(selection)}の成績`;
  const s = data.summary;
  $("roi").textContent = rate(s.roi);
  $("returned").textContent = yen(s.return);
  $("profit").textContent = signed(s.profit);
  $("profit").className = tone(s.profit);
  $("purchased").textContent = yen(s.purchased);
  $("settled").textContent = `うち結果判明分 ${yen(s.bet)}`;
  $("points").textContent = `対象 ${s.points}点 / 結果判明 ${s.settled_points}点 / 的中 ${s.hits}点`;
  $("pending").textContent = `結果未判明 ${s.points - s.settled_points}点・${yen(s.pending_bet)}`;
  $("source").textContent = `読込 ${data.source.races}レース / 購入成功記録 ${data.source.purchased_races}レース`;
  $("purchased-title").textContent = data.mode === "actual" ? "購入総額" : "仮定購入総額";
  $("mode-note").textContent = data.mode === "actual"
    ? `IPAT購入成功メッセージの実金額を使用しています。購入成功記録がない買い目（${data.source.unconfirmed_races}レース）は、シミュレーションで確認できます。`
    : `Discord送信済みの全買い目を1点 ${yen(Number($("stake").value))} ずつ購入したと仮定した成績です。実購入額ではありません。`;
  $("updated").textContent = `最終更新 ${new Date(data.updated_at).toLocaleString("ja-JP", {timeZone: "Asia/Tokyo"})} JST`;
  $("sync").textContent = !data.sync.enabled ? "払戻の自動取得は停止中（保存済み結果のみ使用）。"
    : data.sync.running ? `確定払戻を取得中: ${data.sync.race_id} / 残り ${data.sync.remaining}レース。取得済みの分から順次反映されます。`
      : "払戻はバックグラウンドで自動取得します。未取得分は定期的に再試行します。";

  const selectedPlace = $("place").value;
  const places = [...new Set([...data.places, ...(selectedPlace ? [selectedPlace] : [])])];
  $("place").innerHTML = '<option value="">すべて</option>' + places.map(p => `<option value="${escapeHTML(p)}">${escapeHTML(p)}</option>`).join("");
  $("place").value = selectedPlace;
  const breakdown = periodBreakdown(data.daily, selection);
  $("breakdown-title").textContent = {month: "日別成績", year: "月別成績", all: "年別成績"}[selection.unit];
  $("breakdown-unit").textContent = {month: "開催日", year: "月", all: "年"}[selection.unit];
  $("breakdown-hint").hidden = selection.unit === "month";
  $("breakdown-hint").textContent = selection.unit === "year" ? "月をクリックすると、その月の成績を表示します。" : "年をクリックすると、その年の成績を表示します。";
  const tableRows = selection.unit === "year" ? breakdown : [...breakdown].reverse();
  $("daily").innerHTML = tableRows.map(d => {
    const label = selection.unit === "month" ? escapeHTML(d.date)
      : `<button type="button" class="period-link" data-drill="${escapeHTML(d.date)}">${selection.unit === "year" ? `${Number(d.date.slice(5))}月` : `${escapeHTML(d.date)}年`}</button>`;
    return `<tr><td>${label}</td><td>${yen(d.bet)}</td><td>${yen(d.return)}</td><td class="${tone(d.profit)}">${signed(d.profit)}</td><td>${rate(d.roi)}</td><td>${yen(d.pending_bet)}</td></tr>`;
  }).join("") || emptyRow(6);
  $("by-type").innerHTML = data.by_type.map(d => `<tr><td>${escapeHTML(d.type)}</td><td>${d.hits} / ${d.settled_points}</td><td>${yen(d.bet)}</td><td>${yen(d.return)}</td><td>${rate(d.roi)}</td></tr>`).join("") || emptyRow(5);
  renderChart(breakdown);

  const open = new Set([...document.querySelectorAll(".race[open]")].map(el => el.dataset.id));
  $("race-count").textContent = `${data.races.length}レース`;
  $("races").innerHTML = data.races.map(r => {
    const s = r.summary;
    const status = s.pending_bet ? "未判明あり" : s.hits ? "的中" : "不的中";
    return `<details class="race" data-id="${escapeHTML(r.id)}" ${open.has(r.id) ? "open" : ""}>
      <summary><div class="race-title"><b>${escapeHTML(`${r.place} ${r.number}R ${r.name}`)}</b><small>${escapeHTML(`${r.date} ${r.start.slice(11, 16)} JST · ${r.id}`)}</small></div>
      <span class="badge ${s.pending_bet ? "pending" : ""}">${status}</span><span class="race-stats">購入 ${yen(s.purchased)} / 払戻 ${yen(s.return)} / 回収率 ${rate(s.roi)}</span></summary>
      ${r.pending_reason ? `<p class="note">${escapeHTML(r.pending_reason)}。未判明の買い目は回収率・収支から除外しています。</p>` : ""}
      <div class="table-scroll"><table><thead><tr><th>式別</th><th>買い目</th><th>購入額</th><th>払戻額</th><th>結果</th></tr></thead><tbody>${r.tickets.map(t => `<tr><td>${escapeHTML(t.type)}</td><td>${escapeHTML(t.combo)}</td><td>${yen(t.bet)}</td><td>${t.return === null ? "—" : yen(t.return)}</td><td>${t.return === null ? "未判明" : t.return > 0 ? "的中" : "不的中"}</td></tr>`).join("")}</tbody></table></div></details>`;
  }).join("") || '<p class="empty">条件に一致する買い目がありません。集計対象や期間を変更してください。</p>';
  $("warnings-panel").hidden = !data.warnings.length;
  $("warnings").innerHTML = data.warnings.map(w => `<li>${escapeHTML(w)}</li>`).join("");
}

async function refresh() {
  if (currentRequest) currentRequest.abort();
  const controller = new AbortController();
  currentRequest = controller;
  $("refresh").disabled = true;
  try {
    const selection = {...period};
    const params = new URLSearchParams(filters.map(id => [id, $(id).value]));
    Object.entries(periodRange(selection)).forEach(([key, value]) => params.set(key, value));
    const response = await fetch(`/api/stats?${params}`, {cache: "no-store", signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    render(data, selection);
    $("error").hidden = true;
  } catch (error) {
    if (error.name !== "AbortError") {
      $("error").textContent = `更新できませんでした: ${error.message}。表示中の成績は前回取得時のものです。`;
      $("error").hidden = false;
    }
  } finally {
    if (currentRequest === controller) {
      currentRequest = null;
      $("refresh").disabled = false;
    }
  }
}

filters.forEach(id => $(id).addEventListener("change", () => {
  $("stake-label").hidden = $("mode").value !== "simulation";
  refresh();
}));
$("refresh").addEventListener("click", refresh);
$("reset").addEventListener("click", () => {
  ["place", "type"].forEach(id => { $(id).value = ""; });
  $("stake").value = "100";
  refresh();
});
document.querySelectorAll("[data-period]").forEach(button => button.addEventListener("click", () => {
  selectPeriod({unit: button.dataset.period});
}));
$("year").addEventListener("change", () => selectPeriod({year: Number($("year").value)}));
$("previous-period").addEventListener("click", () => movePeriod(-1));
$("next-period").addEventListener("click", () => movePeriod(1));
$("this-month").addEventListener("click", () => selectPeriod({unit: "month", ...currentMonth()}));
$("this-year").addEventListener("click", () => selectPeriod({unit: "year", ...currentMonth()}));
$("months").innerHTML = Array.from({length: 12}, (_, i) => `<button type="button" data-month="${i + 1}" aria-pressed="false">${i + 1}月</button>`).join("");
$("months").addEventListener("click", event => {
  const button = event.target.closest("[data-month]");
  if (button) selectPeriod({month: Number(button.dataset.month)});
});
$("daily").addEventListener("click", event => {
  const button = event.target.closest("[data-drill]");
  if (!button) return;
  const [year, month] = button.dataset.drill.split("-").map(Number);
  selectPeriod(month ? {unit: "month", year, month} : {unit: "year", year});
});
setInterval(() => { if ($("auto").checked && !document.hidden && !currentRequest) refresh(); }, 30000);
updatePeriodControls();
refresh();
