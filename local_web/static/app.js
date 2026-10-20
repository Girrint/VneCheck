const state = {
  status: null,
  reportFile: null,
  pageOffset: 0,
  pageSize: 20,
  totalRows: 0,
};
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.add("show");
  setTimeout(() => element.classList.remove("show"), 4000);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function percent(value) {
  if (value === "" || value === null || value === undefined) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function booleanValue(value) {
  return value === true || String(value).toLowerCase() === "true";
}

function fillSelect(select, items) {
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    select.append(option);
  });
}

function setBusy(button, busy, text) {
  if (!button.dataset.label) button.dataset.label = button.textContent;
  button.disabled = busy;
  button.textContent = busy ? text : button.dataset.label;
}

function labelFor(kind, value) {
  const items = state.status?.labels?.[kind] || [];
  return items.find((item) => item.value === value)?.label || value || "—";
}

function probabilityBars(values) {
  const entries = Object.entries(values || {}).filter(([, value]) => value !== "" && value !== null && value !== undefined);
  if (!entries.length) return "";
  return entries.sort((a, b) => Number(b[1]) - Number(a[1])).map(([name, value]) => `
    <div class="bar-row">
      <span>${escapeHtml(name)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Number(value) * 100}%"></div></div>
      <strong>${percent(value)}</strong>
    </div>`).join("");
}

function resultCard(data) {
  const truthCategory = labelFor("categories", data.truth.category);
  const truthFreshness = labelFor("freshness", data.truth.freshness);
  const unified = data.unified?.available ? `
    <article class="model-card">
      <div class="model-card-header"><strong>Unified · 1 ONNX</strong><span class="verdict ${data.unified.jointCorrect ? "" : "wrong"}">${data.unified.jointCorrect ? "Đúng" : "Sai"}</span></div>
      <div class="prediction-grid">
        <div class="prediction"><small>Category</small><strong>${escapeHtml(labelFor("categories", data.unified.category))}</strong><span>${percent(data.unified.categoryConfidence)}</span></div>
        <div class="prediction"><small>Freshness</small><strong>${escapeHtml(labelFor("freshness", data.unified.freshness))}</strong><span>${percent(data.unified.freshnessConfidence)}</span></div>
        <div class="prediction"><small>Điểm</small><strong>${escapeHtml(data.unified.score)}</strong><span>/ 10</span></div>
      </div>
      <div class="probabilities">
        <div class="probability-group"><h4>Category probabilities</h4>${probabilityBars(data.unified.categoryProbabilities)}</div>
        <div class="probability-group"><h4>Freshness probabilities</h4>${probabilityBars(data.unified.freshnessProbabilities)}</div>
      </div>
    </article>` : `<div class="model-missing">Unified ONNX chưa sẵn sàng.</div>`;

  const legacy = data.legacy?.available ? `
    <article class="model-card">
      <div class="model-card-header"><strong>Legacy · router → expert</strong><span class="verdict ${data.legacy.jointCorrect ? "" : "wrong"}">${data.legacy.jointCorrect ? "Đúng" : "Sai"}</span></div>
      <div class="prediction-grid">
        <div class="prediction"><small>Router</small><strong>${escapeHtml(labelFor("categories", data.legacy.category))}</strong><span>${percent(data.legacy.categoryConfidence)}</span></div>
        <div class="prediction"><small>Expert ${escapeHtml(data.legacy.classCount)} lớp</small><strong>${escapeHtml(labelFor("freshness", data.legacy.freshness))}</strong><span>${percent(data.legacy.freshnessConfidence)}</span></div>
        <div class="prediction"><small>Oracle expert</small><strong>${escapeHtml(labelFor("freshness", data.legacy.oracleFreshness))}</strong><span>${data.legacy.oracleCorrect ? "Đúng" : "Sai"} · ${escapeHtml(data.legacy.oracleClassCount)} lớp</span></div>
      </div>
      <div class="probabilities">
        <div class="probability-group"><h4>Router probabilities</h4>${probabilityBars(data.legacy.categoryProbabilities)}</div>
        <div class="probability-group"><h4>Pipeline expert probabilities</h4>${probabilityBars(data.legacy.freshnessProbabilities)}</div>
      </div>
    </article>` : `<div class="model-missing">Legacy ONNX chưa sẵn sàng.</div>`;

  return `<article class="image-result">
    <div class="sample-column">
      <img class="sample-image" loading="lazy" src="/api/image?path=${encodeURIComponent(data.path)}" alt="Ảnh test ${escapeHtml(data.path)}">
      <p class="sample-path" title="${escapeHtml(data.path)}">${escapeHtml(data.path)}</p>
      <div class="chips"><span class="chip">Truth: ${escapeHtml(truthCategory)}</span><span class="chip">${escapeHtml(truthFreshness)}</span><span class="chip">${escapeHtml(data.truth.product)}</span></div>
    </div>
    <div class="models-column">${unified}${legacy}</div>
  </article>`;
}

function singleToCard(result) {
  const sample = result.ground_truth;
  const unified = result.unified.ready ? {
    available: true,
    category: result.unified.prediction.category,
    categoryConfidence: result.unified.prediction.category_confidence,
    freshness: result.unified.prediction.freshness,
    freshnessConfidence: result.unified.prediction.freshness_confidence,
    score: result.unified.prediction.estimated_freshness_score,
    jointCorrect: result.unified.prediction.correct.joint,
    categoryProbabilities: result.unified.prediction.probabilities.category,
    freshnessProbabilities: result.unified.prediction.probabilities.freshness,
  } : { available: false };
  const legacy = result.legacy.ready ? {
    available: true,
    category: result.legacy.pipeline.category,
    categoryConfidence: result.legacy.pipeline.category_confidence,
    freshness: result.legacy.pipeline.freshness,
    freshnessConfidence: result.legacy.pipeline.freshness_confidence,
    classCount: result.legacy.pipeline.class_count,
    jointCorrect: result.legacy.pipeline.correct.joint,
    oracleFreshness: result.legacy.oracle.freshness,
    oracleCorrect: result.legacy.oracle.correct.freshness,
    oracleClassCount: result.legacy.oracle.class_count,
    categoryProbabilities: result.legacy.pipeline.category_probabilities,
    freshnessProbabilities: result.legacy.pipeline.freshness_probabilities,
  } : { available: false };
  return { path: sample.path, truth: sample, unified, legacy };
}

function csvToCard(row) {
  const collect = (prefix, names) => Object.fromEntries(names.map((name) => [name, row[`${prefix}${name}`]]).filter(([, value]) => value !== ""));
  return {
    path: row.image_path,
    truth: { category: row.truth_category, freshness: row.truth_freshness, product: row.truth_product },
    unified: {
      available: booleanValue(row.unified_available),
      category: row.unified_category,
      categoryConfidence: row.unified_category_confidence,
      freshness: row.unified_freshness,
      freshnessConfidence: row.unified_freshness_confidence,
      score: row.unified_score,
      jointCorrect: booleanValue(row.unified_joint_correct),
      categoryProbabilities: collect("unified_category_probability_", ["cu", "gia_cam", "gia_suc", "qua", "rau"]),
      freshnessProbabilities: collect("unified_freshness_probability_", ["FRESH", "HALF-FRESH", "SPOILED"]),
    },
    legacy: {
      available: booleanValue(row.legacy_available),
      category: row.legacy_router_category,
      categoryConfidence: row.legacy_router_confidence,
      freshness: row.legacy_pipeline_freshness,
      freshnessConfidence: row.legacy_pipeline_freshness_confidence,
      classCount: row.legacy_pipeline_expert_classes,
      jointCorrect: booleanValue(row.legacy_joint_correct),
      oracleFreshness: row.legacy_oracle_freshness,
      oracleCorrect: booleanValue(row.legacy_oracle_correct),
      oracleClassCount: row.legacy_oracle_expert_classes,
      categoryProbabilities: collect("legacy_router_probability_", ["cu", "gia_cam", "gia_suc", "qua", "rau"]),
      freshnessProbabilities: collect("legacy_pipeline_probability_", ["FRESH", "HALF-FRESH", "SPOILED"]),
    },
  };
}

function showLoading(title, text) {
  $("emptyState").classList.add("hidden");
  $("resultList").classList.add("hidden");
  $("pagination").classList.add("hidden");
  $("loadingState").classList.remove("hidden");
  $("loadingTitle").textContent = title;
  $("loadingText").textContent = text;
}

function showCards(cards) {
  $("loadingState").classList.add("hidden");
  $("emptyState").classList.add("hidden");
  $("resultList").innerHTML = cards.map(resultCard).join("");
  $("resultList").classList.remove("hidden");
}

async function loadStatus() {
  const status = await api("/api/status");
  state.status = status;
  fillSelect($("categoryFilter"), status.labels.categories);
  fillSelect($("freshnessFilter"), status.labels.freshness);
  $("datasetSummary").textContent = `${status.test_images.toLocaleString("vi-VN")} ảnh test đã lập chỉ mục`;
  const anyReady = status.model_ready || status.legacy_ready;
  const card = $("modelStatus");
  card.className = `status-card ${anyReady ? "ready" : "error"}`;
  card.querySelector("strong").textContent = `${status.model_ready ? "Unified ✓" : "Unified —"} · ${status.legacy_ready ? "Legacy ✓" : "Legacy —"}`;
  card.querySelector("small").textContent = `${status.test_images.toLocaleString("vi-VN")} ảnh test · ONNX Runtime CPU`;
  $("randomButton").disabled = !anyReady;
  $("evaluateButton").disabled = !anyReady;
}

async function evaluateOne() {
  const button = $("randomButton");
  setBusy(button, true, "Đang inference…");
  showLoading("Đang đánh giá một ảnh…", "Ảnh được lấy ngẫu nhiên từ bộ lọc bên trái.");
  try {
    const query = new URLSearchParams({ category: $("categoryFilter").value, freshness: $("freshnessFilter").value });
    const sample = await api(`/api/random?${query}`);
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: sample.path }),
    });
    state.reportFile = null;
    $("reportActions").classList.add("hidden");
    $("summaryStrip").classList.add("hidden");
    $("resultContext").textContent = "Một ảnh ngẫu nhiên từ test set";
    $("resultCount").textContent = "1 ảnh";
    showCards([singleToCard(result)]);
  } catch (error) {
    toast(error.message);
    $("loadingState").classList.add("hidden");
    $("emptyState").classList.remove("hidden");
  } finally { setBusy(button, false); }
}

function renderSummary(result) {
  $("summaryStrip").classList.remove("hidden");
  const unified = result.models.unified;
  $("unifiedSummary").textContent = unified.ready ? `${percent(unified.metrics.category_accuracy)} · ${percent(unified.metrics.freshness_accuracy)} · ${percent(unified.metrics.joint_accuracy)}` : "Chưa có model";
  const legacy = result.models.legacy;
  $("legacySummary").textContent = legacy.ready ? `${percent(legacy.metrics.pipeline_category_accuracy)} · ${percent(legacy.metrics.pipeline_freshness_accuracy)} · ${percent(legacy.metrics.pipeline_joint_accuracy)}` : "Chưa có model";
  $("oracleSummary").textContent = legacy.ready ? percent(legacy.metrics.oracle_freshness_accuracy) : "Chưa có model";
}

async function loadReportPage(offset) {
  if (!state.reportFile) return;
  showLoading("Đang tải kết quả từng ảnh…", "Đọc dữ liệu chi tiết từ CSV report.");
  const query = new URLSearchParams({ file: state.reportFile, offset, limit: state.pageSize });
  const page = await api(`/api/report_rows?${query}`);
  state.pageOffset = page.offset;
  state.totalRows = page.total;
  showCards(page.rows.map(csvToCard));
  const pageNumber = Math.floor(page.offset / page.limit) + 1;
  const totalPages = Math.max(1, Math.ceil(page.total / page.limit));
  $("pageInfo").textContent = `Trang ${pageNumber} / ${totalPages}`;
  $("previousPage").disabled = !page.has_previous;
  $("nextPage").disabled = !page.has_next;
  $("pagination").classList.toggle("hidden", page.total <= page.limit);
}

async function evaluateBatch() {
  const button = $("evaluateButton");
  const all = $("batchSize").value === "0";
  setBusy(button, true, all ? "Đang chạy toàn bộ…" : "Đang inference…");
  showLoading(all ? "Đang đánh giá toàn bộ ảnh…" : "Đang đánh giá batch…", "Kết quả được ghi dần ra CSV; thời gian phụ thuộc số ảnh và CPU.");
  try {
    const result = await api("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        category: $("categoryFilter").value,
        freshness: $("freshnessFilter").value,
        limit: Number($("batchSize").value),
        seed: 42,
      }),
    });
    state.reportFile = result.reports.csv.filename;
    $("downloadCsv").href = result.reports.csv.url;
    $("downloadCsv").download = result.reports.csv.filename;
    $("downloadJson").href = result.reports.summary_json.url;
    $("downloadJson").download = result.reports.summary_json.filename;
    $("reportActions").classList.remove("hidden");
    $("resultContext").textContent = result.evaluated_all ? "Toàn bộ ảnh phù hợp bộ lọc" : `Mẫu ngẫu nhiên seed 42 · ${result.sample_count}/${result.candidate_count} ảnh`;
    $("resultCount").textContent = `${result.sample_count.toLocaleString("vi-VN")} ảnh`;
    renderSummary(result);
    await loadReportPage(0);
  } catch (error) {
    toast(error.message);
    $("loadingState").classList.add("hidden");
    $("emptyState").classList.remove("hidden");
  } finally { setBusy(button, false); }
}

$("randomButton").addEventListener("click", evaluateOne);
$("evaluateButton").addEventListener("click", evaluateBatch);
$("previousPage").addEventListener("click", () => loadReportPage(Math.max(0, state.pageOffset - state.pageSize)).catch((error) => toast(error.message)));
$("nextPage").addEventListener("click", () => loadReportPage(state.pageOffset + state.pageSize).catch((error) => toast(error.message)));
loadStatus().catch((error) => toast(error.message));
