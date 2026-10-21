const state = {
  status: null,
  reportFile: null,
  pageOffset: 0,
  pageSize: 10,
  totalRows: 0,
  lastRandomPath: null,
  pageCards: [],
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

function selectedIds(containerId) {
  return [...$(containerId).querySelectorAll('input[type="checkbox"]:checked:not(:disabled)')]
    .map((input) => input.value);
}

function renderPicker(containerId, items, defaultId, type) {
  const container = $(containerId);
  container.innerHTML = "";
  if (!items.length) {
    container.innerHTML = `<small>Không tìm thấy ${type === "unified" ? "Unified ONNX" : "Legacy bundle"}.</small>`;
    return;
  }
  const selectedId = items.some((item) => item.id === defaultId && item.ready)
    ? defaultId
    : items.find((item) => item.ready)?.id;
  items.forEach((item) => {
    const id = `${type}-${item.id}`.replace(/[^a-zA-Z0-9_-]/g, "_");
    const label = document.createElement("label");
    label.className = `model-choice ${item.ready ? "" : "disabled"}`;
    const architecture = type === "unified"
      ? item.architecture
      : `router ${item.router_architecture} · expert ${item.expert_architecture_summary}`;
    label.innerHTML = `
      <input id="${escapeHtml(id)}" type="checkbox" value="${escapeHtml(item.id)}"
        data-invalid="${String(!item.ready)}"
        data-food-gate="${String(Boolean(item.food_gate_ready))}"
        ${item.id === selectedId ? "checked" : ""}
        ${item.ready ? "" : "disabled"}>
      <span>${escapeHtml(item.display_name)}<small>${escapeHtml(architecture || "unknown")}${item.ready ? "" : ` · ${escapeHtml(item.error || "lỗi")}`}</small></span>`;
    container.append(label);
  });
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

function modelEvaluationCard(data) {
  const isSupported = data.truth.food_status === "SUPPORTED";
  const supportedPredictions = isSupported && data.unified?.available ? `
    <div class="prediction"><small>Category</small><strong>${escapeHtml(labelFor("categories", data.unified.category))}</strong><span>${percent(data.unified.categoryConfidence)}</span></div>
    <div class="prediction"><small>Freshness</small><strong>${escapeHtml(labelFor("freshness", data.unified.freshness))}</strong><span>${percent(data.unified.freshnessConfidence)}</span></div>
    <div class="prediction"><small>Điểm</small><strong>${escapeHtml(data.unified.score)}</strong><span>/ 10</span></div>` : "";
  const supportedProbabilities = isSupported && data.unified?.available ? `
    <div class="probability-group"><h4>Category probabilities</h4>${probabilityBars(data.unified.categoryProbabilities)}</div>
    <div class="probability-group"><h4>Freshness probabilities</h4>${probabilityBars(data.unified.freshnessProbabilities)}</div>` : "";
  const unified = data.unified ? (data.unified.available ? `
    <article class="model-card">
      <div class="model-card-header"><strong>${escapeHtml(data.unified.modelName || "Unified")} · ${escapeHtml(data.unified.architecture || "ONNX")}</strong><span class="verdict ${data.unified.jointCorrect ? "" : "wrong"}">${data.unified.jointCorrect ? "Đúng" : "Sai"}</span></div>
      <div class="prediction-grid">
        <div class="prediction"><small>Food status</small><strong>${escapeHtml(labelFor("food_statuses", data.unified.foodStatus))}</strong><span>${percent(data.unified.foodConfidence)}</span></div>
        ${supportedPredictions}
      </div>
      <div class="probabilities">
        <div class="probability-group"><h4>Food status probabilities</h4>${probabilityBars(data.unified.foodProbabilities)}</div>
        ${supportedProbabilities}
      </div>
    </article>` : `<div class="model-missing">${escapeHtml(data.unified.modelName || "Unified")}: ${escapeHtml(data.unified.error || "ONNX chưa sẵn sàng")}</div>`) : "";

  const legacyPipeline = data.legacy?.category ? `
        <div class="prediction"><small>Category</small><strong>${escapeHtml(labelFor("categories", data.legacy.category))}</strong><span>${percent(data.legacy.categoryConfidence)}</span></div>
        <div class="prediction"><small>Expert ${escapeHtml(data.legacy.classCount)}</small><strong>${escapeHtml(labelFor("freshness", data.legacy.freshness))}</strong><span>${percent(data.legacy.freshnessConfidence)}</span></div>` : "";
  const legacyOracle = data.legacy?.oracleFreshness ? `
        <div class="prediction"><small>Oracle expert</small><strong>${escapeHtml(labelFor("freshness", data.legacy.oracleFreshness))}</strong><span>${data.legacy.oracleCorrect ? "Đúng" : "Sai"} · ${escapeHtml(data.legacy.oracleClassCount)} lớp</span></div>` : "";
  const legacy = data.legacy ? (data.legacy.available ? `
    <article class="model-card">
      <div class="model-card-header"><strong>${escapeHtml(data.legacy.bundleName || "Legacy")} · ${escapeHtml(data.legacy.routerArchitecture || "router")} → expert</strong><span class="verdict ${data.legacy.jointCorrect ? "" : "wrong"}">${data.legacy.jointCorrect ? "Đúng" : "Sai"}</span></div>
      <div class="prediction-grid">
        <div class="prediction"><small>Food status</small><strong>${escapeHtml(labelFor("food_statuses", data.legacy.foodStatus))}</strong><span>${percent(data.legacy.foodConfidence)}</span></div>
        ${legacyPipeline}${legacyOracle}
      </div>
      <div class="probabilities">
        <div class="probability-group"><h4>Food status probabilities</h4>${probabilityBars(data.legacy.foodProbabilities)}</div>
        <div class="probability-group"><h4>Router 7 lớp</h4>${probabilityBars(data.legacy.categoryProbabilities)}</div>
        ${data.legacy.freshness ? `<div class="probability-group"><h4>Pipeline expert probabilities</h4>${probabilityBars(data.legacy.freshnessProbabilities)}</div>` : ""}
      </div>
    </article>` : `<div class="model-missing">${escapeHtml(data.legacy.bundleName || "Legacy")}: ${escapeHtml(data.legacy.error || (data.legacy.applicable === false ? "Không áp dụng cho ảnh negative" : "ONNX chưa sẵn sàng"))}</div>`) : "";

  return `${unified}${legacy}`;
}

function groupCardsByImage(cards) {
  const groups = new Map();
  cards.forEach((card) => {
    let group = groups.get(card.path);
    if (!group) {
      group = {
        path: card.path,
        truth: card.truth,
        evaluations: [],
      };
      groups.set(card.path, group);
    }
    group.evaluations.push(card);
  });
  return [...groups.values()];
}

function filterCardsByCorrectness(cards) {
  const filter = $("resultCorrectnessFilter").value;
  if (filter === "all") return cards;
  return cards.filter((card) => {
    const evaluation = card.unified || card.legacy;
    if (!evaluation?.available) return false;
    return filter === "correct"
      ? Boolean(evaluation.jointCorrect)
      : !Boolean(evaluation.jointCorrect);
  });
}

function resultCard(data) {
  const truthFood = labelFor("food_statuses", data.truth.food_status);
  const truthCategory = labelFor("categories", data.truth.category);
  const truthFreshness = labelFor("freshness", data.truth.freshness);
  const isSupported = data.truth.food_status === "SUPPORTED";
  const truthChips = isSupported
    ? `<span class="chip">${escapeHtml(truthCategory)}</span><span class="chip">${escapeHtml(truthFreshness)}</span><span class="chip">${escapeHtml(data.truth.product)}</span>`
    : "";
  const evaluations = data.evaluations.map(modelEvaluationCard).join("");
  return `<article class="image-result">
    <div class="sample-column">
      <img class="sample-image" loading="lazy" src="/api/image?path=${encodeURIComponent(data.path)}" alt="Ảnh test ${escapeHtml(data.path)}">
      <p class="sample-path" title="${escapeHtml(data.path)}">${escapeHtml(data.path)}</p>
      <div class="chips"><span class="chip">Truth: ${escapeHtml(truthFood)}</span>${truthChips}</div>
    </div>
    <div class="models-column">${evaluations}</div>
  </article>`;
}

function unifiedEntryToCard(sample, entry) {
  const prediction = entry.prediction;
  return {
    path: sample.path,
    truth: sample,
    unified: prediction ? {
      available: true,
      modelName: entry.model?.display_name || prediction.model?.display_name,
      architecture: entry.model?.architecture || prediction.model?.architecture,
      foodStatus: prediction.food_status,
      foodConfidence: prediction.food_confidence,
      category: prediction.category,
      categoryConfidence: prediction.category_confidence,
      freshness: prediction.freshness,
      freshnessConfidence: prediction.freshness_confidence,
      score: prediction.estimated_freshness_score,
      jointCorrect: prediction.correct.joint,
      foodProbabilities: prediction.probabilities.food,
      categoryProbabilities: prediction.probabilities.category,
      freshnessProbabilities: prediction.probabilities.freshness,
    } : {
      available: false,
      modelName: entry.model?.display_name,
      error: entry.error,
    },
  };
}

function legacyEntryToCard(sample, entry) {
  return {
    path: sample.path,
    truth: sample,
    legacy: entry.ready ? {
      available: true,
      bundleName: entry.bundle?.display_name,
      routerArchitecture: entry.bundle?.router_architecture,
      foodStatus: entry.food_status,
      foodConfidence: entry.food_confidence,
      foodProbabilities: entry.food_probabilities,
      category: entry.pipeline?.category,
      categoryConfidence: entry.pipeline?.category_confidence,
      freshness: entry.pipeline?.freshness,
      freshnessConfidence: entry.pipeline?.freshness_confidence,
      classCount: entry.pipeline?.class_count,
      jointCorrect: entry.joint_correct,
      oracleFreshness: entry.oracle?.freshness,
      oracleCorrect: entry.oracle?.correct?.freshness,
      oracleClassCount: entry.oracle?.class_count,
      categoryProbabilities: entry.router_probabilities,
      freshnessProbabilities: entry.pipeline?.freshness_probabilities,
    } : {
      available: false,
      applicable: entry.applicable !== false,
      bundleName: entry.bundle?.display_name,
      error: entry.error,
    },
  };
}

function singleToCards(result) {
  const sample = result.ground_truth;
  return [
    ...(result.unified_models || []).map((entry) => unifiedEntryToCard(sample, entry)),
    ...(result.legacy_models || []).map((entry) => legacyEntryToCard(sample, entry)),
  ];
}

function csvToCard(row) {
  const collect = (prefix, names) => Object.fromEntries(
    names.map((name) => [name, row[`${prefix}${name}`]]).filter(([, value]) => value !== "")
  );
  const base = {
    path: row.image_path,
    truth: {
      food_status: row.truth_food_status,
      category: row.truth_category,
      freshness: row.truth_freshness,
      product: row.truth_product,
    },
  };
  if (row.evaluation_model_type === "legacy" || row.legacy_bundle_id) {
    const routedCategory = ["cu", "gia_cam", "gia_suc", "qua", "rau"].includes(row.legacy_router_category)
      ? row.legacy_router_category : "";
    base.legacy = {
      available: booleanValue(row.legacy_available),
      applicable: booleanValue(row.evaluation_applicable),
      bundleName: row.legacy_bundle_name || row.evaluation_model_name,
      routerArchitecture: row.legacy_router_architecture || row.evaluation_model_architecture,
      error: row.evaluation_error,
      foodStatus: row.legacy_food_status || (row.legacy_router_category && ["UNSUPPORTED", "NON_FOOD"].includes(row.legacy_router_category) ? row.legacy_router_category : "SUPPORTED"),
      foodConfidence: row.legacy_food_confidence || row.legacy_router_confidence,
      foodProbabilities: collect("legacy_food_probability_", ["SUPPORTED", "UNSUPPORTED", "NON_FOOD"]),
      category: routedCategory,
      categoryConfidence: row.legacy_router_confidence,
      freshness: row.legacy_pipeline_freshness,
      freshnessConfidence: row.legacy_pipeline_freshness_confidence,
      classCount: row.legacy_pipeline_expert_classes,
      jointCorrect: booleanValue(row.legacy_joint_correct),
      oracleFreshness: row.legacy_oracle_freshness,
      oracleCorrect: booleanValue(row.legacy_oracle_correct),
      oracleClassCount: row.legacy_oracle_expert_classes,
      categoryProbabilities: collect("legacy_router_probability_", ["cu", "gia_cam", "gia_suc", "qua", "rau", "UNSUPPORTED", "NON_FOOD"]),
      freshnessProbabilities: collect("legacy_pipeline_probability_", ["FRESH", "HALF-FRESH", "SPOILED"]),
    };
  } else {
    base.unified = {
      available: booleanValue(row.unified_available),
      modelName: row.unified_model_name || row.evaluation_model_name,
      architecture: row.unified_architecture || row.evaluation_model_architecture,
      error: row.evaluation_error,
      foodStatus: row.unified_food_status,
      foodConfidence: row.unified_food_confidence,
      category: row.unified_category,
      categoryConfidence: row.unified_category_confidence,
      freshness: row.unified_freshness,
      freshnessConfidence: row.unified_freshness_confidence,
      score: row.unified_score,
      jointCorrect: booleanValue(row.unified_joint_correct),
      foodProbabilities: collect("unified_food_probability_", ["SUPPORTED", "UNSUPPORTED", "NON_FOOD"]),
      categoryProbabilities: collect("unified_category_probability_", ["cu", "gia_cam", "gia_suc", "qua", "rau"]),
      freshnessProbabilities: collect("unified_freshness_probability_", ["FRESH", "HALF-FRESH", "SPOILED"]),
    };
  }
  return base;
}

function showLoading(title, text) {
  $("emptyState").classList.add("hidden");
  $("resultList").classList.add("hidden");
  $("pagination").classList.add("hidden");
  $("loadingState").classList.remove("hidden");
  $("loadingTitle").textContent = title;
  $("loadingText").textContent = text;
}

function showCards(cards, remember = true) {
  if (remember) state.pageCards = cards;
  const filtered = filterCardsByCorrectness(cards);
  const groups = groupCardsByImage(filtered);
  $("loadingState").classList.add("hidden");
  $("emptyState").classList.toggle("hidden", groups.length > 0);
  $("resultList").innerHTML = groups.map(resultCard).join("");
  $("resultList").classList.toggle("hidden", groups.length === 0);
}

async function loadStatus() {
  const status = await api("/api/status");
  state.status = status;
  renderPicker("unifiedModelFilter", status.unified_models, status.default_unified_model_id, "unified");
  renderPicker("legacyBundleFilter", status.legacy_bundles, status.default_legacy_bundle_id, "legacy");
  fillSelect($("foodStatusFilter"), status.labels.food_statuses);
  fillSelect($("categoryFilter"), status.labels.categories);
  fillSelect($("freshnessFilter"), status.labels.freshness);
  $("datasetSummary").textContent = `${status.test_images.toLocaleString("vi-VN")} ảnh test đã lập chỉ mục`;
  const unifiedReady = status.unified_models.filter((model) => model.ready).length;
  const legacyReady = status.legacy_bundles.filter((bundle) => bundle.ready).length;
  const anyReady = unifiedReady + legacyReady > 0;
  const card = $("modelStatus");
  card.className = `status-card ${anyReady ? "ready" : "error"}`;
  card.querySelector("strong").textContent = `${unifiedReady} Unified · ${legacyReady} Legacy`;
  card.querySelector("small").textContent = `${status.test_images.toLocaleString("vi-VN")} ảnh test · ONNX Runtime CPU`;
  syncControls();
}

function syncControls() {
  const negative = ["UNSUPPORTED", "NON_FOOD"].includes($("foodStatusFilter").value);
  if (negative) {
    $("categoryFilter").value = "";
    $("freshnessFilter").value = "";
  }
  $("categoryFilter").disabled = negative;
  $("freshnessFilter").disabled = negative;
  const legacyOption = $("modelFilter").querySelector('option[value="legacy"]');
  const unifiedOption = $("modelFilter").querySelector('option[value="unified"]');
  const unifiedReady = state.status?.unified_models?.some((model) => model.ready);
  const legacyReady = state.status?.legacy_bundles?.some(
    (bundle) => bundle.ready && (!negative || bundle.food_gate_ready)
  );
  legacyOption.disabled = !legacyReady;
  unifiedOption.disabled = !unifiedReady;
  if (legacyOption.disabled && $("modelFilter").value === "legacy") $("modelFilter").value = "unified";

  const mode = $("modelFilter").value;
  $("unifiedModelFilter").querySelectorAll("input").forEach((input) => {
    input.disabled = input.dataset.invalid === "true" || mode === "legacy";
    input.closest("label")?.classList.toggle("disabled", input.disabled);
  });
  $("legacyBundleFilter").querySelectorAll("input").forEach((input) => {
    input.disabled = input.dataset.invalid === "true" || mode === "unified"
      || (negative && input.dataset.foodGate !== "true");
    input.closest("label")?.classList.toggle("disabled", input.disabled);
  });
  const unifiedCount = selectedIds("unifiedModelFilter").length;
  const legacyCount = selectedIds("legacyBundleFilter").length;
  const runnable = mode === "unified"
    ? unifiedCount > 0
    : mode === "legacy"
      ? legacyCount > 0
      : unifiedCount + legacyCount > 0;
  $("randomButton").disabled = !runnable;
  $("evaluateButton").disabled = !runnable;
}

function selectedPayload() {
  return {
    model_mode: $("modelFilter").value,
    unified_model_ids: selectedIds("unifiedModelFilter"),
    legacy_bundle_ids: selectedIds("legacyBundleFilter"),
  };
}

async function evaluateOne() {
  const button = $("randomButton");
  setBusy(button, true, "Đang inference…");
  showLoading("Đang đánh giá một ảnh…", "Ảnh được chạy qua tất cả model đã chọn.");
  try {
    const query = new URLSearchParams({
      food_status: $("foodStatusFilter").value,
      category: $("categoryFilter").value,
      freshness: $("freshnessFilter").value,
    });
    if (state.lastRandomPath) query.set("exclude_path", state.lastRandomPath);
    const sample = await api(`/api/random?${query}`);
    state.lastRandomPath = sample.path;
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: sample.path, ...selectedPayload() }),
    });
    const cards = singleToCards(result);
    state.reportFile = null;
    $("reportActions").classList.add("hidden");
    $("summaryStrip").classList.add("hidden");
    $("resultContext").textContent = "Một ảnh · tất cả model đã chọn";
    $("resultCount").textContent = `${cards.length} kết quả model`;
    showCards(cards);
  } catch (error) {
    toast(error.message);
    $("loadingState").classList.add("hidden");
    $("emptyState").classList.remove("hidden");
  } finally { setBusy(button, false); }
}

function renderSummary(result) {
  $("summaryStrip").classList.remove("hidden");
  const unified = result.models.unified || [];
  const unifiedReady = unified.filter((item) => item.ready);
  $("unifiedSummary").textContent = `${unifiedReady.length}/${unified.length} model`;
  $("unifiedSubSummary").innerHTML = unified.length ? unified.map((item) => item.ready
    ? `<b>${escapeHtml(item.model.display_name)}</b> · Food ${percent(item.metrics.food_accuracy)} · Cat ${percent(item.metrics.category_accuracy)} · Fresh ${percent(item.metrics.freshness_accuracy)} · Joint ${percent(item.metrics.joint_accuracy)}`
    : `<b>${escapeHtml(item.model?.display_name || "Unified")}</b> · ${escapeHtml(item.error || "Không sẵn sàng")}`
  ).join("<br>") : "Không chọn Unified";

  const legacy = result.models.legacy || [];
  const legacyReady = legacy.filter((item) => item.ready);
  $("legacySummary").textContent = `${legacyReady.length}/${legacy.length} bundle`;
  $("legacySubSummary").innerHTML = legacy.length ? legacy.map((item) => item.ready
    ? `<b>${escapeHtml(item.bundle.display_name)}</b> · Food ${percent(item.metrics.food_accuracy)} · Cat ${percent(item.metrics.pipeline_category_accuracy)} · Fresh ${percent(item.metrics.pipeline_freshness_accuracy)} · Joint ${percent(item.metrics.pipeline_joint_accuracy)}`
    : `<b>${escapeHtml(item.bundle?.display_name || "Legacy")}</b> · ${escapeHtml(item.error || "Không áp dụng")}`
  ).join("<br>") : "Không chọn Legacy";
  $("oracleSummary").textContent = legacyReady.length ? `${legacyReady.length} bundle` : "Chưa có";
  $("oracleSummary").nextElementSibling.innerHTML = legacyReady.length
    ? legacyReady.map((item) => `<b>${escapeHtml(item.bundle.display_name)}</b> · ${percent(item.metrics.oracle_freshness_accuracy)}`).join("<br>")
    : "Freshness accuracy";
}

async function loadReportPage(offset) {
  if (!state.reportFile) return;
  showLoading("Đang tải kết quả theo ảnh…", "Mỗi ảnh hiển thị tất cả model đã chọn cạnh nhau.");
  const query = new URLSearchParams({
    file: state.reportFile,
    offset,
    limit: state.pageSize,
    correctness: $("resultCorrectnessFilter").value,
  });
  const page = await api(`/api/report_rows?${query}`);
  state.pageOffset = page.offset;
  state.totalRows = page.total;
  showCards(page.rows.map(csvToCard));
  $("resultCount").textContent = `${page.total.toLocaleString("vi-VN")} ảnh · ${page.row_total.toLocaleString("vi-VN")} kết quả`;
  const pageNumber = Math.floor(page.offset / page.limit) + 1;
  const totalPages = Math.max(1, Math.ceil(page.total / page.limit));
  $("pageInfo").textContent = `Trang ${pageNumber} / ${totalPages}`;
  $("previousPage").disabled = !page.has_previous;
  $("nextPage").disabled = !page.has_next;
  $("pagination").classList.toggle("hidden", page.total <= page.limit);
}

async function evaluateBatch() {
  const button = $("evaluateButton");
  const rawBatchSize = $("batchSize").value.trim();
  const requested = Number(rawBatchSize);
  if (!rawBatchSize || !Number.isInteger(requested) || requested < 0) {
    toast("Số ảnh phải là số nguyên từ 0 trở lên");
    $("batchSize").focus();
    return;
  }
  const all = requested === 0;
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  setBusy(button, true, all ? "Đang chạy toàn bộ…" : "Đang inference…");
  showLoading(all ? "Đang đánh giá toàn bộ ảnh…" : "Đang đánh giá batch…", "Mọi model được chạy tuần tự trên cùng danh sách ảnh.");
  try {
    const result = await api("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...selectedPayload(),
        food_status: $("foodStatusFilter").value,
        category: $("categoryFilter").value,
        freshness: $("freshnessFilter").value,
        limit: requested,
        seed,
      }),
    });
    state.reportFile = result.reports.csv.filename;
    $("downloadCsv").href = result.reports.csv.url;
    $("downloadCsv").download = result.reports.csv.filename;
    $("downloadJson").href = result.reports.summary_json.url;
    $("downloadJson").download = result.reports.summary_json.filename;
    $("reportActions").classList.remove("hidden");
    $("resultContext").textContent = result.evaluated_all
      ? "Toàn bộ ảnh phù hợp bộ lọc"
      : `Mẫu seed ${result.filters.seed} · ${result.sample_count}/${result.candidate_count} ảnh`;
    $("resultCount").textContent = `${result.sample_count.toLocaleString("vi-VN")} ảnh · ${result.result_count.toLocaleString("vi-VN")} kết quả`;
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
$("foodStatusFilter").addEventListener("change", syncControls);
$("modelFilter").addEventListener("change", syncControls);
$("unifiedModelFilter").addEventListener("change", syncControls);
$("legacyBundleFilter").addEventListener("change", syncControls);
$("resultCorrectnessFilter").addEventListener("change", () => {
  if (state.reportFile) {
    loadReportPage(0).catch((error) => toast(error.message));
    return;
  }
  const filtered = filterCardsByCorrectness(state.pageCards);
  $("resultCount").textContent = `${filtered.length} kết quả model`;
  showCards(state.pageCards, false);
});
$("previousPage").addEventListener("click", () => loadReportPage(Math.max(0, state.pageOffset - state.pageSize)).catch((error) => toast(error.message)));
$("nextPage").addEventListener("click", () => loadReportPage(state.pageOffset + state.pageSize).catch((error) => toast(error.message)));
loadStatus().catch((error) => toast(error.message));
