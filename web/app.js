const PRODUCE = [
  'apple', 'banana', 'bellpepper', 'carrot', 'cucumber',
  'mango', 'orange', 'potato', 'strawberry', 'tomato',
];

const ARCHITECTURES = {
  mobilenet_v3_large: 'MobileNetV3-Large',
  resnet18: 'ResNet-18',
  efficientnet_b0: 'EfficientNet-B0',
};

const COLORS = ['#c8f169', '#ff865b', '#67c7a2', '#f5c85b', '#8ea8ff', '#e98bb8'];
const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
const state = {
  manifest: null,
  bitmap: null,
  file: null,
  detections: [],
  detectorResult: null,
  report: null,
  items: [],
  reports: [],
  activeImageIndex: -1,
  sessions: new Map(),
  busy: false,
  activeTab: 'detection',
  selectedDetectionIndex: 0,
  uploadErrors: [],
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  runtimeStatus: $('#runtime-status'),
  pulse: $('.pulse'),
  detectorCount: $('#detector-count'),
  classifierCount: $('#classifier-count'),
  modelNotice: $('#model-notice'),
  inferencePanel: $('.stage-card'),
  runButtonLabel: $('#run-button-label'),
  settingsModelCount: $('#settings-model-count'),
  provider: $('#provider'),
  architecture: $('#architecture'),
  architectureField: $('#architecture-field'),
  repeats: $('#repeats'),
  repeatsField: $('#repeats-field'),
  repeatValue: $('#repeat-value'),
  confidence: $('#confidence'),
  confidenceValue: $('#confidence-value'),
  confidenceField: $('#confidence-field'),
  iou: $('#iou'),
  iouValue: $('#iou-value'),
  iouField: $('#iou-field'),
  demoMode: $('#demo-mode'),
  demoField: $('#demo-field'),
  dropZone: $('#drop-zone'),
  imageInput: $('#image-input'),
  uploadIdle: $('#upload-idle'),
  addImages: $('#add-images'),
  canvas: $('#preview-canvas'),
  clearImage: $('#clear-image'),
  imageMeta: $('#image-meta'),
  queueEmpty: $('#queue-empty'),
  batchList: $('#batch-list'),
  batchCount: $('#batch-count'),
  batchStrip: $('#batch-strip'),
  batchSummary: $('#batch-summary'),
  batchTotal: $('#batch-total'),
  batchProcessed: $('#batch-processed'),
  batchFailed: $('#batch-failed'),
  batchSummaryBar: $('#batch-summary-bar'),
  batchSummaryProgress: $('#batch-summary-progress'),
  batchErrors: $('#batch-errors'),
  batchErrorCount: $('#batch-error-count'),
  batchErrorList: $('#batch-error-list'),
  resultBatchStrip: $('#result-batch-strip'),
  runButton: $('#run-button'),
  progressWrap: $('#progress-wrap'),
  progressBar: $('#progress-bar'),
  progressLine: $('#progress-line'),
  progressText: $('#progress-text'),
  emptyResults: $('#empty-results'),
  resultContent: $('#result-content'),
  detectionEmpty: $('#detection-empty'),
  detectionTabCount: $('#detection-tab-count'),
  resultsTabCount: $('#results-tab-count'),
  benchmarkTabCount: $('#benchmark-tab-count'),
  tabButtons: [...document.querySelectorAll('[role="tab"][data-tab]')],
  tabPanels: [...document.querySelectorAll('[role="tabpanel"]')],
  aiItemsGrid: $('#ai-items-grid'),
  aiModelList: $('#ai-model-list'),
  aiModeBadge: $('#ai-mode-badge'),
  detectionSummaryGrid: $('#detection-summary-grid'),
  classificationSummaryGrid: $('#classification-summary-grid'),
  runtimeSummaryGrid: $('#runtime-summary-grid'),
  detectionOutput: $('#detection-output'),
  benchmarkOutput: $('#benchmark-output'),
  cropGallery: $('#crop-gallery'),
  selectedItemDetail: $('#selected-item-detail'),
  benchmarkRoiList: $('#benchmark-roi-list'),
  benchmarkEmpty: $('#benchmark-empty'),
  benchmarkTable: $('#benchmark-table'),
  boxCountLabel: $('#box-count-label'),
  benchmarkBody: $('#benchmark-body'),
  benchmarkBadge: $('#benchmark-badge'),
  technicalDetails: $('#technical-details'),
  detectionDetails: $('#detection-details'),
  exportJson: $('#export-json'),
  exportCsv: $('#export-csv'),
  toast: $('#toast'),
};

function toast(message, error = false) {
  elements.toast.textContent = message;
  elements.toast.className = `toast show${error ? ' error' : ''}`;
  elements.toast.setAttribute('role', error ? 'alert' : 'status');
  elements.toast.setAttribute('aria-live', error ? 'assertive' : 'polite');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { elements.toast.className = 'toast'; }, 3600);
}

function setProgress(percent, message) {
  const value = Math.max(0, Math.min(100, percent));
  elements.progressWrap.hidden = false;
  elements.progressBar.style.width = `${value}%`;
  elements.progressLine.setAttribute('aria-valuenow', Math.round(value));
  elements.progressLine.setAttribute('aria-valuetext', message);
  elements.progressText.textContent = message;
}

function setRuntimeStatus(text, status = '') {
  elements.runtimeStatus.textContent = text;
  elements.pulse.className = `pulse ${status}`.trim();
}

async function loadManifest() {
  const response = await fetch(`./models/manifest.json?t=${Date.now()}`, { cache: 'no-store' });
  if (!response.ok) throw new Error('Không đọc được models/manifest.json');
  state.manifest = await response.json();
  const classifierCount = Object.values(state.manifest.classifiers)
    .reduce((sum, item) => sum + (item.available?.length || 0), 0);
  elements.detectorCount.textContent = `${state.manifest.detector.available ? 1 : 0}/1`;
  elements.classifierCount.textContent = `${classifierCount}/30`;
  elements.settingsModelCount.textContent = `${classifierCount + (state.manifest.detector.available ? 1 : 0)}/31 ONNX`;
  const allReady = state.manifest.detector.available && classifierCount === 30;
  elements.modelNotice.classList.toggle('ready', allReady);
  elements.modelNotice.textContent = allReady
    ? 'Đã tìm thấy đủ 31 model ONNX. Web sẵn sàng chạy benchmark thật.'
    : `Hiện có ${state.manifest.detector.available ? 1 : 0}/1 detector và ${classifierCount}/30 classifier. Có thể dùng Demo giao diện trong lúc chờ train.`;
}

function initRuntime() {
  if (!globalThis.ort) throw new Error('ONNX Runtime Web chưa tải được. Kiểm tra kết nối Internet.');
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  if (!navigator.gpu) {
    elements.provider.value = 'wasm';
    setRuntimeStatus('WASM sẵn sàng', 'ready');
  } else {
    setRuntimeStatus('WebGPU sẵn sàng', 'ready');
  }
}

async function init() {
  try {
    await Promise.all([loadManifest(), document.fonts?.ready || Promise.resolve()]);
    initRuntime();
  } catch (error) {
    console.error(error);
    setRuntimeStatus('Runtime chưa sẵn sàng', 'error');
    toast(error.message, true);
  }
  bindEvents();
  switchTab('detection', false);
  updateRunState();
}

function bindEvents() {
  elements.tabButtons.forEach((button, buttonIndex) => {
    button.addEventListener('click', () => switchTab(button.dataset.tab));
    button.addEventListener('keydown', (event) => {
      const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      let nextIndex = buttonIndex;
      if (event.key === 'ArrowLeft') nextIndex = (buttonIndex - 1 + elements.tabButtons.length) % elements.tabButtons.length;
      if (event.key === 'ArrowRight') nextIndex = (buttonIndex + 1) % elements.tabButtons.length;
      if (event.key === 'Home') nextIndex = 0;
      if (event.key === 'End') nextIndex = elements.tabButtons.length - 1;
      const nextButton = elements.tabButtons[nextIndex];
      switchTab(nextButton.dataset.tab);
      nextButton.focus({ preventScroll: true });
    });
  });
  elements.repeats.addEventListener('input', () => { elements.repeatValue.value = elements.repeats.value; });
  elements.confidence.addEventListener('input', () => { elements.confidenceValue.value = Number(elements.confidence.value).toFixed(2); });
  elements.iou.addEventListener('input', () => { elements.iouValue.value = Number(elements.iou.value).toFixed(2); });
  elements.demoMode.addEventListener('change', updateRunState);
  elements.provider.addEventListener('change', () => {
    state.sessions.forEach((entry) => entry.session.release?.());
    state.sessions.clear();
    toast('Đã xóa model cache để đổi execution provider.');
  });
  elements.imageInput.addEventListener('change', () => handleFiles(elements.imageInput.files));
  elements.uploadIdle.addEventListener('click', () => elements.imageInput.click());
  elements.addImages.addEventListener('click', () => elements.imageInput.click());
  elements.clearImage.addEventListener('click', (event) => { event.preventDefault(); clearImage(); });
  ['dragenter', 'dragover'].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove('dragging');
  }));
  elements.dropZone.addEventListener('drop', (event) => handleFiles(event.dataTransfer.files));
  elements.runButton.addEventListener('click', runYoloTab);
  elements.exportJson.addEventListener('click', exportJson);
  elements.exportCsv.addEventListener('click', exportCsv);
  window.addEventListener('resize', () => {
    if (!state.bitmap) return;
    drawPreview(state.detections);
  });
}

function switchTab(tabName, focusTab = false) {
  const selectedButton = elements.tabButtons.find((button) => button.dataset.tab === tabName);
  if (!selectedButton) return;
  state.activeTab = tabName;
  elements.tabButtons.forEach((button) => {
    const selected = button === selectedButton;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  elements.tabPanels.forEach((panel) => {
    panel.hidden = panel.id !== selectedButton.getAttribute('aria-controls');
  });
  if (focusTab) selectedButton.focus({ preventScroll: true });
}

function updateRunState() {
  const realReady = Boolean(state.manifest?.detector?.available);
  elements.runButton.disabled = !state.items.length || state.busy || (!realReady && !elements.demoMode.checked);
  elements.runButton.setAttribute('aria-busy', String(state.busy));
  elements.inferencePanel?.setAttribute('aria-busy', String(state.busy));
  elements.runButtonLabel.textContent = state.busy ? 'Đang phân tích…' : 'Chạy toàn bộ pipeline';
  elements.imageInput.disabled = state.busy;
  elements.uploadIdle.disabled = state.busy;
  elements.addImages.disabled = state.busy;
  elements.clearImage.disabled = state.busy;
}

function updateExportState() {
  const hasValidReport = state.reports.some((report) => (
    report?.detector?.detections && Array.isArray(report.classifications)
  ));
  elements.exportJson.disabled = !hasValidReport;
  elements.exportCsv.disabled = !hasValidReport;
}

async function handleFiles(fileList) {
  if (!fileList?.length || state.busy) return;
  state.uploadErrors = [];
  const availableSlots = Math.max(0, 20 - state.items.length);
  if (!availableSlots) {
    state.uploadErrors.push({ name: 'Lô ảnh', message: 'Mỗi lô hỗ trợ tối đa 20 ảnh.' });
    renderBatchFeedback();
    toast('Mỗi lô hỗ trợ tối đa 20 ảnh.', true);
    return;
  }
  const selected = Array.from(fileList).slice(0, availableSlots);
  if (selected.length < fileList.length) {
    state.uploadErrors.push({ name: 'Lô ảnh', message: `${fileList.length - selected.length} ảnh không được thêm vì giới hạn tối đa là 20 ảnh.` });
  }
  const valid = selected.filter((file) => {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      state.uploadErrors.push({ name: file.name, message: 'Chỉ hỗ trợ JPG, PNG hoặc WebP.' });
      toast(`${file.name}: chỉ hỗ trợ JPG, PNG hoặc WebP.`, true);
      return false;
    }
    if (file.size > 20 * 1024 * 1024) {
      state.uploadErrors.push({ name: file.name, message: 'Vượt quá giới hạn 20 MB mỗi ảnh.' });
      toast(`${file.name}: vượt quá giới hạn 20 MB.`, true);
      return false;
    }
    return true;
  });
  if (!valid.length) {
    renderBatchFeedback();
    return;
  }

  const firstNewIndex = state.items.length;
  for (const file of valid) {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      state.items.push({
        id: `${Date.now()}-${state.items.length}-${file.name}`,
        file,
        bitmap,
        thumbnail: createThumbnail(bitmap),
        status: 'pending',
        report: null,
        error: null,
      });
    } catch (error) {
      state.uploadErrors.push({ name: file.name, message: `Không mở được ảnh: ${error.message}` });
      toast(`Không mở được ${file.name}: ${error.message}`, true);
    }
  }
  if (!state.items.length) {
    renderBatchFeedback();
    return;
  }

  state.items.forEach((item) => { item.status = 'pending'; item.report = null; item.error = null; });
  state.reports = [];
  elements.imageInput.value = '';
  elements.dropZone.classList.add('has-image');
  selectImage(Math.min(firstNewIndex, state.items.length - 1));
  resetResults();
  renderBatchStrip();
  updateRunState();
  if (selected.length < fileList.length || valid.length < selected.length) {
    toast(`Đã thêm ${valid.length} ảnh hợp lệ. Tối đa 20 ảnh mỗi lô.`);
  }
}

function createThumbnail(bitmap) {
  const canvas = createCanvas(96, 96);
  const ctx = canvas.getContext('2d');
  const scale = Math.max(96 / bitmap.width, 96 / bitmap.height);
  const width = bitmap.width * scale;
  const height = bitmap.height * scale;
  ctx.drawImage(bitmap, (96 - width) / 2, (96 - height) / 2, width, height);
  return canvas.toDataURL('image/jpeg', .78);
}

function batchStatus(item) {
  const states = {
    pending: { label: 'Chờ xử lý', icon: '○' },
    running: { label: 'Đang xử lý', icon: '◌' },
    done: { label: 'Hoàn tất', icon: '✓' },
    error: { label: 'Thất bại', icon: '!' },
  };
  return states[item.status] || states.pending;
}

function renderBatchFeedback() {
  const total = state.items.length;
  const successful = state.items.filter((item) => item.status === 'done').length;
  const failed = state.items.filter((item) => item.status === 'error').length;
  const processed = successful + failed;
  const errors = [
    ...state.uploadErrors,
    ...state.items
      .filter((item) => item.status === 'error' && item.error)
      .map((item) => ({ name: item.file.name, message: item.error })),
  ];

  elements.batchSummary.hidden = total === 0;
  elements.batchTotal.textContent = total;
  elements.batchProcessed.textContent = processed;
  elements.batchFailed.textContent = failed;
  elements.batchSummaryBar.style.width = `${total ? (processed / total) * 100 : 0}%`;
  elements.batchSummaryProgress.setAttribute('aria-valuenow', Math.round(total ? (processed / total) * 100 : 0));
  elements.batchSummaryProgress.setAttribute('aria-valuetext', `${processed} trên ${total} ảnh đã xử lý, ${failed} ảnh thất bại`);

  elements.batchErrors.hidden = errors.length === 0;
  elements.batchErrorCount.textContent = errors.length;
  elements.batchErrorList.innerHTML = errors.map((error) => `
    <li><b>${escapeHtml(error.name)}</b><span>${escapeHtml(error.message)}</span></li>
  `).join('');
  elements.queueEmpty.hidden = total > 0;
  elements.detectionTabCount.textContent = total;
  updateExportState();
}

function renderBatchStrip() {
  const hidden = state.items.length === 0;
  const listMarkup = state.items.map((item, index) => {
    const status = batchStatus(item);
    return `<div class="batch-list-row${index === state.activeImageIndex ? ' active' : ''}" data-status="${item.status}" role="listitem">
      <button class="batch-select" data-index="${index}" type="button" aria-label="Xem ${escapeHtml(item.file.name)}, ${status.label}"${state.busy ? ' disabled' : ''}>
        <img src="${item.thumbnail}" alt="" aria-hidden="true" />
        <span class="batch-item-copy">
          <b title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</b>
          <small class="batch-file-meta">${formatBytes(item.file.size)} · ${index + 1}/${state.items.length}</small>
          <span class="batch-status status-${item.status}"><i aria-hidden="true">${status.icon}</i>${status.label}${item.status === 'done' ? ` · ${item.report?.detector?.detections?.length || 0} ROI` : ''}</span>
          ${item.error ? `<small class="batch-item-error">${escapeHtml(item.error)}</small>` : ''}
        </span>
      </button>
      <button class="batch-remove" data-index="${index}" type="button" aria-label="Xóa ${escapeHtml(item.file.name)}"${state.busy ? ' disabled' : ''}>×</button>
    </div>`;
  }).join('');
  const resultMarkup = state.items.map((item, index) => {
    const status = batchStatus(item);
    return `<button class="batch-item${index === state.activeImageIndex ? ' active' : ''}" data-index="${index}" data-status="${item.status}" type="button"${state.busy ? ' disabled' : ''}>
      <img src="${item.thumbnail}" alt="" aria-hidden="true" />
      <span class="batch-item-copy"><b>${escapeHtml(item.file.name)}</b><small>${index + 1}/${state.items.length}</small><span class="batch-status status-${item.status}"><i aria-hidden="true">${status.icon}</i>${status.label}</span></span>
    </button>`;
  }).join('');
  elements.batchList.hidden = hidden;
  elements.batchCount.textContent = state.items.length;
  elements.batchStrip.hidden = hidden;
  elements.batchStrip.innerHTML = listMarkup;
  elements.batchStrip.querySelectorAll('.batch-select').forEach((button) => {
    button.addEventListener('click', () => selectImage(Number(button.dataset.index), true));
  });
  elements.batchStrip.querySelectorAll('.batch-remove').forEach((button) => {
    button.addEventListener('click', () => removeImage(Number(button.dataset.index)));
  });
  elements.resultBatchStrip.hidden = state.items.length <= 1;
  elements.resultBatchStrip.innerHTML = resultMarkup;
  elements.resultBatchStrip.querySelectorAll('.batch-item').forEach((button) => {
    button.addEventListener('click', () => selectImage(Number(button.dataset.index), true));
  });
  renderBatchFeedback();
}

function removeImage(index) {
  if (state.busy || !state.items[index]) return;
  const activeId = state.items[state.activeImageIndex]?.id;
  state.items[index].bitmap.close();
  state.items.splice(index, 1);
  state.reports = state.items.filter((item) => item.report).map((item) => item.report);
  if (!state.items.length) {
    clearImage();
    return;
  }
  const retainedActiveIndex = state.items.findIndex((item) => item.id === activeId);
  selectImage(retainedActiveIndex >= 0 ? retainedActiveIndex : Math.min(index, state.items.length - 1), true);
  updateRunState();
}

function selectImage(index, showResult = false) {
  const item = state.items[index];
  if (!item) return;
  state.activeImageIndex = index;
  state.bitmap = item.bitmap;
  state.file = item.file;
  state.report = item.report;
  state.detectorResult = item.report?.detector || null;
  state.detections = item.report?.detector?.detections || [];
  state.selectedDetectionIndex = 0;
  elements.imageMeta.textContent = `${state.items.length} ảnh · đang xem ${index + 1}/${state.items.length}: ${item.file.name} · ${item.bitmap.width}×${item.bitmap.height}`;
  renderBatchStrip();
  drawPreview(state.detections);
  if (!showResult) return;
  if (!item.report) {
    resetResults();
    return;
  }
  renderYoloReport(false);
}

function clearImage() {
  if (state.busy) return;
  state.items.forEach((item) => item.bitmap.close());
  state.items = [];
  state.reports = [];
  state.activeImageIndex = -1;
  state.bitmap = null;
  state.file = null;
  state.detections = [];
  state.detectorResult = null;
  state.report = null;
  state.selectedDetectionIndex = 0;
  state.uploadErrors = [];
  elements.imageInput.value = '';
  elements.dropZone.classList.remove('has-image');
  elements.batchStrip.hidden = true;
  elements.batchStrip.innerHTML = '';
  elements.batchList.hidden = true;
  elements.batchCount.textContent = '0';
  elements.resultBatchStrip.hidden = true;
  elements.resultBatchStrip.innerHTML = '';
  elements.imageMeta.textContent = 'Chưa chọn ảnh';
  const context = elements.canvas.getContext('2d');
  context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  resetResults();
  renderBatchFeedback();
  updateRunState();
  elements.uploadIdle.focus();
}

function resetResults() {
  elements.emptyResults.hidden = false;
  elements.resultContent.hidden = true;
  elements.detectionOutput.hidden = true;
  elements.detectionEmpty.hidden = false;
  elements.benchmarkOutput.hidden = true;
  elements.benchmarkEmpty.hidden = false;
  elements.benchmarkTable.hidden = true;
  elements.resultsTabCount.textContent = '0';
  elements.benchmarkTabCount.textContent = '0';
  updateExportState();
}

function drawPreview(detections) {
  const bitmap = state.bitmap;
  if (!bitmap) return;
  const canvas = elements.canvas;
  const displayWidth = Math.max(1, elements.dropZone.clientWidth);
  const displayHeight = Math.max(1, elements.dropZone.clientHeight);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(displayWidth * pixelRatio);
  canvas.height = Math.round(displayHeight * pixelRatio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  ctx.clearRect(0, 0, displayWidth, displayHeight);
  const previewScale = Math.min(displayWidth / bitmap.width, displayHeight / bitmap.height);
  const imageWidth = bitmap.width * previewScale;
  const imageHeight = bitmap.height * previewScale;
  const offsetX = (displayWidth - imageWidth) / 2;
  const offsetY = (displayHeight - imageHeight) / 2;
  ctx.drawImage(bitmap, offsetX, offsetY, imageWidth, imageHeight);
  ctx.lineJoin = 'round';
  detections.forEach((detection, index) => {
    const color = COLORS[detection.classId % COLORS.length];
    const x = offsetX + detection.box[0] * previewScale;
    const y = offsetY + detection.box[1] * previewScale;
    const width = (detection.box[2] - detection.box[0]) * previewScale;
    const height = (detection.box[3] - detection.box[1]) * previewScale;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, displayWidth / 500);
    ctx.strokeRect(x, y, width, height);
    const label = `${index + 1}  ${detection.className}  ${(detection.confidence * 100).toFixed(0)}%`;
    ctx.font = `600 ${Math.max(12, displayWidth / 70)}px Manrope, sans-serif`;
    const textWidth = ctx.measureText(label).width;
    const labelHeight = Math.max(25, displayWidth / 42);
    const labelY = Math.max(0, y - labelHeight);
    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, textWidth + 18, labelHeight);
    ctx.fillStyle = '#17241d';
    ctx.fillText(label, x + 9, labelY + labelHeight * .7);
  });
}

function drawResultCanvas(detections) {
  const bitmap = state.bitmap;
  if (!bitmap || !elements.resultCanvas) return;
  const wrap = elements.resultCanvas.parentElement;
  const displayWidth = Math.max(1, wrap.clientWidth);
  const displayHeight = Math.max(1, wrap.clientHeight);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const canvas = elements.resultCanvas;
  canvas.width = Math.round(displayWidth * pixelRatio);
  canvas.height = Math.round(displayHeight * pixelRatio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  ctx.clearRect(0, 0, displayWidth, displayHeight);
  const scale = Math.min(displayWidth / bitmap.width, displayHeight / bitmap.height);
  const imageWidth = bitmap.width * scale;
  const imageHeight = bitmap.height * scale;
  const offsetX = (displayWidth - imageWidth) / 2;
  const offsetY = (displayHeight - imageHeight) / 2;
  ctx.drawImage(bitmap, offsetX, offsetY, imageWidth, imageHeight);
  ctx.lineJoin = 'round';
  detections.forEach((detection, index) => {
    const color = COLORS[detection.classId % COLORS.length];
    const x = offsetX + detection.box[0] * scale;
    const y = offsetY + detection.box[1] * scale;
    const width = (detection.box[2] - detection.box[0]) * scale;
    const height = (detection.box[3] - detection.box[1]) * scale;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, displayWidth / 420);
    ctx.strokeRect(x, y, width, height);
    const label = `${index + 1}. ${detection.className} ${(detection.confidence * 100).toFixed(1)}%`;
    ctx.font = `600 ${Math.max(11, displayWidth / 58)}px Manrope, sans-serif`;
    const labelWidth = ctx.measureText(label).width + 16;
    const labelHeight = Math.max(24, displayWidth / 40);
    const labelY = Math.max(0, y - labelHeight);
    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, labelWidth, labelHeight);
    ctx.fillStyle = '#17241d';
    ctx.fillText(label, x + 8, labelY + labelHeight * .7);
  });
  elements.boxCountLabel.textContent = `${detections.length} BOX`;
}

function createCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function imageDataToChw(imageData, mean = null, std = null) {
  const pixels = imageData.data;
  const count = imageData.width * imageData.height;
  const output = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const value = pixels[i * 4 + channel] / 255;
      output[channel * count + i] = mean ? (value - mean[channel]) / std[channel] : value;
    }
  }
  return output;
}

function preprocessDetector(bitmap, size) {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  const scale = Math.min(size / bitmap.width, size / bitmap.height);
  const drawWidth = Math.round(bitmap.width * scale);
  const drawHeight = Math.round(bitmap.height * scale);
  const padX = Math.floor((size - drawWidth) / 2);
  const padY = Math.floor((size - drawHeight) / 2);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, padX, padY, drawWidth, drawHeight);
  const data = imageDataToChw(ctx.getImageData(0, 0, size, size));
  return { tensor: new ort.Tensor('float32', data, [1, 3, size, size]), scale, padX, padY };
}

function reflect101(position, length) {
  if (length <= 1) return 0;
  let value = position;
  while (value < 0 || value >= length) {
    value = value < 0 ? -value : (2 * length - value - 2);
  }
  return value;
}

function enhancePixels(imageData, contrast, sharpen) {
  const { width, height, data } = imageData;
  const enhanced = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    for (let c = 0; c < 3; c += 1) enhanced[i + c] = Math.max(0, Math.min(255, (data[i + c] - 127.5) * contrast + 127.5));
    enhanced[i + 3] = 255;
  }
  const result = new Uint8ClampedArray(enhanced.length);
  const kernel = [[1, 1, 1], [1, 5, 1], [1, 1, 1]];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      for (let c = 0; c < 3; c += 1) {
        let smooth = 0;
        for (let ky = -1; ky <= 1; ky += 1) {
          for (let kx = -1; kx <= 1; kx += 1) {
            const sx = reflect101(x + kx, width);
            const sy = reflect101(y + ky, height);
            smooth += enhanced[(sy * width + sx) * 4 + c] * kernel[ky + 1][kx + 1];
          }
        }
        smooth /= 13;
        result[offset + c] = Math.max(0, Math.min(255, smooth * (1 - sharpen) + enhanced[offset + c] * sharpen));
      }
      result[offset + 3] = 255;
    }
  }
  return new ImageData(result, width, height);
}

function expandedBox(box, imageWidth, imageHeight, padding = .03) {
  const width = box[2] - box[0];
  const height = box[3] - box[1];
  return [
    Math.max(0, box[0] - width * padding),
    Math.max(0, box[1] - height * padding),
    Math.min(imageWidth, box[2] + width * padding),
    Math.min(imageHeight, box[3] + height * padding),
  ];
}

function preprocessClassifier(bitmap, box, metadata) {
  const size = metadata.input_shape?.[2] || 224;
  const [x1, y1, x2, y2] = expandedBox(box, bitmap.width, bitmap.height);
  const cropWidth = Math.max(1, x2 - x1);
  const cropHeight = Math.max(1, y2 - y1);
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const padding = metadata.padding_rgb || [114, 114, 114];
  ctx.fillStyle = `rgb(${padding.join(',')})`;
  ctx.fillRect(0, 0, size, size);
  const scale = Math.min(size / cropWidth, size / cropHeight);
  const width = Math.max(1, Math.round(cropWidth * scale));
  const height = Math.max(1, Math.round(cropHeight * scale));
  const left = Math.floor((size - width) / 2);
  const top = Math.floor((size - height) / 2);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, x1, y1, cropWidth, cropHeight, left, top, width, height);
  const enhanced = enhancePixels(
    ctx.getImageData(0, 0, size, size),
    metadata.contrast_enhancement ?? 1.2,
    metadata.sharpen_factor ?? 1.3,
  );
  const values = imageDataToChw(
    enhanced,
    metadata.normalization_mean || [0.485, 0.456, 0.406],
    metadata.normalization_std || [0.229, 0.224, 0.225],
  );
  return new ort.Tensor('float32', values, [1, 3, size, size]);
}

function createRoiPreview(bitmap, box, size = 240) {
  const [x1, y1, x2, y2] = expandedBox(box, bitmap.width, bitmap.height);
  const cropWidth = Math.max(1, x2 - x1);
  const cropHeight = Math.max(1, y2 - y1);
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  const scale = Math.min(size / cropWidth, size / cropHeight);
  const width = Math.max(1, Math.round(cropWidth * scale));
  const height = Math.max(1, Math.round(cropHeight * scale));
  const left = Math.floor((size - width) / 2);
  const top = Math.floor((size - height) / 2);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, x1, y1, cropWidth, cropHeight, left, top, width, height);
  return canvas.toDataURL('image/jpeg', 0.86);
}

async function getSession(key, path) {
  if (state.sessions.has(key)) return { ...state.sessions.get(key), cached: true };
  const requestedProvider = elements.provider.value;
  const providers = requestedProvider === 'webgpu' && navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
  const start = performance.now();
  const sizePromise = fetch(path, { method: 'HEAD' })
    .then((response) => Number(response.headers.get('content-length')) || null)
    .catch(() => null);
  let session;
  try {
    session = await ort.InferenceSession.create(path, {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    });
  } catch (error) {
    if (requestedProvider !== 'webgpu') throw error;
    toast('WebGPU không khởi tạo được, đã chuyển sang WASM.');
    elements.provider.value = 'wasm';
    session = await ort.InferenceSession.create(path, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  }
  const entry = { session, loadMs: performance.now() - start, modelBytes: await sizePromise, cached: false };
  state.sessions.set(key, entry);
  return entry;
}

function outputAt(data, dims, attribute, index) {
  const attributesFirst = dims.length === 3 && dims[1] < dims[2];
  const boxCount = attributesFirst ? dims[2] : dims[1];
  return attributesFirst ? data[attribute * boxCount + index] : data[index * dims[2] + attribute];
}

function decodeYolo(output, transform, threshold) {
  const { data, dims } = output;
  const attributesFirst = dims.length === 3 && dims[1] < dims[2];
  const boxCount = attributesFirst ? dims[2] : dims[1];
  const attributeCount = attributesFirst ? dims[1] : dims[2];
  const classCount = attributeCount - 4;
  const candidates = [];
  for (let index = 0; index < boxCount; index += 1) {
    let classId = 0;
    let confidence = -Infinity;
    for (let c = 0; c < classCount; c += 1) {
      const score = outputAt(data, dims, c + 4, index);
      if (score > confidence) { confidence = score; classId = c; }
    }
    if (confidence < threshold || classId >= PRODUCE.length) continue;
    const cx = outputAt(data, dims, 0, index);
    const cy = outputAt(data, dims, 1, index);
    const width = outputAt(data, dims, 2, index);
    const height = outputAt(data, dims, 3, index);
    const x1 = Math.max(0, (cx - width / 2 - transform.padX) / transform.scale);
    const y1 = Math.max(0, (cy - height / 2 - transform.padY) / transform.scale);
    const x2 = Math.min(state.bitmap.width, (cx + width / 2 - transform.padX) / transform.scale);
    const y2 = Math.min(state.bitmap.height, (cy + height / 2 - transform.padY) / transform.scale);
    if (x2 - x1 < 2 || y2 - y1 < 2) continue;
    candidates.push({ classId, className: PRODUCE[classId], confidence, box: [x1, y1, x2, y2] });
  }
  return candidates;
}

function boxIou(a, b) {
  const left = Math.max(a[0], b[0]);
  const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / Math.max(1e-9, areaA + areaB - intersection);
}

function boxContainment(a, b) {
  const left = Math.max(a[0], b[0]);
  const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / Math.max(1e-9, Math.min(areaA, areaB));
}

function nonMaximumSuppression(candidates, iouThreshold, maxDetections = 100) {
  const sorted = [...candidates].sort((a, b) => b.confidence - a.confidence);
  const selected = [];
  while (sorted.length && selected.length < maxDetections) {
    const best = sorted.shift();
    selected.push(best);
    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      const sameClass = sorted[i].classId === best.classId;
      const overlapsTooMuch = boxIou(sorted[i].box, best.box) > iouThreshold;
      const nestedDuplicate = boxContainment(sorted[i].box, best.box) > 0.85;
      if (sameClass && (overlapsTooMuch || nestedDuplicate)) sorted.splice(i, 1);
    }
  }
  return selected;
}

function softmax(values) {
  const max = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - max));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

async function fetchMetadata(path) {
  try {
    const response = await fetch(path, { cache: 'no-store' });
    if (response.ok) return response.json();
  } catch (error) {
    console.warn('Metadata fallback:', error);
  }
  return {
    classes: ['fresh', 'rotten'], input_shape: [1, 3, 224, 224],
    normalization_mean: [0.485, 0.456, 0.406], normalization_std: [0.229, 0.224, 0.225],
    contrast_enhancement: 1.2, sharpen_factor: 1.3, padding_rgb: [114, 114, 114],
  };
}

async function runDetector(repeats) {
  const detector = state.manifest.detector;
  const prepared = preprocessDetector(state.bitmap, detector.input_size || 640);
  const loaded = await getSession('detector', detector.path);
  const inputName = loaded.session.inputNames[0];
  let outputs;
  const times = [];
  await loaded.session.run({ [inputName]: prepared.tensor });
  for (let index = 0; index < repeats; index += 1) {
    const start = performance.now();
    outputs = await loaded.session.run({ [inputName]: prepared.tensor });
    times.push(performance.now() - start);
  }
  const output = outputs[loaded.session.outputNames[0]];
  const candidates = decodeYolo(output, prepared, Number(elements.confidence.value));
  return {
    detections: nonMaximumSuppression(candidates, Number(elements.iou.value)),
    loadMs: loaded.cached ? 0 : loaded.loadMs,
    modelBytes: loaded.modelBytes,
    latencyMs: average(times),
  };
}

function classifierPaths(architecture, produce) {
  const base = `models/classifiers/${architecture}/${produce}`;
  return { model: `${base}/${produce}_${architecture}.onnx`, metadata: `${base}/${produce}_${architecture}.json` };
}

async function runClassifier(architecture, detection, repeats) {
  const paths = classifierPaths(architecture, detection.className);
  const metadata = await fetchMetadata(paths.metadata);
  const tensor = preprocessClassifier(state.bitmap, detection.box, metadata);
  const key = `classifier:${architecture}:${detection.className}`;
  const loaded = await getSession(key, paths.model);
  const inputName = loaded.session.inputNames[0];
  let outputs;
  const times = [];
  await loaded.session.run({ [inputName]: tensor });
  for (let index = 0; index < repeats; index += 1) {
    const start = performance.now();
    outputs = await loaded.session.run({ [inputName]: tensor });
    times.push(performance.now() - start);
  }
  const logits = Array.from(outputs[loaded.session.outputNames[0]].data);
  const probabilities = softmax(logits);
  const classIndex = probabilities[1] > probabilities[0] ? 1 : 0;
  return {
    architecture,
    architectureName: ARCHITECTURES[architecture],
    produce: detection.className,
    freshness: metadata.classes?.[classIndex] || ['fresh', 'rotten'][classIndex],
    confidence: probabilities[classIndex],
    probabilities: { fresh: probabilities[0], rotten: probabilities[1] },
    loadMs: loaded.cached ? 0 : loaded.loadMs,
    modelBytes: loaded.modelBytes,
    latencyMs: average(times),
    validationMacroF1: metadata.validation_macro_f1 ?? null,
  };
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function demoDetection() {
  const { width, height } = state.bitmap;
  const canvas = createCanvas(32, 32);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(state.bitmap, 0, 0, 32, 32);
  const pixels = ctx.getImageData(0, 0, 32, 32).data;
  let r = 0; let g = 0; let b = 0;
  for (let i = 0; i < pixels.length; i += 4) { r += pixels[i]; g += pixels[i + 1]; b += pixels[i + 2]; }
  const classId = Math.abs(Math.round((r * 3 + g * 5 + b * 7) / 100000)) % PRODUCE.length;
  return [{
    classId, className: PRODUCE[classId], confidence: .82,
    box: [width * .18, height * .13, width * .82, height * .88],
  }];
}

function demoClassification(architecture, detection, index) {
  const seed = (detection.classId + 1) * 13 + Object.keys(ARCHITECTURES).indexOf(architecture) * 7 + index;
  const confidence = .67 + (seed % 27) / 100;
  const freshness = seed % 4 === 0 ? 'rotten' : 'fresh';
  return {
    architecture, architectureName: ARCHITECTURES[architecture], produce: detection.className,
    freshness, confidence, probabilities: freshness === 'fresh' ? { fresh: confidence, rotten: 1 - confidence } : { fresh: 1 - confidence, rotten: confidence },
    loadMs: 180 + seed * 3.4, modelBytes: 20_000_000 + seed * 120_000,
    latencyMs: 9 + (seed % 19) * 1.7, validationMacroF1: null,
  };
}

function selectedArchitectures() {
  return elements.architecture.value === 'all' ? Object.keys(ARCHITECTURES) : [elements.architecture.value];
}

function classifierAvailable(architecture, produce) {
  return state.manifest.classifiers?.[architecture]?.available?.includes(produce);
}

async function runYoloTab() {
  if (!state.items.length || state.busy) return;
  state.busy = true;
  state.uploadErrors = [];
  state.reports = [];
  updateRunState();
  resetResults();
  const demo = elements.demoMode.checked;
  const repeats = Number(elements.repeats.value);
  const architectures = selectedArchitectures();
  state.items.forEach((item) => { item.status = 'pending'; item.report = null; item.error = null; });
  renderBatchStrip();
  const failures = [];
  try {
    for (let imageIndex = 0; imageIndex < state.items.length; imageIndex += 1) {
      const item = state.items[imageIndex];
      selectImage(imageIndex);
      item.status = 'running';
      renderBatchStrip();
      try {
        const report = await runActivePipeline({ demo, repeats, architectures, imageIndex, imageCount: state.items.length });
        item.report = report;
        item.status = 'done';
        state.reports.push(report);
      } catch (error) {
        console.error(error);
        item.status = 'error';
        item.error = error.message;
        failures.push(`${item.file.name}: ${error.message}`);
      }
      renderBatchStrip();
    }

    const firstCompleted = state.items.findIndex((item) => item.status === 'done');
    if (firstCompleted < 0) throw new Error(failures[0] || 'Không ảnh nào được xử lý thành công.');
    selectImage(firstCompleted);
    setProgress(100, `Hoàn tất ${state.reports.length}/${state.items.length} ảnh · ${state.reports.reduce((sum, report) => sum + report.detector.detections.length, 0)} ROI`);
    renderYoloReport();
    if (failures.length) toast(`${failures.length} ảnh có lỗi; ${state.reports.length} ảnh đã hoàn tất.`, true);
    setTimeout(() => { elements.progressWrap.hidden = true; }, 1200);
  } catch (error) {
    console.error(error);
    resetResults();
    elements.progressWrap.hidden = true;
    toast(error.message, true);
  } finally {
    state.busy = false;
    renderBatchStrip();
    updateRunState();
  }
}

async function runActivePipeline({ demo, repeats, architectures, imageIndex, imageCount }) {
  const started = performance.now();
  const reportProgress = (localPercent, message) => {
    const overall = ((imageIndex + localPercent / 100) / imageCount) * 100;
    setProgress(overall, `Ảnh ${imageIndex + 1}/${imageCount} · ${message}`);
  };
  reportProgress(8, demo ? 'Đang tạo detection mô phỏng…' : 'Đang chạy YOLO11l…');
  const detectorResult = demo
    ? { detections: demoDetection(), loadMs: 420, modelBytes: 51_000_000, latencyMs: 76 }
    : await runDetector(1);
  state.detectorResult = detectorResult;
  state.detections = detectorResult.detections;
  drawPreview(state.detections);
  if (!state.detections.length) throw new Error('Không phát hiện đối tượng nào. Hãy thử giảm confidence hoặc chọn ảnh khác.');
  const detectionWallMs = performance.now() - started;

  const missing = [];
  if (!demo) {
    architectures.forEach((architecture) => state.detections.forEach((detection) => {
      if (!classifierAvailable(architecture, detection.className)) missing.push(`${ARCHITECTURES[architecture]} / ${detection.className}`);
    }));
    if (missing.length) throw new Error(`Thiếu classifier: ${[...new Set(missing)].join(', ')}`);
  }

  const classifierStarted = performance.now();
  const rows = [];
  const total = state.detections.length * architectures.length;
  let completed = 0;
  for (let detectionIndex = 0; detectionIndex < state.detections.length; detectionIndex += 1) {
    const detection = state.detections[detectionIndex];
    for (const architecture of architectures) {
      completed += 1;
      reportProgress(15 + (completed / total) * 80, `Crop ${detectionIndex + 1}/${state.detections.length} · ${ARCHITECTURES[architecture]}`);
      const classification = demo
        ? demoClassification(architecture, detection, detectionIndex)
        : await runClassifier(architecture, detection, repeats);
      rows.push({ objectIndex: detectionIndex + 1, detectionConfidence: detection.confidence, ...classification });
    }
  }
  const benchmarkWallMs = performance.now() - classifierStarted;
  return {
    schemaVersion: 2,
    createdAt: new Date().toISOString(),
    demo,
    batch: { imageIndex: imageIndex + 1, imageCount },
    image: { name: state.file.name, width: state.bitmap.width, height: state.bitmap.height, bytes: state.file.size },
    settings: {
      requestedProvider: elements.provider.value,
      architectures,
      repeats,
      confidenceThreshold: Number(elements.confidence.value),
      iouThreshold: Number(elements.iou.value),
    },
    detector: { architecture: 'YOLO11l', ...detectorResult },
    classifications: rows,
    detectionWallMs,
    benchmarkWallMs,
    totalWallMs: performance.now() - started,
  };
}

function renderYoloReport() {
  const report = state.report;
  if (!report) return;
  elements.emptyResults.hidden = true;
  elements.resultContent.hidden = false;
  elements.detectionOutput.hidden = false;
  elements.detectionEmpty.hidden = true;
  elements.benchmarkOutput.hidden = false;
  elements.benchmarkEmpty.hidden = true;
  elements.boxCountLabel.textContent = `${report.detector.detections.length} ROI`;
  elements.resultsTabCount.textContent = report.detector.detections.length;
  elements.benchmarkTabCount.textContent = report.classifications.length;
  updateExportState();
  renderTechnicalSummaries(report, state.selectedDetectionIndex);
  renderCropGallery(report);
  renderAiItems(report);
  renderSelectedDetection(report, state.selectedDetectionIndex);
  renderAiModelList(report, state.selectedDetectionIndex);
  renderBenchmarkRoiList(report, state.selectedDetectionIndex);
  renderBenchmarkTable(report);
}

function summaryCards(rows) {
  return rows.map(([label, value, detail]) => `
    <article class="summary-card"><span>${label}</span><b>${value}</b><small>${detail}</small></article>
  `).join('');
}

function renderTechnicalSummaries(report, selectedIndex) {
  const detection = report.detector.detections[selectedIndex];
  const box = detection?.box || [0, 0, 0, 0];
  const classifierLatency = report.classifications.length
    ? average(report.classifications.map((row) => row.latencyMs))
    : 0;
  elements.detectionSummaryGrid.innerHTML = summaryCards([
    ['Đối tượng', report.detector.detections.length, 'sau NMS'],
    ['YOLO latency', formatMs(report.detector.latencyMs), 'detector 640×640'],
    ['ROI đang chọn', detection ? `${selectedIndex + 1}` : '—', detection ? `${(detection.confidence * 100).toFixed(1)}% confidence` : 'không có detection'],
    ['Kích thước crop', detection ? `${Math.round(box[2] - box[0])}×${Math.round(box[3] - box[1])}` : '—', 'pixel trên ảnh gốc'],
  ]);
  elements.classificationSummaryGrid.innerHTML = summaryCards([
    ['Classifier', report.settings.architectures.length, 'kiến trúc đã chạy'],
    ['Prediction', report.classifications.length, 'tất cả ROI'],
    ['Classification wall', formatMs(report.benchmarkWallMs), 'toàn bộ classifier'],
    ['Latency trung bình', formatMs(classifierLatency), 'mỗi inference'],
  ]);
  elements.runtimeSummaryGrid.innerHTML = summaryCards([
    ['Execution provider', escapeHtml(report.settings.requestedProvider), report.demo ? 'dữ liệu mô phỏng' : 'real inference'],
    ['Số lần đo', report.settings.repeats, 'classifier repeats'],
    ['Confidence threshold', report.settings.confidenceThreshold.toFixed(2), 'YOLO filter'],
    ['IoU threshold', report.settings.iouThreshold.toFixed(2), 'NMS'],
    ['Tổng thời gian', formatMs(report.totalWallMs), escapeHtml(report.image.name)],
    ['Chế độ', report.demo ? 'Demo' : 'ONNX', `${state.reports.length}/${state.items.length} ảnh hoàn tất`],
  ]);
}

function freshnessLabel(value) {
  return value === 'fresh' ? 'Tươi' : value === 'rotten' ? 'Hỏng' : 'Chưa có kết quả';
}

// Presentation-only vote summary. It never invents an aggregate confidence.
function deriveDisplayVerdict(report, objectIndex) {
  const predictions = report.classifications.filter((row) => row.objectIndex === objectIndex + 1);
  if (!predictions.length) {
    return { key: 'unavailable', label: 'Chưa có kết quả', confidence: null, detail: 'Không có classifier đã chạy' };
  }

  if (report.settings.architectures.length === 1 || predictions.length === 1) {
    const prediction = predictions[0];
    return {
      key: prediction.freshness,
      label: freshnessLabel(prediction.freshness),
      confidence: prediction.confidence,
      detail: `${prediction.architectureName} · chế độ một model`,
    };
  }

  const counts = predictions.reduce((result, row) => {
    result[row.freshness] = (result[row.freshness] || 0) + 1;
    return result;
  }, {});
  const fresh = counts.fresh || 0;
  const rotten = counts.rotten || 0;
  const disagreement = fresh > 0 && rotten > 0;
  const majority = fresh >= rotten ? 'fresh' : 'rotten';

  return {
    key: disagreement ? 'unstable' : majority,
    label: disagreement ? 'Các model bất đồng' : `${counts[majority]}/${predictions.length} model: ${freshnessLabel(majority)}`,
    confidence: null,
    detail: `Fresh ${fresh} · Rotten ${rotten}`,
  };
}

function renderCropGallery(report) {
  elements.cropGallery.innerHTML = report.detector.detections.map((detection, index) => {
    return `
      <button class="roi-card${index === state.selectedDetectionIndex ? ' active' : ''}" type="button" data-detection-index="${index}" aria-pressed="${index === state.selectedDetectionIndex}">
        <div class="roi-card-image">
          <span>${String(index + 1).padStart(2, '0')}</span>
          <img src="${createRoiPreview(state.bitmap, detection.box)}" alt="ROI ${index + 1}: ${escapeHtml(detection.className)}" />
        </div>
        <span><b>${escapeHtml(detection.className)}</b><small>${(detection.confidence * 100).toFixed(1)}% YOLO</small></span>
      </button>`;
  }).join('');
  elements.cropGallery.querySelectorAll('.roi-card').forEach((button) => {
    button.addEventListener('click', () => selectDetection(Number(button.dataset.detectionIndex), report));
  });
}

function renderAiItems(report) {
  elements.aiItemsGrid.innerHTML = report.detector.detections.map((detection, index) => {
    const verdict = deriveDisplayVerdict(report, index);
    const confidence = verdict.confidence == null ? '' : `<small>Confidence ${(verdict.confidence * 100).toFixed(1)}%</small>`;
    return `<button class="ai-item-card${index === state.selectedDetectionIndex ? ' active' : ''}" type="button" data-detection-index="${index}" aria-pressed="${index === state.selectedDetectionIndex}">
      <img src="${createRoiPreview(state.bitmap, detection.box)}" alt="ROI ${index + 1}: ${escapeHtml(detection.className)}" />
      <span class="ai-item-copy"><span><b>${escapeHtml(detection.className)}</b><small>${(detection.confidence * 100).toFixed(1)}% YOLO</small></span><em class="primary-verdict ${verdict.key}">${verdict.label}</em>${confidence}<small>${verdict.detail}</small></span>
      <i aria-hidden="true">→</i>
    </button>`;
  }).join('');
  elements.aiItemsGrid.querySelectorAll('.ai-item-card').forEach((button) => {
    button.addEventListener('click', () => selectDetection(Number(button.dataset.detectionIndex), report));
  });
}

function selectDetection(index, report = state.report) {
  if (!report?.detector?.detections[index]) return;
  state.selectedDetectionIndex = index;
  renderCropGallery(report);
  renderAiItems(report);
  renderSelectedDetection(report, index);
  renderAiModelList(report, index);
  renderBenchmarkRoiList(report, index);
  renderTechnicalSummaries(report, index);
}

function renderBenchmarkReport(report) {
  renderYoloReport(report);
}

function renderBenchmarkTable(report) {
  elements.benchmarkBadge.textContent = report.demo ? 'SIMULATED DATA' : 'REAL INFERENCE';
  elements.benchmarkBadge.classList.toggle('demo', report.demo);
  elements.benchmarkTable.hidden = false;
  elements.benchmarkBody.innerHTML = report.classifications.map((row) => `
    <tr>
      <td class="mono">${String(row.objectIndex).padStart(2, '0')}</td>
      <td><b>${escapeHtml(row.produce)}</b><br><small>${(row.detectionConfidence * 100).toFixed(1)}% detection</small></td>
      <td>${escapeHtml(row.architectureName)}</td>
      <td><span class="result-pill ${row.freshness}">${row.freshness.toUpperCase()}</span></td>
      <td><span class="confidence-bar"><i style="width:${row.confidence * 100}%"></i></span><span class="mono">${(row.confidence * 100).toFixed(1)}%</span></td>
      <td class="mono">${(row.probabilities.fresh * 100).toFixed(1)}%</td>
      <td class="mono">${(row.probabilities.rotten * 100).toFixed(1)}%</td>
      <td class="mono">${row.modelBytes ? formatBytes(row.modelBytes) : '—'}</td>
      <td class="mono">${row.validationMacroF1 == null ? '—' : row.validationMacroF1.toFixed(3)}</td>
      <td class="mono">${formatMs(row.loadMs)}${row.loadMs === 0 ? ' cached' : ''}</td>
      <td class="mono">${formatMs(row.latencyMs)}</td>
    </tr>
  `).join('');
}

function renderSelectedDetection(report, index) {
  const detection = report.detector.detections[index];
  if (!detection) {
    elements.selectedItemDetail.innerHTML = '<p>Không có đối tượng để hiển thị.</p>';
    elements.benchmarkRoiList.innerHTML = '';
    elements.benchmarkEmpty.hidden = false;
    return;
  }
  const preview = createRoiPreview(state.bitmap, detection.box);
  const verdict = deriveDisplayVerdict(report, index);
  const confidenceMarkup = verdict.confidence == null
    ? '<small>Không tạo confidence tổng hợp</small>'
    : `<strong>${(verdict.confidence * 100).toFixed(1)}%</strong>`;
  elements.selectedItemDetail.innerHTML = `
    <div class="selected-preview"><img src="${preview}" alt="Ảnh crop ROI ${index + 1}: ${escapeHtml(detection.className)}" /></div>
    <div class="selected-copy">
      <span class="eyebrow">ROI ${String(index + 1).padStart(2, '0')}</span>
      <h3>${escapeHtml(detection.className)}</h3>
      <div class="selected-primary-verdict ${verdict.key}">
        <span><small>Kết luận hiển thị</small><b>${verdict.label}</b><em>${escapeHtml(verdict.detail)}</em></span>
        ${confidenceMarkup}
      </div>
      <div class="selected-metrics">
        <span><small>YOLO confidence</small><b>${(detection.confidence * 100).toFixed(1)}%</b></span>
        <span><small>Đối tượng</small><b>ROI ${String(index + 1).padStart(2, '0')}</b></span>
      </div>
    </div>`;
  renderTechnicalSummaries(report, index);
}

function renderAiModelList(report, index) {
  const predictions = report.classifications.filter((row) => row.objectIndex === index + 1);
  elements.aiModeBadge.textContent = predictions.length > 1 ? `${predictions.length} MODEL OUTPUTS` : 'SINGLE MODEL OUTPUT';
  elements.aiModelList.innerHTML = predictions.length ? predictions.map((row) => `
    <article class="ai-model-card ${row.freshness}">
      <div class="ai-model-heading"><strong>${escapeHtml(row.architectureName)}</strong><span class="result-pill ${row.freshness}">${freshnessLabel(row.freshness)}</span></div>
      <div class="model-confidence"><small>Confidence của model</small><b>${(row.confidence * 100).toFixed(1)}%</b></div>
      <div class="probability-grid">
        <span><small>P(Fresh)</small><b>${(row.probabilities.fresh * 100).toFixed(1)}%</b></span>
        <span><small>P(Rotten)</small><b>${(row.probabilities.rotten * 100).toFixed(1)}%</b></span>
      </div>
    </article>
  `).join('') : '<div class="inline-empty">Không có classifier nào được chạy cho ROI này.</div>';
}

function renderBenchmarkRoiList(report, index = state.selectedDetectionIndex) {
  const detection = report.detector.detections[index];
  const predictions = report.classifications.filter((row) => row.objectIndex === index + 1);
  elements.benchmarkEmpty.hidden = Boolean(predictions.length);
  const predictionRows = Object.entries(ARCHITECTURES).map(([architecture, architectureName]) => {
    const row = predictions.find((prediction) => prediction.architecture === architecture);
    if (!row) {
      return `<div class="benchmark-prediction unavailable">
        <div class="benchmark-prediction-top"><strong>${escapeHtml(architectureName)}</strong><span class="benchmark-verdict unavailable">CHƯA CHẠY</span></div>
        <p class="model-unavailable-copy">Model này không được chạy trong cấu hình hiện tại.</p>
      </div>`;
    }
    return `<div class="benchmark-prediction ${row.freshness}">
        <div class="benchmark-prediction-top">
          <strong>${escapeHtml(row.architectureName)}</strong>
          <span class="benchmark-verdict ${row.freshness}"><i aria-hidden="true"></i>${freshnessLabel(row.freshness).toUpperCase()}</span>
        </div>
        <div class="model-confidence"><small>Confidence</small><b>${(row.confidence * 100).toFixed(1)}%</b></div>
        <div class="benchmark-metrics">
          <span><small>P(Fresh)</small><b>${(row.probabilities.fresh * 100).toFixed(1)}%</b></span>
          <span><small>P(Rotten)</small><b>${(row.probabilities.rotten * 100).toFixed(1)}%</b></span>
          <span><small>Latency</small><b>${formatMs(row.latencyMs)}</b></span>
          <span><small>Model size</small><b>${row.modelBytes ? formatBytes(row.modelBytes) : '—'}</b></span>
          <span><small>Validation Macro-F1</small><b>${row.validationMacroF1 == null ? '—' : row.validationMacroF1.toFixed(3)}</b></span>
        </div>
      </div>
    `;
  }).join('');
  elements.benchmarkRoiList.innerHTML = detection ? `
      <article class="benchmark-roi-card">
        <div class="benchmark-roi-preview">
          <img src="${createRoiPreview(state.bitmap, detection.box)}" alt="Ảnh crop ROI ${index + 1}: ${escapeHtml(detection.className)}" />
          <span>ROI ${String(index + 1).padStart(2, '0')}</span>
        </div>
        <div class="benchmark-result-side">
          <div class="benchmark-result-heading">
            <div><span>ROI ${String(index + 1).padStart(2, '0')}</span><h3>${escapeHtml(detection.className)}</h3></div>
            <b>${(detection.confidence * 100).toFixed(1)}% YOLO</b>
          </div>
          <div class="benchmark-predictions">${predictionRows}</div>
        </div>
      </article>
  ` : '';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function formatMs(value) {
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(1)} ms`;
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function exportJson() {
  if (!state.reports.length) return;
  const payload = {
    schemaVersion: 2,
    createdAt: new Date().toISOString(),
    requestedImageCount: state.items.length,
    successfulImageCount: state.reports.length,
    failures: state.items
      .map((item, index) => ({ imageIndex: index + 1, imageName: item.file.name, status: item.status, error: item.error }))
      .filter((item) => item.status === 'error'),
    reports: state.reports,
  };
  download(`vnecheck-batch-${Date.now()}.json`, JSON.stringify(payload, null, 2), 'application/json');
}

function exportCsv() {
  if (!state.reports.length) return;
  const header = ['image_index', 'image_name', 'status', 'error', 'object_index', 'produce', 'architecture', 'freshness', 'confidence', 'fresh_probability', 'rotten_probability', 'detection_confidence', 'model_bytes', 'validation_macro_f1', 'model_load_ms', 'inference_latency_ms', 'demo'];
  const lines = state.items.flatMap((item, imageIndex) => {
    if (!item.report) {
      return [[imageIndex + 1, item.file.name, item.status, item.error || '', ...Array(13).fill('')].map(csvCell).join(',')];
    }
    return item.report.classifications.map((row) => [
      imageIndex + 1, item.report.image.name, 'success', '', row.objectIndex, row.produce, row.architecture, row.freshness, row.confidence,
      row.probabilities.fresh, row.probabilities.rotten, row.detectionConfidence, row.modelBytes ?? '',
      row.validationMacroF1 ?? '', row.loadMs, row.latencyMs, item.report.demo,
    ].map(csvCell).join(','));
  });
  download(`vnecheck-batch-${Date.now()}.csv`, [header.join(','), ...lines].join('\n'), 'text/csv;charset=utf-8');
}

function csvCell(value) {
  const text = String(value ?? '');
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const sleep = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function recordCaption(title, detail = '') {
  let overlay = document.querySelector('#record-demo-caption');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'record-demo-caption';
    overlay.innerHTML = '<strong></strong><span></span>';
    Object.assign(overlay.style, {
      position: 'fixed', left: '50%', bottom: '28px', zIndex: '9999', transform: 'translateX(-50%)',
      width: 'min(1120px, calc(100vw - 80px))', padding: '18px 28px', borderRadius: '18px',
      background: 'rgba(5, 20, 30, 0.92)', border: '1px solid rgba(61, 220, 151, 0.75)',
      color: '#f5fbff', boxShadow: '0 18px 55px rgba(0, 0, 0, 0.35)', textAlign: 'center',
      fontFamily: 'Inter, Segoe UI, sans-serif', pointerEvents: 'none', backdropFilter: 'blur(12px)',
    });
    Object.assign(overlay.querySelector('strong').style, { display: 'block', fontSize: '28px', lineHeight: '1.2' });
    Object.assign(overlay.querySelector('span').style, { display: 'block', marginTop: '6px', color: '#b8ccd7', fontSize: '18px' });
    document.body.appendChild(overlay);
  }
  overlay.querySelector('strong').textContent = title;
  overlay.querySelector('span').textContent = detail;
}

async function loadRecordDemoImages() {
  const names = [
    'test_0022.jpg',
    'test_0044.jpg',
    'test_0080.png',
    'test_0090.png',
    'test_0098.png',
    'test_0117.png',
  ];
  const transfer = new DataTransfer();
  for (const name of names) {
    const response = await fetch(`./demo_assets/${name}`);
    if (!response.ok) throw new Error(`Không tải được ảnh demo ${name}`);
    const blob = await response.blob();
    transfer.items.add(new File([blob], name, { type: blob.type, lastModified: Date.now() }));
  }
  await handleFiles(transfer.files);
}

async function runRecordDemo() {
  if (new URLSearchParams(window.location.search).get('record') !== '1') return;
  document.body.dataset.recordDemo = 'true';
  recordCaption('VnECheck e30', 'Demo giao diện thật · không dùng dữ liệu mô phỏng');
  await sleep(9000);

  elements.architecture.value = 'all';
  elements.provider.value = navigator.gpu ? 'webgpu' : 'wasm';
  elements.repeats.value = '1';
  elements.repeats.dispatchEvent(new Event('input', { bubbles: true }));
  elements.demoMode.checked = false;
  updateRunState();

  recordCaption('Nạp sáu ảnh từ Downloads/test', 'Có ảnh nhiều loại, nhiều trái và nhiều ROI');
  await loadRecordDemoImages();
  await sleep(5000);

  recordCaption('Chạy toàn bộ pipeline', 'YOLO11l e30 → crop ROI → ba kiến trúc classifier');
  const progressMirror = window.setInterval(() => {
    const value = elements.progressText?.textContent?.trim();
    if (value) recordCaption('Đang phân tích bằng model e30', value);
  }, 600);
  await runYoloTab();
  window.clearInterval(progressMirror);
  await sleep(3000);

  for (let index = 0; index < state.items.length; index += 1) {
    selectImage(index, true);
    switchTab('detection', false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    const report = state.items[index].report;
    const boxCount = report?.detector?.detections?.length || 0;
    recordCaption(`Ảnh ${index + 1}/${state.items.length} · Nhận diện và crop ROI`, `${state.items[index].file.name} · ${boxCount} đối tượng được phát hiện`);
    await sleep(5500);

    switchTab('results', false);
    if (report?.detector?.detections?.length) selectDetection(0, report);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    recordCaption(`Ảnh ${index + 1}/${state.items.length} · Kết quả AI`, 'Mỗi ROI có kết quả Fresh / Rotten và xác suất riêng');
    await sleep(5000);

    switchTab('benchmark', false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    recordCaption(`Ảnh ${index + 1}/${state.items.length} · Benchmark`, 'So sánh MobileNetV3-Large, ResNet-18 và EfficientNet-B0');
    await sleep(5500);

    const roiButtons = Array.from(document.querySelectorAll('#benchmark-roi-list button'));
    if (roiButtons.length > 1) {
      roiButtons[Math.min(1, roiButtons.length - 1)].click();
      recordCaption(`Ảnh ${index + 1}/${state.items.length} · Chuyển ROI`, 'Benchmark cập nhật theo đối tượng đang được chọn');
      await sleep(4000);
    }
  }

  switchTab('benchmark', false);
  document.querySelector('#technical-details')?.setAttribute('open', '');
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  recordCaption('Chi tiết hiệu năng', 'Latency không gồm thời gian tải model · kết quả được giữ riêng cho từng ROI');
  await sleep(9000);

  window.scrollTo({ top: 0, behavior: 'smooth' });
  recordCaption('Demo hoàn tất', 'Pipeline e30 đã chạy bằng inference thật trên sáu ảnh ngoài');
  document.title = 'VNECHECK_DEMO_DONE';
  await sleep(12000);
}

init().then(runRecordDemo).catch((error) => {
  console.error(error);
  recordCaption('Không thể hoàn tất demo', error.message || String(error));
  document.title = 'VNECHECK_DEMO_DONE_ERROR';
});
