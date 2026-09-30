"use strict";

const $ = (id) => document.getElementById(id);
const icons = () => window.lucide?.createIcons();
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const icon = (name) => `<i data-lucide="${name}"></i>`;
const cameraNames = {top_head:"顶视相机",hand_right:"右腕相机",hand_left:"左腕相机",book_side:"书架侧视"};
const stateNames = {CONNECTING:"连接中",READY:"就绪",ARMABLE:"可开始 · 目标待初始化",WAITING:"等待设备",RECORDING:"录制中",SAVING:"处理数据中",REVIEW:"待审核",ERROR:"服务异常"};
let status = null, connected = false, busy = false, taskDirty = false, view = "capture";
let catalogKey = "", deviceKey = "", pendingKey = "", camerasKey = "", toastTimer;
let recording = null, reviewRequest = 0, videos = [], seeking = false, playRequested = false;
let discardId = null, lastPlaybackFrame = -1;
const cameraRequests = new Map();
const selected = new Set();
let selectionRevision = "", deleteRequest = null;
let lastCommandError = "";
const pendingSelected = new Set();
let batchRequest = null, exportRequest = null, robotSelection = null, robotPlan = null, ownedReplay = null;
const rowKey = (row) => `${row.kind}:${row.id}`;
const unavailable = () => !connected || busy || ["RECORDING","SAVING","REPLAYING"].includes(status?.state);
function applyStatus(next) {
  if (status?.instance === next.instance && (next.sequence ?? 0) < (status.sequence ?? 0)) return;
  status = next; connected = true;
}

function timeText(seconds, fractional = false) {
  seconds = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
  return `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}${fractional ? `.${Math.floor((seconds % 1) * 10)}` : ""}`;
}
function toast(message, error = false) {
  $("toast").textContent = message; $("toast").className = `toast${error ? " error" : ""}`;
  $("toast").hidden = false; clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").hidden = true, error ? 7000 : 3000);
}
function setView(name) {
  view = name; $("capture-view").hidden = name !== "capture"; $("history-view").hidden = name !== "history";
  $("view-title").textContent = name === "capture" ? "采集工作台" : "数据记录";
  document.querySelectorAll("[data-view]").forEach((el) => {
    el.classList.toggle("active", el.dataset.view === name);
    if (el.getAttribute("role") === "tab") el.setAttribute("aria-selected", String(el.dataset.view === name));
  });
}
function sourceSummary(field) {
  const entries = Object.entries(status?.streams || {}).filter(([key]) => key.includes(`.${field}.`));
  const ready = entries.length > 0 && entries.every(([,value]) => value.ready);
  const hz = ready ? Math.round(Math.min(...entries.map(([,value]) => value.hz))) : 0;
  return {ready, hz, missing:entries.filter(([,value]) => !value.ready).map(([key]) => key)};
}
function describeProblem() {
  if (!connected) return "采集服务连接中断";
  if (status?.state === "ERROR") return status.error || "采集服务异常";
  if (status?.replay?.active) return "真机回放中";
  if (status?.replay_needs_reconnect) return "回放已结束，恢复主从连接后请重新连接采集设备";
  if (status?.source_error?.includes("CAN")) return "CAN 数据连接异常";
  if (status?.awaiting_action) return "已开始，等待首组主臂目标后写入数据";
  if (status?.can_start && !status.ready) return "可以开始；主臂目标到达后自动写入数据";
  const leader = sourceSummary("action"), follower = sourceSummary("state");
  if (!leader.ready) return "等待主臂目标数据";
  if (!follower.ready) return "等待从臂反馈";
  if (status.cameras?.some((c) => !status.streams[c.name]?.ready)) return "等待相机画面";
  if (!status.ready) return "等待采样时间轴对齐";
  return "全部数据流就绪";
}
async function poll() {
  try {
    const response = await fetch("/api/status", {cache:"no-store",signal:AbortSignal.timeout(4000)});
    if (!response.ok) throw new Error("status");
    applyStatus(await response.json());
    if (status.cameras) render();
  } catch (_) {
    connected = false; $("connection-text").textContent = "服务离线";
    $("connection-dot").className = "dot red"; $("record-button").disabled = true;
    $("notice").hidden = false; $("notice-text").textContent = "采集服务连接中断";
    document.querySelectorAll(".camera-frame").forEach((el) => el.classList.remove("live"));
    updateReviewActions();
  }
  setTimeout(poll, 400);
}
function render() {
  const recordingNow = status.state === "RECORDING", processing = ["SAVING","REPLAYING"].includes(status.state);
  $("mode-badge").textContent = status.mock ? "模拟模式" : "实机模式";
  $("mode-badge").className = `badge ${status.mock ? "amber" : "neutral"}`;
  $("connection-text").textContent = "本地已连接"; $("connection-dot").className = "dot green";
  $("session-state").textContent = status.state === "REPLAYING" ? "真机回放中" : status.awaiting_action ? "等待首组目标" : stateNames[status.state] || status.state;
  $("session-dot").className = `dot ${recordingNow ? "red" : status.ready ? "green" : "amber"}`;
  $("stat-episodes").innerHTML = `${status.history.length}<span>条</span>`;
  $("success-count").textContent = status.success_count;
  $("failure-count").textContent = status.failure_count;
  $("stat-duration").innerHTML = `${timeText(status.total_frames / status.fps)}<span>时长</span>`;
  $("stat-disk").innerHTML = `${status.free_gb ?? "--"}<span>GB</span>`;
  $("output-path").textContent = status.output; $("output-path").title = status.output;
  if (!taskDirty && document.activeElement !== $("task-input")) $("task-input").value = status.task;
  $("task-input").disabled = recordingNow || processing || busy;
  $("batch-mode").disabled = recordingNow || processing || busy;
  $("arm-type").textContent = status.arms.length === 1 ? "单右臂 · 7 维" : "双臂 · 14 维";
  $("capture-fps").innerHTML = `${status.fps} <small>FPS</small>`;
  $("capture-limit").innerHTML = `${status.max_episode_s} <small>秒</small>`;
  $("record-clock").textContent = timeText(status.duration_s, true);
  $("record-frames").textContent = status.frames;
  $("record-state-label").textContent = status.awaiting_action ? "等待首组主臂目标" : recordingNow ? "正在录制" : "本条轨迹";
  const button = $("record-button");
  const mode = processing ? "saving" : recordingNow ? ($("batch-mode").checked ? "review" : "stop") : "start";
  if (button.dataset.mode !== mode) {
    button.dataset.mode = mode;
    button.innerHTML = `${icon(processing ? "loader-circle" : recordingNow ? "square" : "circle")}<span>${processing ? (status.state === "REPLAYING" ? "真机回放中" : "处理数据中") : recordingNow ? ($("batch-mode").checked ? "停止并待审核" : "停止并保存") : "开始录制"}</span>`;
    icons();
  }
  button.classList.toggle("recording", recordingNow);
  button.disabled = busy || processing || (!recordingNow && (!status.can_start || !$("task-input").value.trim()));
  $("failure-button").disabled = busy || processing || (!recordingNow && status.pending.length !== 1);
  $("stop-review-button").disabled = busy || processing || !recordingNow;
  $("quality-text").textContent = status.pending.length && !recordingNow ? "等待轨迹审核" : describeProblem();
  $("quality-dot").className = `dot ${status.ready ? "green" : "amber"}`;
  $("queue-depth").textContent = recordingNow ? `队列 ${status.queue_depth}` : "";
  const warning = status.error || (!status.ready ? describeProblem() : "");
  $("notice").hidden = !warning; $("notice-text").textContent = warning;
  renderCameras(); renderPositions(); renderCatalog(); renderEvents(); renderDevices(); updateReviewActions(); renderRobot();
}
function renderCameras() {
  const key = JSON.stringify(status.cameras);
  if (key !== camerasKey) {
    camerasKey = key;
    $("camera-grid").innerHTML = status.cameras.map((c) => `<figure class="camera-view"><div class="camera-frame" id="camera-${escapeHTML(c.name)}"><div class="camera-placeholder">${icon("camera-off")}<span>等待画面</span></div><img alt="${escapeHTML(cameraNames[c.name] || c.name)}实时画面"><span class="camera-live-badge"><b class="dot green"></b>LIVE</span><button class="icon-button fullscreen-button" data-camera="${escapeHTML(c.name)}" title="全屏画面" aria-label="${escapeHTML(cameraNames[c.name] || c.name)}全屏">${icon("maximize")}</button></div><figcaption class="camera-caption"><span class="camera-name">${escapeHTML(cameraNames[c.name] || c.name)}</span><span class="camera-rate" id="rate-${escapeHTML(c.name)}">-- FPS</span></figcaption><div class="camera-serial mono">${escapeHTML(c.serial || c.device)} · ${c.width} × ${c.height}</div></figure>`).join("");
    icons();
  }
  const ready = status.cameras.filter((c) => status.streams[c.name]?.ready).length;
  $("camera-summary").textContent = `${ready} / ${status.cameras.length} 在线`;
  for (const camera of status.cameras) {
    const stream = status.streams[camera.name];
    $(`rate-${camera.name}`).textContent = stream?.ready ? `${stream.hz.toFixed(1)} FPS` : "未就绪";
    if (!stream?.ready) $(`camera-${camera.name}`).classList.remove("live");
  }
}
async function refreshCamera(camera) {
  if (cameraRequests.get(camera.name)?.loading) return;
  const previous = cameraRequests.get(camera.name) || {};
  cameraRequests.set(camera.name, {...previous, loading:true});
  try {
    const response = await fetch(`/api/cameras/${camera.name}.jpg`, {cache:"no-store", signal:AbortSignal.timeout(2000)});
    if (!response.ok) throw new Error("camera offline");
    const url = URL.createObjectURL(await response.blob());
    const frame = $(`camera-${camera.name}`), img = frame?.querySelector("img");
    if (img) {
      img.onload = () => { if (connected && status.streams[camera.name]?.ready) frame.classList.add("live"); };
      img.src = url;
    }
    if (previous.url) URL.revokeObjectURL(previous.url);
    cameraRequests.set(camera.name, {url,loading:false});
  } catch (_) {
    $(`camera-${camera.name}`)?.classList.remove("live");
    cameraRequests.set(camera.name, {...previous,loading:false});
  }
}
setInterval(() => {
  if (!document.hidden && connected && view === "capture" && !$("review-dialog").open) {
    status?.cameras?.forEach((c) => { if (status.streams[c.name]?.ready) refreshCamera(c); });
  }
}, 120);
function renderPositions() {
  if ($( "joint-grid").children.length !== status.names.length) {
    $("joint-grid").innerHTML = status.names.map((name,i) => `<div class="joint"><div class="joint-label">${status.arms.length > 1 ? (i < 7 ? "左 " : "右 ") : ""}${i % 7 === 6 ? "夹爪" : `J${i % 7 + 1}`}<small>${i % 7 === 6 ? "mm" : "rad"}</small></div><div class="joint-value mono" id="joint-state-${i}">--</div><div class="joint-value target mono" id="joint-action-${i}">--</div></div>`).join("");
  }
  for (const field of ["state","action"]) {
    status.positions[field].forEach((value,i) => {
      $(`joint-${field}-${i}`).textContent = value === null ? "--" : (value * (i % 7 === 6 ? 1000 : 1)).toFixed(i % 7 === 6 ? 1 : 3);
    });
  }
  for (const [id,field,label] of [["follower-status","state","从臂反馈"],["leader-status","action","主臂目标"]]) {
    const source = sourceSummary(field);
    $(id).innerHTML = `<span class="dot ${source.ready ? "green" : "amber"}"></span>${label} ${source.ready ? (source.hz ? `${source.hz} Hz` : "目标保持") : "未就绪"}`;
  }
  $("can-label").textContent = [...new Set(status.arms.flatMap((a) => [a.state_can,a.command_can]))].join(" / ");
}
function renderCatalog() {
  if (selectionRevision !== status.catalog_revision) {
    selected.clear(); selectionRevision = status.catalog_revision;
  }
  $("history-count").textContent = status.history.length; $("saved-count").textContent = status.history.length;
  $("pending-count").textContent = status.pending.length;
  const key = JSON.stringify(status.pending);
  if (key !== pendingKey) {
    pendingKey = key;
    $("pending-list").innerHTML = status.pending.length ? status.pending.map((row) => `<article class="pending-item ${row.issues?.length ? "failed" : ""}"><div class="pending-top"><input type="checkbox" data-pending-select="${row.id}" aria-label="选择待审核轨迹 ${row.id.slice(0,8)}"><span class="badge ${row.issues?.length ? "amber" : "green"}">${row.issues?.length ? "检查异常" : "待审核"}</span><span class="mono">${escapeHTML(row.id.slice(0,8))}</span></div><p>${escapeHTML(row.task || "未完成的录制")}</p><div class="pending-bottom"><span>${row.length ?? "--"} 帧 · ${timeText((row.length || 0) / status.fps)}</span><div class="pending-buttons"><button class="icon-button" data-discard="${row.id}" title="丢弃轨迹" aria-label="丢弃轨迹">${icon("trash-2")}</button>${row.length ? `<button class="button secondary-button" data-review="${row.id}" data-kind="pending">${icon("play")}审核</button>` : ""}</div></div></article>`).join("") : `<div class="empty-inline">${icon("clipboard-check")}<span>暂无待审核轨迹</span></div>`;
    icons();
  }
  const historyKey = JSON.stringify(status.history) + $("history-search").value + $("outcome-filter").value;
  if (historyKey !== catalogKey) {
    catalogKey = historyKey;
    const filtered = filteredHistory();
    $("history-empty").hidden = filtered.length > 0;
    $("history-empty").querySelector("h3").textContent = status.history.length ? "没有匹配的轨迹" : "暂无采集记录";
    $("history-body").innerHTML = filtered.map((row) => `<tr><td><input type="checkbox" data-select="${rowKey(row)}" aria-label="选择轨迹 ${escapeHTML(row.id)}"></td><td class="mono">${row.kind === "saved" ? `episode_${String(row.episode_index).padStart(6,"0")}` : row.id.slice(0,8)}</td><td class="export-name mono" data-export-preview="${rowKey(row)}"><span class="export-directory"></span><span class="export-episode">--</span></td><td class="history-task">${escapeHTML(row.task)}</td><td class="mono">${timeText(row.length / status.fps,true)}</td><td class="mono">${row.length}</td><td><span class="badge ${row.outcome === "success" ? "green" : "red"}">${row.outcome === "success" ? "成功" : "失败"}</span></td><td class="align-right"><button class="button secondary-button" data-review="${row.id}" data-kind="${row.kind}" ${row.length ? "" : "disabled"}>${icon("play")}查看</button></td></tr>`).join("");
    icons();
  }
  renderSelection(); renderBatch();
}
function renderBatch() {
  const ids = new Set(status.pending.map((r) => r.id));
  for (const id of pendingSelected) if (!ids.has(id)) pendingSelected.delete(id);
  $("batch-toolbar").hidden = !ids.size;
  $("pending-select-all").checked = ids.size > 0 && pendingSelected.size === ids.size;
  $("pending-select-all").indeterminate = pendingSelected.size > 0 && pendingSelected.size < ids.size;
  $("pending-selection-count").textContent = `已选 ${pendingSelected.size} 条`;
  for (const id of ["batch-accept","batch-failure","batch-discard"]) $(id).disabled = unavailable() || !pendingSelected.size;
  $("pending-select-all").disabled = unavailable();
  document.querySelectorAll("[data-pending-select]").forEach((el) => {
    el.checked = pendingSelected.has(el.dataset.pendingSelect); el.disabled = unavailable();
  });
  document.querySelectorAll("[data-discard]").forEach((el) => { el.disabled = unavailable(); });
}
function filteredHistory() {
  const query = $("history-search").value.toLowerCase(), outcome = $("outcome-filter").value;
  return status.history.filter((row) => (outcome === "all" || row.outcome === outcome) && `${row.task} ${row.id} ${String(row.episode_index ?? "").padStart(6,"0")}`.toLowerCase().includes(query));
}
function renderSelection() {
  const rows = filteredHistory(), count = rows.filter((r) => selected.has(rowKey(r))).length;
  const exportNames = selectedExportNames();
  $("select-all").checked = rows.length > 0 && count === rows.length;
  $("select-all").indeterminate = count > 0 && count < rows.length;
  $("select-all").disabled = !rows.length || busy;
  $("selection-count").textContent = `已选 ${selected.size} 条`;
  $("export-selected").disabled = $("delete-selected").disabled = unavailable() || !selected.size;
  $("select-number").disabled = busy || !rows.length;
  document.querySelectorAll("[data-select]").forEach((el) => {
    el.checked = selected.has(el.dataset.select); el.disabled = busy;
    el.closest("tr").classList.toggle("selected", el.checked);
  });
  document.querySelectorAll("[data-export-preview]").forEach((el) => {
    const name = exportNames.get(el.dataset.exportPreview);
    el.querySelector(".export-directory").textContent = name ? `${name.directory}/` : "";
    el.querySelector(".export-episode").textContent = name?.episode || "--";
    el.title = name ? `${name.directory}/${name.episode}` : "未选中";
  });
}
function selectedExportNames() {
  const chosen = status.history.filter((row) => selected.has(rowKey(row))), names = new Map();
  // Match DatasetStore.export_selected: saved episodes sort by index; failures retain request order.
  const saved = chosen.filter((row) => row.kind === "saved").sort((a,b) => a.episode_index - b.episode_index);
  const failed = chosen.filter((row) => row.kind === "failed");
  for (const [rows,directory] of [[saved,"success"],[failed,"failed"]]) {
    rows.forEach((row,index) => names.set(rowKey(row), {directory,episode:`episode_${String(index).padStart(6,"0")}`}));
  }
  return names;
}
function selectionPayload() {
  return {revision:selectionRevision, selections:status.history.filter((r) => selected.has(rowKey(r))).map(({kind,id}) => ({kind,id}))};
}
function renderEvents() {
  const events = status.events.slice(0,4);
  const key = JSON.stringify(events);
  if ($( "event-list").dataset.key === key) return;
  $("event-list").dataset.key = key;
  $("event-list").innerHTML = events.map((event) => `<li><time>${new Date(event.time * 1000).toLocaleTimeString("zh-CN",{hour12:false})}</time><span>${escapeHTML(event.message)}</span></li>`).join("");
}
function renderDevices() {
  if (!$("devices-dialog").open) return;
  const entries = Object.entries(status.streams);
  const key = JSON.stringify(entries);
  if (key !== deviceKey) {
    deviceKey = key;
    $("device-list").innerHTML = entries.map(([name,data]) => `<div class="device-row"><span class="mono">${escapeHTML(name)}</span><span>${data.hz.toFixed(0)} Hz</span><span>${data.age_s === null ? "--" : `${(data.age_s * 1000).toFixed(0)} ms`}</span><span class="badge ${data.ready ? "green" : "amber"}">${data.ready ? "在线" : "缺数据"}</span></div>`).join("");
  }
  $("device-error").textContent = (status.source_error || "") + (!sourceSummary("action").ready ? "\n主臂命令 CAN ID: 0x155 / 0x156 / 0x157 / 0x159" : "");
  $("device-mode").textContent = status.mock ? "模拟数据源" : "被动监听 · 不发送运动指令";
  $("reconnect-button").disabled = unavailable();
}
async function command(name, extra = {}) {
  if (busy || !connected) return false;
  lastCommandError = "";
  busy = true; render();
  try {
    const response = await fetch("/api/command", {method:"POST",headers:{"Content-Type":"application/json","X-Capture-Token":status.token},body:JSON.stringify({command:name,...extra})});
    const result = await response.json();
    if (result.status) applyStatus(result.status);
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "操作失败");
    if (name === "start") taskDirty = false;
    return result;
  } catch (error) { lastCommandError = error.message; toast(error.message, true); return false; }
  finally { busy = false; render(); }
}
async function openReview(kind, id) {
  pauseVideos(); recording = null; videos = [];
  const request = ++reviewRequest;
  $("review-title").textContent = kind === "saved" ? `轨迹 ${String(id).padStart(6,"0")}` : `轨迹 ${id.slice(0,8)}`;
  $("review-kind").textContent = kind === "pending" ? "采集审核" : "轨迹回放";
  $("review-loading").hidden = false; $("review-loading").textContent = "加载轨迹中";
  $("review-content").hidden = true; $("review-actions").hidden = true; $("review-details").textContent = "";
  $("review-videos").replaceChildren(); $("review-dialog").showModal();
  try {
    const response = await fetch(`/api/recordings/${kind}/${id}`);
    if (!response.ok) throw new Error("轨迹不存在或录制未完成");
    const data = await response.json();
    if (request !== reviewRequest || !$("review-dialog").open) return;
    recording = data;
    $("review-task").textContent = data.task;
    $("review-details").textContent = `${data.length} 帧 · ${timeText(data.length / data.fps, true)} · ${data.fps} FPS`;
    $("review-videos").innerHTML = Object.entries(data.videos).map(([name,url]) => `<figure><video preload="auto" muted playsinline src="${url}" aria-label="${escapeHTML(cameraNames[name] || name)}回放"></video><figcaption>${escapeHTML(cameraNames[name] || name)}</figcaption></figure>`).join("");
    videos = [...$("review-videos").querySelectorAll("video")];
    for (const video of videos) {
      video.addEventListener("error", () => toast("视频解码失败", true));
      video.addEventListener("loadedmetadata", () => { updatePlayback(); updateReviewActions(); });
      video.playbackRate = Number($("playback-speed").value);
    }
    videos[0].addEventListener("ended", () => {
      if ($( "loop-toggle").checked) { seekTo(0); startVideos(); } else pauseVideos();
    });
    $("seek-input").max = data.length - 1; $("seek-input").value = 0;
    $("joint-select").innerHTML = data.names.map((name,i) => `<option value="${i}">${data.names.length > 7 ? (i < 7 ? "左臂 " : "右臂 ") : ""}${i % 7 === 6 ? "夹爪 / mm" : `J${i % 7 + 1} / rad`}</option>`).join("");
    $("review-issues").innerHTML = data.issues.length ? data.issues.map((s) => `<div>${escapeHTML(s)}</div>`).join("") : '<span class="badge green">数据完整性检查通过</span>';
    $("review-loading").hidden = true; $("review-content").hidden = false;
    updateReviewActions(); updatePlayback(); drawChart();
  } catch (error) { $("review-loading").textContent = error.message; }
}
function updateReviewActions() {
  const pending = recording?.kind === "pending";
  $("review-actions").hidden = !pending;
  const stillExists = status?.pending?.some((r) => r.id === recording?.id);
  const available = !unavailable() && stillExists;
  $("accept-button").disabled = !available || Boolean(recording?.issues.length);
  $("discard-button").disabled = !available;
  $("review-failure-button").disabled = !available;
  $("robot-replay-button").disabled = unavailable() || !recording || Boolean(recording?.issues.length);
  $("play-button").disabled = !videos.length || videos.some((v) => v.readyState < 1);
}
function pauseVideos() {
  playRequested = false; videos.forEach((v) => v.pause());
  $("play-button").innerHTML = icon("play"); $("play-button").setAttribute("aria-label","播放"); $("play-button").title = "播放"; icons();
}
async function startVideos() {
  if (!videos.length || !recording) return;
  if (videos[0].currentTime >= (recording.length - 1) / recording.fps) seekTo(0);
  playRequested = true;
  const time = videos[0].currentTime;
  videos.forEach((v) => { if (Math.abs(v.currentTime-time) > 0.04) v.currentTime = time; });
  try {
    await Promise.all(videos.map((v) => v.play()));
    if (!playRequested) { videos.forEach((v) => v.pause()); return; }
    $("play-button").innerHTML = icon("pause"); $("play-button").setAttribute("aria-label","暂停"); $("play-button").title = "暂停"; icons();
  } catch (_) { pauseVideos(); toast("回放暂不可用，请等待视频加载", true); }
}
function seekTo(frame) {
  if (!recording) return;
  const time = Math.max(0,Math.min(recording.length - 1,frame)) / recording.fps;
  videos.forEach((v) => { if (v.readyState >= 1) v.currentTime = time; });
  $("seek-input").value = Math.round(time * recording.fps); drawChart();
}
function updatePlayback() {
  if (!recording || !videos.length) return;
  const time = videos[0].currentTime;
  $("playback-time").textContent = `${timeText(time,true)} / ${timeText(recording.length / recording.fps,true)}`;
  if (!seeking) $("seek-input").value = Math.min(recording.length-1,Math.round(time * recording.fps));
  if (playRequested) videos.slice(1).forEach((v) => {
    if (v.readyState >= 2 && Math.abs(v.currentTime - time) > 0.1) v.currentTime = time;
  });
  const frame = Math.round(time * recording.fps);
  if (lastPlaybackFrame !== frame) { lastPlaybackFrame = frame; drawChart(); }
}
setInterval(updatePlayback, 60);
function drawChart() {
  if (!recording || $("review-content").hidden) return;
  const canvas = $("trajectory-chart"), rect = canvas.getBoundingClientRect(), scale = window.devicePixelRatio || 1;
  if (!rect.width) return;
  canvas.width = Math.round(rect.width * scale); canvas.height = Math.round(rect.height * scale);
  const ctx = canvas.getContext("2d"); ctx.scale(scale,scale);
  const w = rect.width, h = rect.height, p = {left:43,right:10,top:12,bottom:23};
  const joint = Number($("joint-select").value), factor = joint % 7 === 6 ? 1000 : 1;
  const chart = recording.chart, series = [chart.state.map((v) => v[joint]*factor),chart.action.map((v) => v[joint]*factor)];
  let min = Math.min(...series.flat()), max = Math.max(...series.flat());
  const margin = Math.max((max-min)*.12, .005); min -= margin; max += margin;
  const x = (i) => p.left + i / Math.max(1,recording.length-1) * (w-p.left-p.right);
  const y = (v) => h-p.bottom - (v-min)/(max-min)*(h-p.top-p.bottom);
  ctx.font = "10px monospace"; ctx.textAlign = "right";
  for (let i=0;i<3;i++) {
    const value = min + i*(max-min)/2, top = y(value);
    ctx.strokeStyle = "#e2e8e9"; ctx.beginPath(); ctx.moveTo(p.left,top); ctx.lineTo(w-p.right,top); ctx.stroke();
    ctx.fillStyle = "#8b989d"; ctx.fillText(value.toFixed(2),p.left-7,top+3);
  }
  series.forEach((values,index) => {
    ctx.strokeStyle = ["#248b68","#c58248"][index]; ctx.lineWidth = 1.7; ctx.beginPath();
    values.forEach((v,i) => { if (i) ctx.lineTo(x(chart.indices[i]),y(v)); else ctx.moveTo(x(chart.indices[i]),y(v)); }); ctx.stroke();
  });
  ctx.fillStyle = "#8b989d"; ctx.textAlign = "left"; ctx.fillText("0 s",p.left,h-5);
  ctx.textAlign = "right"; ctx.fillText(`${((recording.length-1)/recording.fps).toFixed(1)} s`,w-p.right,h-5);
  ctx.strokeStyle = "#98a5a9"; ctx.lineWidth = 1; ctx.setLineDash([3,3]); ctx.beginPath();
  const cursor = x(Number($("seek-input").value));ctx.moveTo(cursor,p.top);ctx.lineTo(cursor,h-p.bottom);ctx.stroke();
}
function closeReview() {
  reviewRequest++; pauseVideos(); videos.forEach((v) => { v.removeAttribute("src"); v.load(); });
  videos = []; recording = null; $("review-dialog").close();
}
function requestDiscard(id, task) {
  discardId = id; $("discard-task").textContent = task || "未完成的录制"; $("discard-dialog").showModal();
}

document.addEventListener("click", (event) => {
  const viewButton = event.target.closest("[data-view]"); if (viewButton) setView(viewButton.dataset.view);
  const reviewButton = event.target.closest("[data-review]"); if (reviewButton) openReview(reviewButton.dataset.kind,reviewButton.dataset.review);
  const discardButton = event.target.closest("[data-discard]"); if (discardButton) requestDiscard(discardButton.dataset.discard,status.pending.find((r) => r.id === discardButton.dataset.discard)?.task);
  const fullscreen = event.target.closest(".fullscreen-button"); if (fullscreen) $(`camera-${fullscreen.dataset.camera}`).requestFullscreen?.().catch(() => toast("无法进入全屏",true));
});
$("task-input").addEventListener("input", () => { taskDirty = true; if (status) render(); });
$("history-search").addEventListener("input", () => status && renderCatalog());
$("outcome-filter").addEventListener("change", () => status && renderCatalog());
async function toggleRecording() {
  if ($("record-button").disabled) return;
  if (status?.state === "RECORDING") {
    const review = $("batch-mode").checked;
    if (await command(review ? "stop" : "save_success")) toast(review ? "轨迹已加入待审核" : "成功示教已保存");
  } else if (await command("start", {task:$("task-input").value.trim()})) toast("开始录制");
}
async function saveFailure() {
  if ($("failure-button").disabled) return;
  const extra = status.state === "RECORDING" ? {} : {recording_id:status.pending[0].id};
  if (await command(status.state === "RECORDING" ? "save_failure" : "mark_failure", extra)) toast("失败示教已保存");
}
$("record-button").addEventListener("click", toggleRecording);
$("batch-mode").addEventListener("change", () => status && render());
$("failure-button").addEventListener("click", saveFailure);
$("stop-review-button").addEventListener("click", async () => {
  if (await command("stop")) toast("轨迹待审核");
});
document.addEventListener("keydown", (event) => {
  if (event.repeat || event.isComposing || event.ctrlKey || event.metaKey || event.altKey || !connected || busy || view !== "capture") return;
  if (document.querySelector("dialog[open]") || event.target.closest("input,textarea,select,[contenteditable=true]")) return;
  if (event.code === "Space") { event.preventDefault(); toggleRecording(); }
  if (event.code === "KeyL") { event.preventDefault(); saveFailure(); }
});
$("history-body").addEventListener("change", (event) => {
  const key = event.target.dataset.select;
  if (key) { event.target.checked ? selected.add(key) : selected.delete(key); renderSelection(); }
});
$("select-all").addEventListener("change", () => {
  filteredHistory().forEach((r) => $("select-all").checked ? selected.add(rowKey(r)) : selected.delete(rowKey(r))); renderSelection();
});
$("select-number").addEventListener("click", () => {
  const count = Number($("selection-limit").value);
  if (!Number.isInteger(count) || count < 1) return toast("请输入正整数条数",true);
  selected.clear(); filteredHistory().slice(0,count).forEach((r) => selected.add(rowKey(r))); renderSelection();
});
$("export-selected").addEventListener("click", () => {
  exportRequest = selectionPayload();
  $("export-summary").textContent = `已选 ${exportRequest.selections.length} 条轨迹`;
  $("export-error").textContent = "";
  $("export-destination").value = `/media/ubun/16T/Dataset/piper_eva_data/recognition/export_${new Date().toISOString().replace(/[-:]/g,"").slice(0,15)}`;
  $("export-dialog").showModal();
});
$("close-export").addEventListener("click", () => $("export-dialog").close());
async function exportRecords(toDirectory) {
  if (toDirectory && !$("export-destination").value.trim()) {
    $("export-error").textContent = "请输入本机绝对路径"; return;
  }
  $("confirm-export").disabled = $("download-export").disabled = true;
  const result = await command("export", {...exportRequest,...(toDirectory ? {destination:$("export-destination").value.trim()} : {})});
  $("confirm-export").disabled = $("download-export").disabled = false;
  if (!result) { $("export-error").textContent = lastCommandError || "导出未完成，请修改路径或重新选择记录"; return; }
  $("export-dialog").close();
  $("export-result").hidden = false;
  if (result.download) {
    const link = document.createElement("a"); link.href = result.download; link.download = "";
    document.body.append(link); link.click(); link.remove();
    const retry = document.createElement("a"); retry.href = result.download; retry.download = ""; retry.textContent = "重新下载 ZIP";
    $("export-result").replaceChildren("导出完成。", retry);
  } else $("export-result").textContent = `已导出到：${result.destination}`;
  toast("导出完成");
}
$("confirm-export").addEventListener("click", () => exportRecords(true));
$("download-export").addEventListener("click", () => exportRecords(false));
$("pending-list").addEventListener("change", (event) => {
  const id = event.target.dataset.pendingSelect;
  if (id) { event.target.checked ? pendingSelected.add(id) : pendingSelected.delete(id); renderBatch(); }
});
$("pending-select-all").addEventListener("change", () => {
  pendingSelected.clear();
  if ($("pending-select-all").checked) status.pending.forEach((r) => pendingSelected.add(r.id));
  renderBatch();
});
for (const [id,decision,label] of [["batch-accept","accept","接受并保存"],["batch-failure","mark_failure","保存为失败"],["batch-discard","discard","丢弃"]]) {
  $(id).addEventListener("click", () => {
    batchRequest = {decision,recording_ids:[...pendingSelected]};
    $("batch-summary").textContent = `${label}所选 ${pendingSelected.size} 条轨迹？`;
    $("batch-dialog").showModal();
  });
}
$("cancel-batch").addEventListener("click", () => $("batch-dialog").close());
$("confirm-batch").addEventListener("click", async () => {
  $("batch-dialog").close();
  const result = await command("review_batch",batchRequest);
  if (!result) return;
  $("batch-result").hidden = false;
  $("batch-result").textContent = `已处理 ${result.processed} 条，未处理 ${result.remaining} 条。` + result.results.filter((r) => !r.ok).map((r) => `${r.id.slice(0,8)}：${r.error}`).join("；");
  renderBatch();
});
$("delete-selected").addEventListener("click", () => {
  deleteRequest = selectionPayload();
  $("delete-summary").textContent = `将永久删除所选 ${deleteRequest.selections.length} 条轨迹及其视频。`;
  $("delete-dialog").showModal();
});
$("cancel-delete").addEventListener("click", () => $("delete-dialog").close());
$("confirm-delete").addEventListener("click", async () => {
  $("delete-dialog").close();
  if (await command("delete", deleteRequest)) { selected.clear(); renderSelection(); toast("所选轨迹已删除"); }
});
$("copy-path").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(status.output); toast("数据目录已复制"); } catch (_) { toast("无法访问剪贴板",true); }
});
$("devices-button").addEventListener("click", () => { $("devices-dialog").showModal(); if(status) renderDevices(); });
$("close-devices").addEventListener("click", () => $("devices-dialog").close());
$("reconnect-button").addEventListener("click", async () => { if (await command("reconnect")) toast("正在重新连接设备"); });
$("close-review").addEventListener("click", closeReview);
$("review-dialog").addEventListener("cancel", (e) => { e.preventDefault(); closeReview(); });
$("play-button").addEventListener("click", () => playRequested ? pauseVideos() : startVideos());
$("step-back").addEventListener("click", () => { pauseVideos(); seekTo(Number($("seek-input").value)-1); });
$("step-forward").addEventListener("click", () => { pauseVideos(); seekTo(Number($("seek-input").value)+1); });
$("seek-input").addEventListener("input", () => { seeking = true; pauseVideos(); seekTo(Number($("seek-input").value)); });
$("seek-input").addEventListener("change", () => { seeking = false; });
$("playback-speed").addEventListener("change", () => videos.forEach((v) => v.playbackRate = Number($("playback-speed").value)));
$("joint-select").addEventListener("change", drawChart);
window.addEventListener("resize", drawChart);
$("accept-button").addEventListener("click", async () => {
  if (recording && await command("accept", {recording_id:recording.id})) { closeReview(); toast("轨迹已保存到数据集"); }
});
$("review-failure-button").addEventListener("click", async () => {
  if (recording && await command("mark_failure", {recording_id:recording.id})) { closeReview(); toast("失败示教已保存"); }
});
$("discard-button").addEventListener("click", () => recording && requestDiscard(recording.id,recording.task));
$("cancel-discard").addEventListener("click", () => $("discard-dialog").close());
$("confirm-discard").addEventListener("click", async () => {
  $("discard-dialog").close();
  if (await command("discard", {recording_id:discardId})) { closeReview(); toast("轨迹已丢弃"); }
});
function renderRobot() {
  const replay = status?.replay, active = Boolean(replay?.active);
  const phases = {checking:"总线与状态检查中",moving_to_start:"正在回起点",settling_start:"确认起点到位",playing:"回放中",completed:"回放完成",stopped:"回放已停止",error:"回放异常"};
  const returning = ["moving_to_start","settling_start"].includes(replay?.phase);
  const elapsed = returning ? replay.start_elapsed_s : replay?.elapsed_s;
  const duration = returning ? replay.start_duration_s : replay?.duration_s;
  const progress = replay ? `${replay.mock ? "模拟回放" : "真机回放"} · ${phases[replay.phase] || replay.phase} · ${timeText(elapsed,true)} / ${timeText(duration,true)}` : "";
  $("replay-banner").hidden = !active;
  $("replay-status").textContent = progress;
  $("robot-progress").textContent = progress;
  if (replay?.error && ownedReplay === replay.id) $("robot-error").textContent = replay.error;
  if (!active && ownedReplay && replay?.id === ownedReplay) ownedReplay = null;
  $("robot-stop").disabled = !active;
  $("robot-start").disabled = unavailable() || !robotPlan || !$("leader-disconnected").checked || !$("workspace-clear").checked;
  for (const id of ["replay-check","robot-field","robot-speed","robot-gripper","leader-disconnected","workspace-clear"]) $(id).disabled = active || busy;
  document.querySelectorAll("[data-robot-current]").forEach((element) => {
    const i = Number(element.dataset.robotCurrent), value = status.positions.state[i];
    element.textContent = value === null ? "--" : (value * (i === 6 ? 1000 : 1)).toFixed(3);
  });
}
$("robot-replay-button").addEventListener("click", () => {
  pauseVideos(); robotSelection = {kind:recording.kind,recording_id:recording.id,recording_revision:recording.recording_revision}; robotPlan = null;
  $("robot-title").textContent = status.mock ? "模拟真机回放" : "真机回放";
  $("robot-start").innerHTML = `${icon("play")}回起点并回放`; icons();
  $("robot-plan").textContent = "等待预检"; $("robot-error").textContent = ""; $("robot-poses").replaceChildren();
  $("leader-disconnected").checked = $("workspace-clear").checked = false;
  $("robot-dialog").showModal(); renderRobot();
});
function invalidateRobotPlan() {
  robotPlan = null; $("robot-plan").textContent = "等待预检";
  $("robot-poses").replaceChildren(); $("robot-error").textContent = ""; renderRobot();
}
$("robot-field").addEventListener("change", invalidateRobotPlan);
$("robot-speed").addEventListener("change", invalidateRobotPlan);
$("robot-gripper").addEventListener("change", invalidateRobotPlan);
$("leader-disconnected").addEventListener("change", renderRobot);
$("workspace-clear").addEventListener("change", renderRobot);
$("replay-check").addEventListener("click", async () => {
  robotPlan = null; $("robot-error").textContent = ""; $("robot-poses").replaceChildren();
  const result = await command("replay_prepare",{...robotSelection,field:$("robot-field").value,speed:Number($("robot-speed").value),with_gripper:$("robot-gripper").checked});
  if (result?.plan) {
    robotPlan = result.plan;
    const speedLabel = robotPlan.speed === 1 ? "1 倍（采集原速）" : `${robotPlan.speed} 倍`;
    $("robot-plan").textContent = `${robotPlan.frames} 帧 · ${speedLabel} · 轨迹预计 ${timeText(robotPlan.trajectory_duration_s,true)} · 终点保持 ${robotPlan.end_hold_s} 秒（另计回起点） · 原值回放 · 无软件位置限位 · ${robotPlan.with_gripper ? "六轴与夹爪" : "仅六轴，夹爪不动作"}`;
    $("robot-poses").innerHTML = `<table class="pose-table"><thead><tr><th>关节</th><th>起点</th><th>当前</th><th>单位</th></tr></thead><tbody>${robotPlan.first_pose.map((value,i) => `<tr><td>${i === 6 ? "夹爪" : `J${i+1}`}</td><td>${(value*(i===6?1000:1)).toFixed(3)}</td><td data-robot-current="${i}">--</td><td>${i===6?"mm":"rad"}</td></tr>`).join("")}</tbody></table>`;
  } else { $("robot-plan").textContent = "预检未通过"; $("robot-error").textContent = lastCommandError || status.error || "请检查轨迹"; }
  renderRobot();
});
$("robot-start").addEventListener("click", async () => {
  const result = await command("replay_start",{plan_token:robotPlan?.plan_token,leader_disconnected:$("leader-disconnected").checked,workspace_clear:$("workspace-clear").checked});
  if (result?.replay_id) { ownedReplay = result.replay_id; robotPlan = null; renderRobot(); }
});
async function replaySignal(name, id = ownedReplay || status?.replay?.id) {
  if (!id || !status?.token) return;
  try {
    const response = await fetch("/api/command",{method:"POST",keepalive:true,headers:{"Content-Type":"application/json","X-Capture-Token":status.token},body:JSON.stringify({command:name,replay_id:id}),signal:AbortSignal.timeout(1000)});
    if (!response.ok && name === "replay_stop") toast("停止指令未确认，请使用硬件急停",true);
  } catch (_) { if (name === "replay_stop") toast("连接中断，请使用硬件急停",true); }
}
$("robot-stop").addEventListener("click", () => replaySignal("replay_stop"));
$("replay-stop-global").addEventListener("click", () => replaySignal("replay_stop"));
function closeRobot() {
  if (status?.replay?.active) replaySignal("replay_stop");
  ownedReplay = null; robotPlan = null; $("robot-dialog").close();
}
$("close-robot").addEventListener("click",closeRobot);
$("robot-dialog").addEventListener("cancel", (event) => { event.preventDefault(); closeRobot(); });
setInterval(() => { if (ownedReplay && !document.hidden) replaySignal("replay_heartbeat",ownedReplay); },300);
window.addEventListener("pagehide", () => { if (ownedReplay) replaySignal("replay_stop",ownedReplay); });
document.addEventListener("visibilitychange", () => { if (document.hidden && ownedReplay) { replaySignal("replay_stop",ownedReplay); ownedReplay = null; } });
icons(); poll();
