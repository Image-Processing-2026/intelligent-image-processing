"use strict";
const $ = id => document.getElementById(id);
const modelIds = () => Object.keys(config?.models || {});
const isFreeRun = job => job.protocol_version === "vision-free-v1";
const stageNames = job => job.stages || ["analysis","plan"];
let config, selectedFile, previewURL, currentRun, busy = false, timer;
const labels = {pending:"Đang xếp hàng",running:"Đang gọi model…",waiting:"Chờ quota",done:"Hoàn tất",error:"Có lỗi",skipped:"Chưa thực hiện",cancelled:"Đã dừng",not_in_run:"Không có trong lượt"};
const escapeHTML = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function inline(text) { return escapeHTML(text).replace(/`([^`]+)`/g,"<code>$1</code>").replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>"); }
// A deliberately small Markdown renderer: escaped model text, no raw HTML or links.
function markdown(text) {
  const lines = text.replace(/\r/g,"").split("\n"); let out = [], i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      let code = []; i++; while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      out.push(`<pre>${escapeHTML(code.join("\n"))}</pre>`); i++; continue;
    }
    if (/^\s*#{1,6}\s/.test(line)) { out.push(`<h5>${inline(line.replace(/^\s*#{1,6}\s+/,""))}</h5>`); i++; continue; }
    if (line.includes("|") && i+1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i+1])) {
      const cells = s => s.trim().replace(/^\||\|$/g,"").split("|").map(v => v.trim());
      out.push(`<div class="table-wrap"><table><thead><tr>${cells(line).map(v=>`<th>${inline(v)}</th>`).join("")}</tr></thead><tbody>`); i+=2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) out.push(`<tr>${cells(lines[i++]).map(v=>`<td>${inline(v)}</td>`).join("")}</tr>`);
      out.push("</tbody></table></div>"); continue;
    }
    if (/^\s*([-*+] |\d+\. )/.test(line)) {
      const tag = /^\s*\d+\./.test(line) ? "ol" : "ul";
      out.push(`<${tag}>`);
      while(i < lines.length && /^\s*([-*+] |\d+\. )/.test(lines[i])) out.push(`<li>${inline(lines[i++].replace(/^\s*([-*+] |\d+\. )/,""))}</li>`);
      out.push(`</${tag}>`); continue;
    }
    if (/^\s*[-*_]{3,}\s*$/.test(line)) { i++; continue; }
    out.push(`<p>${inline(line)}</p>`); i++;
  }
  return out.join("");
}
async function api(url, options) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : `Yêu cầu thất bại (HTTP ${response.status}).`);
  return value;
}
function notice(text) { $("notice").textContent = text; $("notice").hidden = !text; }
function setBusy(value) {
  busy = value; $("run").disabled = value || !config; $("cancel").hidden = !value; $("image").disabled = value;
  $("system-prompt").readOnly=value;
  for(const id of ["max-edge","max-output","reset-prompts"]) $(id).disabled=value;
}
function chooseFile(file) {
  if (!file || busy) return;
  if (file.size > 20*1024*1024) { notice("Ảnh tối đa 20 MB. Hãy chọn ảnh nhỏ hơn."); return; }
  notice(""); selectedFile = file;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  $("preview").src = previewURL; $("preview").hidden = false; $("image-empty").hidden = true; $("zoom").hidden = false;
  $("large-image").src = previewURL;
  $("image-info").textContent = `${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`;
  $("preview").onerror = () => notice("Trình duyệt không xem trước được ảnh này; server sẽ kiểm tra định dạng khi chạy.");
}
$("image").addEventListener("change", e => chooseFile(e.target.files[0]));
$("dropzone").addEventListener("dragover", e => { e.preventDefault(); if(!busy) $("dropzone").classList.add("dragover"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragover"));
$("dropzone").addEventListener("drop", e => { e.preventDefault(); $("dropzone").classList.remove("dragover"); chooseFile(e.dataTransfer.files[0]); });
$("zoom").onclick = () => $("image-dialog").showModal();
$("close-dialog").onclick = () => $("image-dialog").close();
$("reset-prompts").onclick = () => { $("system-prompt").value = config.system_prompt; };

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return; } catch { /* Fallback for browsers restricting the Clipboard API. */ }
  const active = document.activeElement, field = document.createElement("textarea");
  field.value=text; field.readOnly=true; field.style.cssText="position:fixed;left:-9999px;top:0";
  document.body.append(field); field.select();
  try { if (!document.execCommand("copy")) throw new Error("Clipboard unavailable"); }
  finally { field.remove(); active?.focus(); }
}
for (const button of document.querySelectorAll("[data-copy]")) button.onclick = async () => {
  const result=currentRun?.results[button.dataset.copy];
  if (!result || !stageNames(currentRun).some(stage=>result[stage]?.text)) return;
  const section=(name,stage)=>`## ${name}\n${stage.text || `Chưa có kết quả (${labels[stage.status] || stage.status}).`}${stage.truncated ? "\n\n[Lưu ý: câu trả lời đã chạm giới hạn output và có thể bị cắt.]" : ""}`;
  const content=isFreeRun(currentRun) ? section("Nhận xét của model",result.analysis) : `${section("Phân tích ảnh · lượt cũ",result.analysis)}\n\n${section("Đề xuất xử lý · lượt cũ",result.plan)}`;
  const text=`${result.label} (${result.model})\n\n${content}`;
  try {
    await copyText(text); button.textContent="Đã copy ✓";
    $("copy-feedback").textContent=`Đã copy kết quả ${result.label}.`;
    setTimeout(()=>{button.textContent="Copy kết quả";},1800);
  } catch {
    notice("Trình duyệt không cho phép copy. Bạn có thể mở ‘Xem nguyên văn’ và sao chép nội dung thủ công.");
  }
};

function renderStage(provider, stage, data, job) {
  const node = $(`${provider}-${stage}`), badge = node.querySelector(".stage-status"), body = node.querySelector(".stage-body"), meta = node.querySelector(".stage-meta");
  badge.dataset.status = data.status;
  badge.textContent = data.status === "waiting" ? `Chờ quota · ${data.wait_seconds || 0}s` : data.truncated ? "Chạm giới hạn output" : labels[data.status] || data.status;
  const signature = JSON.stringify([data.status,data.text,data.error,job.status]);
  if (node.dataset.signature === signature) return;
  node.dataset.signature = signature;
  body.className = "stage-body"; body.replaceChildren(); meta.replaceChildren();
  if (data.text) {
    body.innerHTML = markdown(data.text);
    const usage = data.usage || {};
    meta.textContent = `${data.seconds}s API · ${usage.input ?? "?"} input / ${usage.output ?? "?"} output tokens`;
    if (data.truncated) {
      const warning = document.createElement("div"); warning.className = "warning";
      warning.textContent = "Đã chạm giới hạn output: câu trả lời có thể bị cắt. " + (job.max_output_tokens < 1000 ? "Tăng output lên 1.000 tokens hoặc rút gọn yêu cầu cho lượt tiếp theo." : "Rút ngắn prompt hoặc giảm số vấn đề/bước cần trình bày cho lượt tiếp theo."); meta.append(warning);
    }
    const raw = document.createElement("details"), summary = document.createElement("summary"), pre = document.createElement("pre");
    summary.textContent = "Xem nguyên văn"; pre.textContent = data.text; raw.append(summary,pre); body.append(raw);
  } else if (data.status === "error" || data.status === "skipped") {
    const error = document.createElement("div"); error.className = "error-message"; error.textContent = data.error || "Không có kết quả."; body.append(error);
    const hint = document.createElement("p"); hint.className="hint";
    hint.textContent = data.error?.includes("429") ? "Nhà cung cấp giới hạn quota. Chờ rồi thử lại bước này." : "Kết quả của model còn lại vẫn được giữ. Có thể thử lại sau khi lượt này kết thúc."; body.append(hint);
  } else {
    body.classList.add("empty");
    body.textContent = data.status === "running" ? (stage === "analysis" ? "Model đang quan sát ảnh và bối cảnh bạn cung cấp…" : "Model đang lập kế hoạch từ phân tích ở bước 1…")
      : data.status === "waiting" ? "Đang chờ cửa sổ quota trước khi gửi request. Bạn có thể đọc kết quả đã có trong lúc chờ."
      : data.status === "cancelled" ? "Bước này đã dừng trước khi gửi đến API."
      : data.status === "not_in_run" ? "Model này chưa tham gia lượt đã lưu. Chọn lại ảnh và chạy lượt mới để so sánh cả ba model."
      : stage === "plan" ? "Không có kết quả đề xuất trong lượt cũ này." : "Đang chuẩn bị gửi cùng ảnh và system prompt cho các model…";
  }
  if (isFreeRun(job) && !busy && ["error","cancelled"].includes(data.status) && job.status !== "interrupted") {
    const button = document.createElement("button"); button.className="retry"; button.type="button"; button.textContent = `Thử lại ${stage === "analysis" ? "model này" : "bước đề xuất"}`;
    button.onclick = async () => { try { await api(`/api/runs/${job.id}/retry/${provider}/${stage}`,{method:"POST"}); notice(""); setBusy(true); poll(job.id); } catch(e) {notice(e.message);} };
    body.append(button);
  }
}
function render(job) {
  currentRun = job; setBusy(job.status === "running");
  for(const button of document.querySelectorAll("[data-copy]")) {
    const result=job.results[button.dataset.copy];
    button.disabled=!result || !stageNames(job).some(stage=>result[stage]?.text);
  }
  for(const provider of modelIds()) {
    $(`${provider}-plan`).hidden=isFreeRun(job);
    $(`${provider}-analysis`).querySelector("h4").textContent=isFreeRun(job) ? "Nhận xét của model" : "Phân tích ảnh · lượt cũ";
    $(`${provider}-plan`).querySelector("h4").textContent="Đề xuất xử lý · lượt cũ";
    for(const stage of stageNames(job)) renderStage(provider,stage,job.results[provider]?.[stage] || {status:"not_in_run"},job);
  }
  $("protocol-note").textContent=isFreeRun(job) ? "Luồng tự do · 1 request ảnh/model · Không ràng buộc toolbox" : "Kết quả lượt cũ: hai bước và đề xuất theo toolbox. Lượt mới sẽ dùng instruction tự do ở trên; không tự khôi phục prompt cũ.";
  const stages = Object.values(job.results).flatMap(r=>stageNames(job).map(stage=>r[stage]));
  const completed = stages.filter(s=>s.status === "done").length;
  const total=stages.length;
  const unit=isFreeRun(job) ? "model" : "bước của lượt cũ";
  $("overall-status").textContent = job.status === "running" ? `${completed}/${total} ${unit} đã có kết quả. Các model chạy độc lập.`
    : job.status === "interrupted" ? "Lượt trước bị ngắt khi server khởi động lại. Các kết quả đã nhận vẫn được giữ."
    : `${completed}/${total} ${unit} hoàn tất${stages.some(s=>s.status === "error") ? " · Có lỗi API, xem chi tiết tại cột tương ứng." : stages.some(s=>s.status === "cancelled") ? " · Đã dừng các request chưa gửi." : " · Sẵn sàng đối chiếu hoặc thử ảnh tiếp theo."}`;
  const truncatedCount=stages.filter(s=>s.truncated).length;
  if(truncatedCount) $("overall-status").textContent+=` · ${truncatedCount} phản hồi chạm giới hạn output, nội dung có thể bị cắt.`;
  $("download").hidden=false; $("download").href=`/api/runs/${job.id}/download`;
  $("image-sent").hidden=false;
  $("image-sent").textContent = `${job.filename} · Gốc ${job.image.original_size.join(" × ")} px → Gửi ${job.image.sent_size.join(" × ")} px · Cùng byte ảnh cho ${Object.keys(job.results).length} model · Lượt ${job.id.slice(0,8)}`;
}
async function poll(id) {
  clearTimeout(timer);
  try {
    const job = await api(`/api/runs/${id}`); render(job);
    if (job.status === "running") timer = setTimeout(()=>poll(id),1200);
  } catch(e) {
    notice(`Không lấy được tiến độ: ${e.message} Đang thử kết nối lại; không gửi lại request model.`);
    timer = setTimeout(()=>poll(id),3500);
  }
}
$("experiment").addEventListener("submit", async event => {
  event.preventDefault(); if (busy) return;
  if (!selectedFile) { notice("Hãy chọn một tấm ảnh trước khi chạy."); $("image").focus(); return; }
  const data = new FormData(); data.append("image",selectedFile); data.append("system_prompt",$("system-prompt").value);
  data.append("max_edge",$("max-edge").value); data.append("max_output_tokens",$("max-output").value);
  notice(""); setBusy(true); $("overall-status").textContent="Đang nhận và chuẩn hóa ảnh…";
  try {
    const job = await api("/api/runs",{method:"POST",body:data}); localStorage.setItem("image-doctor-last-free-run",job.id);
    await poll(job.id);
  } catch(e) { setBusy(false); notice(e.message); $("overall-status").textContent="Chưa tạo được lượt thử. Kiểm tra lỗi rồi chạy lại."; }
});
$("cancel").onclick = async () => {
  if (!currentRun || !busy) return;
  try { await api(`/api/runs/${currentRun.id}/cancel`,{method:"POST"}); notice("Đã yêu cầu dừng các bước chưa gửi. Request đang gọi API vẫn hoàn tất hoặc timeout."); } catch(e) {notice(e.message);}
};
async function init() {
  try {
    config = await api("/api/config"); $("system-prompt").value=config.system_prompt;
    for(const provider of modelIds()) {
      $(`${provider}-plan`).hidden=true;
      $(`${provider}-analysis`).querySelector("h4").textContent="Nhận xét của model";
      $(`${provider}-analysis`).querySelector(".stage-body").textContent="Nội dung, chi tiết, chất lượng ảnh và hướng cải thiện nếu cần, theo cách trình bày của model.";
    }
    for (const [name,spec] of Object.entries(config.models)) { const node=$(`${name}-key`); node.textContent=spec.ready ? "Đã có key" : "Thiếu API key"; node.classList.toggle("missing",!spec.ready); }
    setBusy(false);
    const last = localStorage.getItem("image-doctor-last-free-run") || localStorage.getItem("image-doctor-last-run");
    if (last) {
      try {
        const job=await api(`/api/runs/${last}`);
        if(isFreeRun(job)) $("system-prompt").value=job.system_prompt;
        $("max-output").value=String(job.max_output_tokens); $("max-edge").value=String(job.max_edge || 1280); $("preview").src=`/api/runs/${last}/image`; $("preview").hidden=false; $("image-empty").hidden=true;
        $("zoom").hidden=false; $("large-image").src=$("preview").src; $("image-info").textContent=`${job.filename} · Lượt đã lưu; chọn lại file để chạy lượt mới`;
        render(job); if(job.status === "running") poll(last);
      } catch {localStorage.removeItem("image-doctor-last-free-run");localStorage.removeItem("image-doctor-last-run");}
    }
  } catch(e) {notice(`Không tải được cấu hình: ${e.message}. Kiểm tra server và tải lại trang.`);}
}
init();
