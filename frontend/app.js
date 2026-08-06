const API = "";

/** Stand-in UI is group-only. 1:1 remains available via the API / eval battery. */
const mode = "group";

/** @type {1 | 2} */
let setupStep = 1;

/** @type {"setup" | "session"} */
let view = "setup";

// --- 1:1 state ---
let sessionId = null;
let profiles = [];
let messages = [];
let lastProfileDetail = null;

// --- Group state ---
let groupSessionId = null;
let groupTranscript = [];
let groupDisplayNames = {};
let groupStudents = [];
let lastGroupMeta = null;

/** AbortController for in-flight child-speech reveal (client-side pacing). */
let speechAbortController = null;
/** Increments on each new speech generation; stale loops must stop. */
let speechGeneration = 0;

// DOM
const setupView = document.getElementById("setup-view");
const sessionView = document.getElementById("session-view");
const sessionChrome = document.getElementById("session-chrome");
const sessionSummary = document.getElementById("session-summary");
const endSessionBtn = document.getElementById("end-session-btn");

const stepConfigure = document.getElementById("step-configure");
const stepReady = document.getElementById("step-ready");
const stepperItems = document.querySelectorAll(".stepper-item");

const prepareBtn = document.getElementById("prepare-btn");
const backToSetupBtn = document.getElementById("back-to-setup-btn");
const startPracticeBtn = document.getElementById("start-practice-btn");

const configGroup = document.getElementById("config-group");
const stepConfigLede = document.getElementById("step-config-lede");
const readySummary = document.getElementById("ready-summary");
const readyGroupExtras = document.getElementById("ready-group-extras");

const profileSelect = document.getElementById("profile-select");
const profileInfo = document.getElementById("profile-info");
const taskInput = document.getElementById("task-input");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const statusEl = document.getElementById("status");
const kgList = document.getElementById("kg-list");
const activeList = document.getElementById("active-list");
const playbookList = document.getElementById("playbook-list");
const constructList = document.getElementById("construct-list");
const pisaList = document.getElementById("pisa-list");
const behaviorInfo = document.getElementById("behavior-info");

const sidebar1to1 = document.getElementById("sidebar-1to1");
const sidebarGroup = document.getElementById("sidebar-group");

const groupPreset = document.getElementById("group-preset");
const groupCustomPicks = document.getElementById("group-custom-picks");
const groupP1 = document.getElementById("group-p1");
const groupP2 = document.getElementById("group-p2");
const groupP3 = document.getElementById("group-p3");
const groupTaskSelect = document.getElementById("group-task-select");
const groupTaskCustomWrap = document.getElementById("group-task-custom-wrap");
const groupTaskInput = document.getElementById("group-task-input");
const groupExportBtn = document.getElementById("group-export-btn");
const groupResetBtn = document.getElementById("group-reset-btn");
const groupAdvanceBtn = document.getElementById("group-advance-btn");
const groupRoster = document.getElementById("group-roster");
const copyDemoBtn = document.getElementById("copy-demo-btn");

const TRAIT_LABELS = {
  Openness: "Curiosity",
  Conscientiousness: "Organization",
  Extraversion: "Outgoingness",
  Agreeableness: "Cooperativeness",
  Neuroticism: "Sensitivity to stress",
};

/** Plain-language chip labels for High / Low Big Five levels. */
const TRAIT_TAG_HIGH = {
  Openness: "Curious",
  Conscientiousness: "Organized",
  Extraversion: "Outgoing",
  Agreeableness: "Cooperative",
  Neuroticism: "Stress-sensitive",
};

const TRAIT_TAG_LOW = {
  Openness: "Concrete",
  Conscientiousness: "Casual",
  Extraversion: "Quiet",
  Agreeableness: "Direct",
  Neuroticism: "Steady",
};

/** Classroom-salient order — show up to 3 distinctive tags per student. */
const TRAIT_TAG_PRIORITY = [
  "Extraversion",
  "Conscientiousness",
  "Neuroticism",
  "Openness",
  "Agreeableness",
];

const TASK_LABELS = {
  phone_plans_linear_01: "Phone plans — compare linear costs",
  frac_compare_01: "Compare fractions — which is larger, 1/3 or 1/4?",
  frac_add_common_01: "Add fractions — 2/5 + 1/5",
  frac_sub_unlike_01: "Subtract fractions with different denominators",
  frac_word_maria_pizza_01: "Word problem — Maria’s pizza",
  ratio_identify_01: "Identify a ratio",
};

function formatApiError(err) {
  if (err == null) return "Request failed";
  if (typeof err === "string") return err;
  if (Array.isArray(err)) {
    return err
      .map((e) => (typeof e === "object" ? e.msg || JSON.stringify(e) : String(e)))
      .join("; ");
  }
  if (typeof err === "object" && err.msg) return err.msg;
  return JSON.stringify(err);
}

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(formatApiError(err.detail) || res.statusText);
  }
  return res.json();
}

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.classList.toggle("warning", kind === "warning");
  statusEl.classList.toggle("error", kind === "error");
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeAttribute(text) {
  return escapeHtml(text).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function capitalize(id) {
  if (!id) return "";
  return id.charAt(0).toUpperCase() + id.slice(1);
}

/**
 * Pick 1–3 distinctive personality tags from a High/Low Big Five map.
 * @param {Record<string, string> | null | undefined} personality
 * @param {number} [max=3]
 * @returns {{ key: string, level: string, label: string }[]}
 */
function personalityTags(personality, max = 3) {
  if (!personality || typeof personality !== "object") return [];
  const tags = [];
  for (const key of TRAIT_TAG_PRIORITY) {
    const level = personality[key];
    if (level !== "High" && level !== "Low") continue;
    const label =
      level === "High" ? TRAIT_TAG_HIGH[key] : TRAIT_TAG_LOW[key];
    if (!label) continue;
    tags.push({ key, level, label });
    if (tags.length >= max) break;
  }
  return tags;
}

function personalityTagsHtml(personality, max = 3) {
  const tags = personalityTags(personality, max);
  if (tags.length === 0) return "";
  const chips = tags
    .map((t) => {
      const full = `${TRAIT_LABELS[t.key] || t.key}: ${t.level}`;
      return `<span class="persona-tag persona-${t.level.toLowerCase()}" title="${escapeAttribute(
        full
      )}">${escapeHtml(t.label)}</span>`;
    })
    .join("");
  return `<span class="persona-tags" aria-label="Personality">${chips}</span>`;
}

/** Optional chip when a student asked the teacher for help. */
function helpSeekingTagHtml(helpSeeking) {
  if (!helpSeeking) return "";
  return `<span class="persona-tag persona-help-seek" title="Asked the teacher a clarifying question">Asked a question</span>`;
}

/** Personality chips plus optional help-seeking badge. */
function studentMetaTagsHtml(personality, helpSeeking = false, max = 3) {
  const chips = [];
  for (const t of personalityTags(personality, max)) {
    const full = `${TRAIT_LABELS[t.key] || t.key}: ${t.level}`;
    chips.push(
      `<span class="persona-tag persona-${t.level.toLowerCase()}" title="${escapeAttribute(
        full
      )}">${escapeHtml(t.label)}</span>`
    );
  }
  const ask = helpSeekingTagHtml(helpSeeking);
  if (ask) chips.push(ask);
  if (chips.length === 0) return "";
  return `<span class="persona-tags" aria-label="Message tags">${chips.join(
    ""
  )}</span>`;
}

/** Resolve personality for a profile id from group slot or profiles cache. */
function personalityForProfile(profileId) {
  if (!profileId) return null;
  const fromGroup = groupStudents.find((s) => s.profile_id === profileId);
  if (fromGroup?.personality) return fromGroup.personality;
  const fromCache = profiles.find((p) => p.id === profileId);
  return fromCache?.personality || null;
}

function clearChatEmpty() {
  const empty = document.getElementById("chat-empty");
  if (empty) empty.remove();
}

function showChatEmpty() {
  cancelSpeechStreams();
  chatLog.innerHTML = `
    <div class="chat-empty" id="chat-empty">
      <p class="chat-empty-title">Conversation will appear here</p>
      <p class="chat-empty-desc">Start practice from the setup steps to begin teaching.</p>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Child-speech reveal (client-side pacing of completed API replies)
// Backend returns full message text; we animate display at speaking rhythm.
// ---------------------------------------------------------------------------

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function cancelSpeechStreams() {
  if (speechAbortController) {
    speechAbortController.abort();
    speechAbortController = null;
  }
  speechGeneration += 1;
  removeTypingIndicator();
  chatLog
    .querySelectorAll(".bubble.is-speaking")
    .forEach((el) => el.classList.remove("is-speaking"));
}

/** Start a new speech generation; cancels any prior reveal. @returns {{ signal: AbortSignal, gen: number }} */
function beginSpeechGeneration() {
  cancelSpeechStreams();
  speechAbortController = new AbortController();
  return { signal: speechAbortController.signal, gen: speechGeneration };
}

function isSpeechCurrent(gen) {
  return gen === speechGeneration;
}

function scrollChatToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

/**
 * Delay (ms) after revealing `ch`, tuned for ~child speaking pace.
 * Base ~38–48ms/char with longer pauses on punctuation.
 */
function speechDelayAfterChar(ch, index) {
  let ms = 40 + (index % 5 === 0 ? 6 : 0);
  if (ch === "\n") return ms + 200;
  if (".?!".includes(ch)) return ms + 300;
  if (",;:".includes(ch)) return ms + 140;
  if (ch === " " && index > 0 && index % 11 === 0) return ms + 45;
  if (ch === "-" || ch === "—") return ms + 80;
  return ms;
}

/** Scale delays so long replies stay under maxMs without feeling robotic. */
function planSpeechDelays(text) {
  const delays = [];
  for (let i = 0; i < text.length; i++) {
    delays.push(speechDelayAfterChar(text[i], i));
  }
  const total = delays.reduce((a, b) => a + b, 0);
  const maxMs = 12000;
  if (total <= maxMs || total === 0) return delays;
  const scale = maxMs / total;
  return delays.map((d) => Math.max(18, Math.round(d * scale)));
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const id = setTimeout(resolve, ms);
    const onAbort = () => {
      clearTimeout(id);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Progressively reveal `fullText` into `textEl` at child-speech pace.
 * On abort or reduced-motion, snaps to full text.
 */
async function streamTextInto(textEl, fullText, signal, bubbleEl) {
  const text = fullText == null ? "" : String(fullText);
  if (!textEl) return;

  if (prefersReducedMotion() || text.length === 0) {
    textEl.textContent = text;
    bubbleEl?.classList.remove("is-speaking");
    scrollChatToBottom();
    return;
  }

  bubbleEl?.classList.add("is-speaking");
  textEl.textContent = "";
  const delays = planSpeechDelays(text);

  try {
    for (let i = 0; i < text.length; i++) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
      textEl.textContent = text.slice(0, i + 1);
      if (i % 4 === 0 || "\n.?!".includes(text[i])) scrollChatToBottom();
      await sleep(delays[i], signal);
    }
    scrollChatToBottom();
  } catch (err) {
    textEl.textContent = text;
    scrollChatToBottom();
    if (err?.name !== "AbortError") throw err;
  } finally {
    bubbleEl?.classList.remove("is-speaking");
  }
}

function appendTeacherBubble(content) {
  clearChatEmpty();
  const div = document.createElement("div");
  div.className = "bubble teacher";
  div.innerHTML = `<span class="meta">You (teacher)</span><span class="bubble-text">${escapeHtml(
    content
  )}</span>`;
  chatLog.appendChild(div);
  scrollChatToBottom();
  return div;
}

/**
 * @param {{ name: string, tagsHtml?: string, speakerClass?: string, content?: string }} opts
 * @returns {{ bubble: HTMLElement, textEl: HTMLElement }}
 */
function appendStudentBubble(opts) {
  clearChatEmpty();
  const { name, tagsHtml = "", speakerClass = "", content = "" } = opts;
  const div = document.createElement("div");
  div.className = `bubble student${speakerClass ? ` ${speakerClass}` : ""}`;
  div.innerHTML = `<span class="meta"><span class="meta-name">${escapeHtml(
    name
  )}</span>${tagsHtml}</span><span class="bubble-body"><span class="bubble-text">${escapeHtml(
    content
  )}</span><span class="speech-caret" aria-hidden="true"></span></span>`;
  chatLog.appendChild(div);
  scrollChatToBottom();
  return { bubble: div, textEl: div.querySelector(".bubble-text") };
}

function removeTypingIndicator() {
  chatLog.querySelectorAll(".bubble.typing").forEach((el) => el.remove());
}

/**
 * Show a thinking/typing bubble while waiting for the API.
 * @param {{ name?: string, tagsHtml?: string, speakerClass?: string, label?: string }} [opts]
 */
function showTypingIndicator(opts = {}) {
  removeTypingIndicator();
  clearChatEmpty();
  const name = opts.name || "Student";
  const tagsHtml = opts.tagsHtml || "";
  const speakerClass = opts.speakerClass || "";
  const label = opts.label || `${name} is thinking`;
  const div = document.createElement("div");
  div.className = `bubble student typing${speakerClass ? ` ${speakerClass}` : ""}`;
  div.setAttribute("aria-live", "polite");
  div.setAttribute("aria-label", label);
  div.innerHTML = `<span class="meta"><span class="meta-name">${escapeHtml(
    name
  )}</span>${tagsHtml}</span><span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>`;
  chatLog.appendChild(div);
  scrollChatToBottom();
  return div;
}

function oneToOneStudentMeta(helpSeeking = false) {
  const name =
    lastProfileDetail?.name || capitalize(profileSelect.value) || "Student";
  const tagsHtml = studentMetaTagsHtml(
    lastProfileDetail?.personality ||
      personalityForProfile(profileSelect.value),
    helpSeeking
  );
  return { name, tagsHtml };
}

/**
 * Reveal 1:1 messages: teacher instant, students streamed in order.
 * Used on session start (and as a full replay helper).
 */
async function revealMessages1to1(msgs, signal, gen) {
  clearChatEmpty();
  chatLog.innerHTML = "";
  for (const msg of msgs) {
    if (!isSpeechCurrent(gen) || signal?.aborted) return;
    if (msg.role === "teacher") {
      appendTeacherBubble(msg.content);
    } else {
      const meta = oneToOneStudentMeta(!!msg.help_seeking);
      const { bubble, textEl } = appendStudentBubble({
        name: meta.name,
        tagsHtml: meta.tagsHtml,
      });
      await streamTextInto(textEl, msg.content, signal, bubble);
    }
  }
}

/**
 * Reveal group transcript entries; stream only student lines.
 * When `streamFromIndex` is set, entries before it render instantly.
 */
async function revealGroupEntries(transcript, signal, gen, streamFromIndex = 0) {
  clearChatEmpty();
  chatLog.innerHTML = "";
  for (let i = 0; i < transcript.length; i++) {
    if (!isSpeechCurrent(gen) || signal?.aborted) return;
    const entry = transcript[i];
    const isTeacher =
      entry.speaker_type === "teacher" || entry.speaker_id === "teacher";
    if (isTeacher) {
      const div = document.createElement("div");
      div.className = "bubble teacher";
      div.innerHTML = `<span class="meta">${escapeHtml(
        speakerLabel(entry)
      )}</span><span class="bubble-text">${escapeHtml(entry.content)}</span>`;
      chatLog.appendChild(div);
      scrollChatToBottom();
      continue;
    }
    const sid = entry.speaker_id || "student";
    const { bubble, textEl } = appendStudentBubble({
      name: speakerLabel(entry),
      tagsHtml: studentMetaTagsHtml(
        personalityForProfile(sid),
        !!entry.help_seeking
      ),
      speakerClass: `speaker-${sid}`,
      content: i < streamFromIndex ? entry.content : "",
    });
    if (i < streamFromIndex) {
      continue;
    }
    await streamTextInto(textEl, entry.content, signal, bubble);
  }
}

/** Stream only the new student replies after an optimistic teacher bubble. */
async function streamGroupReplies(replies, signal, gen) {
  for (const reply of replies || []) {
    if (!isSpeechCurrent(gen) || signal?.aborted) return;
    const sid = reply.speaker_id || "student";
    const { bubble, textEl } = appendStudentBubble({
      name: groupDisplayNames[sid] || capitalize(sid),
      tagsHtml: studentMetaTagsHtml(
        personalityForProfile(sid),
        !!reply.help_seeking
      ),
      speakerClass: `speaker-${sid}`,
    });
    await streamTextInto(textEl, reply.content, signal, bubble);
  }
}

// ---------------------------------------------------------------------------
// Wizard / view navigation
// ---------------------------------------------------------------------------

function setSetupStep(step) {
  setupStep = step;
  stepConfigure.classList.toggle("hidden", step !== 1);
  stepReady.classList.toggle("hidden", step !== 2);

  stepperItems.forEach((item) => {
    const n = Number(item.dataset.step);
    item.classList.toggle("is-current", n === step);
    item.classList.toggle("is-done", n < step);
    if (n === step) {
      item.setAttribute("aria-current", "step");
    } else {
      item.removeAttribute("aria-current");
    }
  });

  if (step === 1) {
    syncConfigPanels();
  }
  if (step === 2) {
    renderReadySummary();
  }
}

function showSetup() {
  view = "setup";
  setupView.classList.remove("hidden");
  sessionView.classList.add("hidden");
  sessionChrome.classList.add("hidden");
  messageInput.disabled = true;
  sendBtn.disabled = true;
}

function showSession() {
  view = "session";
  setupView.classList.add("hidden");
  sessionView.classList.remove("hidden");
  sessionChrome.classList.remove("hidden");
  sidebarGroup.classList.remove("hidden");
  updateSessionChrome();
}

function syncConfigPanels() {
  configGroup.classList.remove("hidden");
  stepConfigLede.textContent =
    "Choose three students and the math problem they’ll discuss together.";
}

function updateSessionChrome() {
  if (mode === "1to1") {
    const name = lastProfileDetail?.name || capitalize(profileSelect.value);
    sessionSummary.textContent = `One-on-one · ${name}`;
  } else {
    const names =
      (groupStudents.length
        ? groupStudents.map((s) => s.display_name || capitalize(s.profile_id))
        : selectedProfileIds().map(capitalize)
      ).join(", ");
    sessionSummary.textContent = `Group · ${names}`;
  }
}

function taskLabelForSelect() {
  const id = groupTaskSelect.value;
  if (id === "custom") {
    return groupTaskInput.value.trim() || "Custom problem";
  }
  return TASK_LABELS[id] || id;
}

function renderReadySummary() {
  readyGroupExtras.classList.toggle("hidden", mode !== "group");

  if (mode === "1to1") {
    const name = lastProfileDetail?.name || capitalize(profileSelect.value);
    const desc = lastProfileDetail?.description || "";
    const problem = taskInput.value.trim() || "Suggested problem for this student";
    readySummary.innerHTML = `
      <h2>Session overview</h2>
      <dl class="ready-meta">
        <dt>Mode</dt>
        <dd>One-on-one tutoring</dd>
        <dt>Student</dt>
        <dd>
          <div class="ready-student">
            <strong>${escapeHtml(name)}</strong>
            ${personalityTagsHtml(lastProfileDetail?.personality)}
          </div>
          ${desc ? `<p class="ready-student-desc">${escapeHtml(desc)}</p>` : ""}
        </dd>
        <dt>Math problem</dt>
        <dd>${escapeHtml(problem)}</dd>
      </dl>
    `;
  } else {
    const studentLine = selectedProfileIds()
      .map((id) => {
        const p = profiles.find((x) => x.id === id);
        const name = p?.name || capitalize(id);
        return `<span class="ready-student-chip"><strong>${escapeHtml(
          name
        )}</strong>${personalityTagsHtml(p?.personality)}</span>`;
      })
      .join("");
    const preset =
      groupPreset.value === "demo"
        ? "Phone plans demo (recommended)"
        : groupPreset.value === "custom"
          ? "Custom group"
          : "Preset group";
    readySummary.innerHTML = `
      <h2>Session overview</h2>
      <dl class="ready-meta">
        <dt>Mode</dt>
        <dd>Small-group discussion (you + 3 students)</dd>
        <dt>Group</dt>
        <dd>
          <div class="ready-group-students">${studentLine}</div>
          <p class="ready-student-desc">${escapeHtml(preset)}</p>
        </dd>
        <dt>Math problem</dt>
        <dd>${escapeHtml(taskLabelForSelect())}</dd>
      </dl>
    `;
  }

  const ready =
    (mode === "1to1" && !!sessionId) || (mode === "group" && !!groupSessionId);
  startPracticeBtn.disabled = !ready;
}

// ---------------------------------------------------------------------------
// Shared chat rendering
// ---------------------------------------------------------------------------

function renderMessages1to1(msgs) {
  cancelSpeechStreams();
  clearChatEmpty();
  chatLog.innerHTML = "";
  for (const msg of msgs) {
    if (msg.role === "teacher") {
      appendTeacherBubble(msg.content);
    } else {
      const meta = oneToOneStudentMeta(!!msg.help_seeking);
      appendStudentBubble({
        name: meta.name,
        tagsHtml: meta.tagsHtml,
        content: msg.content,
      });
    }
  }
  scrollChatToBottom();
}

function speakerLabel(entry) {
  if (entry.speaker_type === "teacher" || entry.speaker_id === "teacher") {
    return "You (teacher)";
  }
  const id = entry.speaker_id || "student";
  return groupDisplayNames[id] || capitalize(id);
}

function renderGroupTranscript(transcript) {
  cancelSpeechStreams();
  clearChatEmpty();
  chatLog.innerHTML = "";
  for (const entry of transcript) {
    const isTeacher =
      entry.speaker_type === "teacher" || entry.speaker_id === "teacher";
    if (isTeacher) {
      const div = document.createElement("div");
      div.className = "bubble teacher";
      div.innerHTML = `<span class="meta">${escapeHtml(
        speakerLabel(entry)
      )}</span><span class="bubble-text">${escapeHtml(entry.content)}</span>`;
      chatLog.appendChild(div);
    } else {
      const sid = entry.speaker_id || "student";
      appendStudentBubble({
        name: speakerLabel(entry),
        tagsHtml: studentMetaTagsHtml(
          personalityForProfile(sid),
          !!entry.help_seeking
        ),
        speakerClass: `speaker-${sid}`,
        content: entry.content,
      });
    }
  }
  scrollChatToBottom();
}

function renderConceptList(ul, concepts) {
  ul.innerHTML = "";
  if (!concepts || concepts.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nothing tagged yet.";
    ul.appendChild(li);
    return;
  }
  for (const item of concepts) {
    const li = document.createElement("li");
    const cls = item.state.toLowerCase();
    li.className = cls;
    li.textContent = `${item.concept}: ${item.state}`;
    ul.appendChild(li);
  }
}

function renderKg(concepts) {
  renderConceptList(kgList, concepts);
}

function renderActive(concepts) {
  renderConceptList(activeList, concepts);
}

function renderConstructs(constructs) {
  constructList.innerHTML = "";
  if (!constructs || constructs.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nothing tagged yet.";
    constructList.appendChild(li);
    return;
  }
  for (const c of constructs) {
    const li = document.createElement("li");
    li.textContent = `${c.name}: level ${c.level} (${c.label})`;
    constructList.appendChild(li);
  }
}

function renderPisa(profile) {
  pisaList.innerHTML = "";
  if (!profile || profile.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nothing tagged yet.";
    pisaList.appendChild(li);
    return;
  }
  for (const a of profile) {
    const li = document.createElement("li");
    li.className = a.level;
    li.textContent = `${a.name}: ${a.level}`;
    pisaList.appendChild(li);
  }
}

function renderBehavior(bp, resolution) {
  if (!bp || (!bp.primary_construct && !bp.behavior_mode)) {
    behaviorInfo.textContent = resolution
      ? `Problem classified (${resolution.source}, ${Math.round((resolution.confidence || 0) * 100)}%): ${(resolution.constructs || []).join(", ") || "—"}`
      : "No learning focus tagged for this problem yet.";
    return;
  }
  const parts = [];
  if (resolution?.source) {
    parts.push(
      `Classified via ${resolution.source} (${Math.round((resolution.confidence || 0) * 100)}%)`
    );
  }
  if (bp.behavior_mode) parts.push(`Likely stance: ${bp.behavior_mode}`);
  if (bp.primary_construct) parts.push(`Focus idea: ${bp.primary_construct}`);
  if (bp.student_stack_level != null && bp.target_stack_level != null) {
    parts.push(`Mastery level ${bp.student_stack_level} → aiming for ${bp.target_stack_level}`);
  }
  parts.push(`Correctness tendency: ${bp.likely_correctness}`);
  if (bp.error_types?.length) {
    parts.push(`Common errors: ${bp.error_types.join("; ")}`);
  }
  if (bp.weak_attributes?.length) {
    parts.push(`Weaker skill areas: ${bp.weak_attributes.join(", ")}`);
  }
  behaviorInfo.textContent = parts.join(" · ");
}

function renderPlaybook(playbook) {
  playbookList.innerHTML = "";
  const patterns = playbook?.patterns || [];
  if (patterns.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No predicted mistake patterns for this problem.";
    playbookList.appendChild(li);
    return;
  }
  for (const p of patterns) {
    const li = document.createElement("li");
    li.textContent = `${p.concept} (${p.type}): ${p.example}`;
    playbookList.appendChild(li);
  }
}

function renderProfileDetail(detail) {
  lastProfileDetail = detail;
  const p = detail.personality;
  const tagsHtml = personalityTagsHtml(p);
  const traits = Object.entries(p || {})
    .map(([key, val]) => {
      const label = TRAIT_LABELS[key] || key;
      return `<li><strong>${escapeHtml(label)}</strong> ${escapeHtml(String(val))}</li>`;
    })
    .join("");

  profileInfo.innerHTML = `
    <p class="preview-name">${escapeHtml(detail.name)}${
      tagsHtml ? ` ${tagsHtml}` : ""
    }</p>
    <p class="preview-desc">${escapeHtml(detail.description)}</p>
    <ul class="preview-traits">${traits}</ul>
  `;
  if (!taskInput.value.trim()) {
    taskInput.value = detail.default_task;
  }
  renderKg(detail.kg_concepts);
  renderConstructs(detail.construct_levels);
  renderPisa(detail.pisa_profile);
}

function renderGroupRoster(students, activity = null) {
  if (!students || students.length === 0) {
    groupRoster.textContent = "No students yet.";
    return;
  }
  groupStudents = students;
  groupRoster.innerHTML = "";
  groupDisplayNames = {};
  const replied = new Set(activity?.replied || []);
  const observers = new Set(activity?.observers || []);
  const reasons = new Map(
    (activity?.decisions || []).map((decision) => [
      decision.profile_id,
      decision.reason,
    ])
  );
  for (const s of students) {
    groupDisplayNames[s.profile_id] = s.display_name || capitalize(s.profile_id);
    const bp = s.behavior_profile || {};
    const card = document.createElement("div");
    card.className = `roster-card speaker-${s.profile_id}`;
    const modeLabel = bp.behavior_mode || s.predicted_behavior || "—";
    const construct = bp.primary_construct || "—";
    const state = replied.has(s.profile_id)
      ? "Speaking"
      : observers.has(s.profile_id)
        ? "Listening"
        : activity
          ? "Waiting"
          : "Ready";
    const stateClass = state.toLowerCase();
    const reason = reasons.get(s.profile_id) || "";
    const tagsHtml = personalityTagsHtml(
      s.personality || personalityForProfile(s.profile_id)
    );
    card.innerHTML = `
      <div class="roster-heading">
        <strong>${escapeHtml(groupDisplayNames[s.profile_id])}</strong>
        <span class="roster-state ${stateClass}"${
          reason ? ` title="${escapeAttribute(reason)}"` : ""
        }>${state}</span>
      </div>
      ${tagsHtml ? `<div class="roster-tags">${tagsHtml}</div>` : ""}
      <span class="roster-detail">Stance: ${escapeHtml(modeLabel)}</span>
      <span class="roster-detail">Focus idea: ${escapeHtml(construct)}</span>
      ${
        typeof s.receptivity === "number"
          ? `<span class="roster-detail">Scaffold uptake: ${Math.round(
              s.receptivity * 100
            )}%</span>`
          : ""
      }
    `;
    groupRoster.appendChild(card);
  }
  updateSessionChrome();
}

// ---------------------------------------------------------------------------
// 1:1 flows
// ---------------------------------------------------------------------------

async function loadProfiles() {
  profiles = await api("/api/profiles");
  // Prefer Jordan first — recommended demo student for newcomers
  profiles = [...profiles].sort((a, b) => {
    if (a.id === "jordan") return -1;
    if (b.id === "jordan") return 1;
    return a.name.localeCompare(b.name);
  });
  if (profileSelect) {
    profileSelect.innerHTML = profiles
      .map((p) => `<option value="${p.id}">${p.name} - ${p.description}</option>`)
      .join("");
    profileSelect.value = profiles.some((p) => p.id === "jordan")
      ? "jordan"
      : profiles[0]?.id;
  }

  const opts = profiles
    .map((p) => `<option value="${p.id}">${p.name}</option>`)
    .join("");
  groupP1.innerHTML = opts;
  groupP2.innerHTML = opts;
  groupP3.innerHTML = opts;
  if (profiles.some((p) => p.id === "jordan")) {
    groupP1.value = "jordan";
  }
  if (profiles.some((p) => p.id === "sam")) {
    groupP2.value = "sam";
  }
  if (profiles.some((p) => p.id === "alex")) {
    groupP3.value = "alex";
  }

  if (profileSelect) {
    await onProfileChange();
  }
}

async function onProfileChange() {
  const id = profileSelect.value;
  const detail = await api(`/api/profiles/${id}`);
  renderProfileDetail(detail);
}

async function createSession() {
  setStatus("Preparing session (matching the problem to learning ideas)…");
  prepareBtn.disabled = true;
  startPracticeBtn.disabled = true;
  sendBtn.disabled = true;
  messageInput.disabled = true;
  showChatEmpty();
  messages = [];

  const body = {
    profile_id: profileSelect.value,
    task_text: taskInput.value.trim() || undefined,
  };

  try {
    const data = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify(body),
    });

    sessionId = data.session_id;
    renderKg(data.kg_concepts);
    renderConstructs(data.construct_levels);
    renderPisa(data.pisa_profile);
    renderBehavior(data.behavior_profile, data.task_resolution);
    renderPlaybook(data.error_playbook);
    renderActive([]);
    setStatus(`Session ready for ${data.student_id}. Click Start practice.`);
    setSetupStep(2);
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    prepareBtn.disabled = false;
  }
}

async function startSession() {
  if (!sessionId) return;
  setStatus("Starting conversation…");
  startPracticeBtn.disabled = true;
  const { signal, gen } = beginSpeechGeneration();
  const meta = oneToOneStudentMeta();
  showSession();
  clearChatEmpty();
  chatLog.innerHTML = "";
  showTypingIndicator({
    name: meta.name,
    tagsHtml: meta.tagsHtml,
    label: `${meta.name} is thinking`,
  });

  try {
    const data = await api(`/api/sessions/${sessionId}/start`, { method: "POST" });
    if (!isSpeechCurrent(gen) || signal.aborted) return;
    messages = data.messages || [];
    removeTypingIndicator();
    renderActive(data.active_concepts || []);
    messageInput.placeholder = "Type your teaching response…";
    setStatus(
      data.vague_warning
        ? "That sounded like a short confirmation — the student may not move forward on their own."
        : `${meta.name} is speaking…`
    );
    await revealMessages1to1(messages, signal, gen);
    if (!isSpeechCurrent(gen) || signal.aborted) return;
    if (data.vague_warning) {
      setStatus(
        "That sounded like a short confirmation — the student may not move forward on their own.",
        "warning"
      );
    } else {
      setStatus(`Turn ${data.turn}. Type your next teaching move.`);
    }
    messageInput.disabled = false;
    sendBtn.disabled = false;
    messageInput.focus();
  } catch (err) {
    if (err?.name === "AbortError") return;
    removeTypingIndicator();
    setStatus(`Error: ${err.message}`, "error");
    startPracticeBtn.disabled = false;
  }
}

async function sendMessage1to1(text) {
  if (!sessionId || !text.trim()) return;
  const trimmed = text.trim();
  // Snap any in-progress speech to committed history before a new turn
  if (messages.length) renderMessages1to1(messages);
  const { signal, gen } = beginSpeechGeneration();
  const meta = oneToOneStudentMeta();
  const historyLen = messages.length;

  sendBtn.disabled = true;
  messageInput.disabled = true;
  messageInput.value = "";
  appendTeacherBubble(trimmed);
  showTypingIndicator({
    name: meta.name,
    tagsHtml: meta.tagsHtml,
    label: `${meta.name} is thinking`,
  });
  setStatus(`${meta.name} is thinking…`);

  try {
    const data = await api(`/api/sessions/${sessionId}/message`, {
      method: "POST",
      body: JSON.stringify({ message: trimmed }),
    });
    if (!isSpeechCurrent(gen) || signal.aborted) return;

    removeTypingIndicator();
    messages = messages.slice(0, historyLen).concat(data.messages || []);
    renderActive(data.active_concepts || []);

    const studentMsgs = (data.messages || []).filter((m) => m.role !== "teacher");
    setStatus(
      studentMsgs.length
        ? `${meta.name} is speaking…`
        : `Turn ${data.turn}. Continue the tutoring dialogue.`
    );

    messageInput.disabled = false;
    sendBtn.disabled = false;
    messageInput.focus();

    for (const msg of studentMsgs) {
      if (!isSpeechCurrent(gen) || signal.aborted) return;
      const speakMeta = oneToOneStudentMeta(!!msg.help_seeking);
      const { bubble, textEl } = appendStudentBubble({
        name: speakMeta.name,
        tagsHtml: speakMeta.tagsHtml,
      });
      await streamTextInto(textEl, msg.content, signal, bubble);
    }
    if (!isSpeechCurrent(gen) || signal.aborted) return;

    if (data.vague_warning) {
      setStatus(
        "That sounded like a short confirmation — the student may not move forward on their own.",
        "warning"
      );
    } else {
      setStatus(`Turn ${data.turn}. Continue the tutoring dialogue.`);
    }
  } catch (err) {
    if (err?.name === "AbortError") return;
    removeTypingIndicator();
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    if (isSpeechCurrent(gen) && !signal.aborted && sessionId) {
      messageInput.disabled = false;
      sendBtn.disabled = false;
    }
  }
}

async function endOneToOneSession() {
  cancelSpeechStreams();
  if (sessionId) {
    try {
      await api(`/api/sessions/${sessionId}`, { method: "DELETE" });
    } catch (_) {
      /* ignore */
    }
  }
  sessionId = null;
  messages = [];
  showChatEmpty();
  renderActive([]);
}

// ---------------------------------------------------------------------------
// Group flows
// ---------------------------------------------------------------------------

function selectedProfileIds() {
  const preset = groupPreset.value;
  if (preset === "demo") {
    return ["jordan", "sam", "alex"];
  }
  if (preset === "custom") {
    return [groupP1.value, groupP2.value, groupP3.value];
  }
  return preset.split(",");
}

function syncGroupPresetUI() {
  groupCustomPicks.classList.toggle("hidden", groupPreset.value !== "custom");
  if (groupPreset.value === "demo") {
    groupTaskSelect.value = "phone_plans_linear_01";
    syncGroupTaskUI();
  }
}

function syncGroupTaskUI() {
  groupTaskCustomWrap.classList.toggle(
    "hidden",
    groupTaskSelect.value !== "custom"
  );
}

async function createGroupSession() {
  const profile_ids = selectedProfileIds();
  const unique = new Set(profile_ids);
  if (profile_ids.length !== 3 || unique.size !== 3) {
    setStatus("Pick exactly 3 different students.", "error");
    return;
  }

  setStatus("Preparing group session…");
  prepareBtn.disabled = true;
  startPracticeBtn.disabled = true;
  groupExportBtn.disabled = true;
  groupResetBtn.disabled = true;
  if (groupAdvanceBtn) groupAdvanceBtn.disabled = true;
  sendBtn.disabled = true;
  messageInput.disabled = true;
  showChatEmpty();
  groupTranscript = [];

  const body = {
    profile_ids,
    orchestration: "teacher_directed_self_select",
    enable_self_select: true,
  };
  if (groupTaskSelect.value === "custom") {
    const text = groupTaskInput.value.trim();
    if (!text) {
      setStatus("Enter your own problem text, or pick a prepared one.", "error");
      prepareBtn.disabled = false;
      return;
    }
    body.task_text = text;
  } else {
    body.task_id = groupTaskSelect.value;
  }

  try {
    const data = await api("/api/group-sessions", {
      method: "POST",
      body: JSON.stringify(body),
    });
    groupSessionId = data.session_id;
    lastGroupMeta = data;
    renderGroupRoster(data.students);
    const names = (data.profile_ids || []).map(capitalize).join(", ");
    setStatus(`Group ready: ${names}. Click Start practice.`);
    groupResetBtn.disabled = false;
    setSetupStep(2);
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    prepareBtn.disabled = false;
  }
}

async function startGroupSession() {
  if (!groupSessionId) return;
  setStatus("Starting group session (first student may reply)…");
  startPracticeBtn.disabled = true;
  const { signal, gen } = beginSpeechGeneration();
  showSession();
  clearChatEmpty();
  chatLog.innerHTML = "";
  showTypingIndicator({
    name: "Students",
    label: "Students are thinking",
  });

  try {
    const data = await api(`/api/group-sessions/${groupSessionId}/start`, {
      method: "POST",
    });
    if (!isSpeechCurrent(gen) || signal.aborted) return;

    groupTranscript = data.transcript || [];
    removeTypingIndicator();
    const replyIds = (data.replies || []).map((reply) => reply.speaker_id);
    renderGroupRoster(data.students, {
      replied: replyIds,
      observers: data.observers || [],
      decisions: data.speak_decisions || [],
    });

    const replyCount = (data.replies || []).length;
    const streamFrom = Math.max(0, groupTranscript.length - replyCount);
    const firstReply = (data.replies || [])[0];
    const speakingName = firstReply
      ? groupDisplayNames[firstReply.speaker_id] ||
        capitalize(firstReply.speaker_id)
      : null;
    setStatus(
      speakingName ? `${speakingName} is speaking…` : `Turn ${data.turn}.`
    );

    await revealGroupEntries(groupTranscript, signal, gen, streamFrom);
    if (!isSpeechCurrent(gen) || signal.aborted) return;

    const replyNames = (data.replies || [])
      .map((r) => groupDisplayNames[r.speaker_id] || capitalize(r.speaker_id))
      .join(", ");
    setStatus(
      `Turn ${data.turn}. Replied: ${replyNames || "—"}. Address a student or open the floor.`
    );
    messageInput.placeholder =
      'Address a student by name, or say "Discuss together"...';
    messageInput.disabled = false;
    sendBtn.disabled = false;
    groupExportBtn.disabled = false;
    groupResetBtn.disabled = false;
    if (groupAdvanceBtn) groupAdvanceBtn.disabled = false;
    messageInput.focus();
  } catch (err) {
    if (err?.name === "AbortError") return;
    removeTypingIndicator();
    setStatus(`Error: ${err.message}`, "error");
    startPracticeBtn.disabled = false;
  }
}

async function sendGroupMessage(text) {
  if (!groupSessionId || !text.trim()) return;
  const trimmed = text.trim();
  // Snap any in-progress speech to committed transcript before a new turn
  if (groupTranscript.length) renderGroupTranscript(groupTranscript);
  const { signal, gen } = beginSpeechGeneration();

  sendBtn.disabled = true;
  messageInput.disabled = true;
  messageInput.value = "";
  appendTeacherBubble(trimmed);
  showTypingIndicator({
    name: "Students",
    label: "Students are thinking",
  });
  setStatus(
    "Students are thinking (this can take a bit if more than one speaks)…"
  );

  try {
    const data = await api(`/api/group-sessions/${groupSessionId}/message`, {
      method: "POST",
      body: JSON.stringify({ message: trimmed }),
    });
    if (!isSpeechCurrent(gen) || signal.aborted) return;

    removeTypingIndicator();
    groupTranscript = data.transcript || groupTranscript;
    const replies = data.replies || [];
    const replyIds = replies.map((r) => r.speaker_id);
    const observerIds = data.observers || [];
    renderGroupRoster(data.students, {
      replied: replyIds,
      observers: observerIds,
      decisions: data.speak_decisions || [],
    });

    if (replies.length) {
      const first = replies[0];
      const name =
        groupDisplayNames[first.speaker_id] || capitalize(first.speaker_id);
      setStatus(`${name} is speaking…`);
    }

    messageInput.disabled = false;
    sendBtn.disabled = false;
    messageInput.focus();

    await streamGroupReplies(replies, signal, gen);
    if (!isSpeechCurrent(gen) || signal.aborted) return;

    const continuation = data.multi_round_replies || [];
    if (continuation.length) {
      setStatus("Peers continuing the discussion…");
      await streamGroupReplies(continuation, signal, gen);
      if (!isSpeechCurrent(gen) || signal.aborted) return;
    }

    const allReplyIds = [
      ...replyIds,
      ...continuation.map((r) => r.speaker_id),
    ];
    renderGroupRoster(data.students, {
      replied: allReplyIds,
      observers: observerIds,
      decisions: data.speak_decisions || [],
    });

    const replyNames = allReplyIds
      .map((id) => groupDisplayNames[id] || capitalize(id))
      .join(" → ");
    const observerNames = observerIds
      .map((id) => groupDisplayNames[id] || capitalize(id))
      .join(", ");
    const listening = observerNames ? ` · Listening: ${observerNames}` : "";
    const stopBit = data.stop_reason
      ? ` · ${formatPeerStopReason(data.stop_reason)}`
      : "";
    if (data.teaching_warning) {
      setStatus(
        data.teaching_warning_message ||
          `Turn ${data.turn}. Giving the answer doesn't raise mastery — try a scaffold instead.${listening}`,
        "warning"
      );
    } else if (data.vague_warning) {
      setStatus(
        `Turn ${data.turn}. Replied: ${replyNames || "—"}${listening}${stopBit}. Your prompt may not give a clear next step.`,
        "warning"
      );
    } else {
      setStatus(
        `Turn ${data.turn}. Replied: ${replyNames || "—"}${listening}${stopBit}.`
      );
    }
    if (groupAdvanceBtn) groupAdvanceBtn.disabled = false;
  } catch (err) {
    if (err?.name === "AbortError") return;
    removeTypingIndicator();
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    if (isSpeechCurrent(gen) && !signal.aborted && groupSessionId) {
      messageInput.disabled = false;
      sendBtn.disabled = false;
    }
  }
}

function formatPeerStopReason(reason) {
  if (!reason) return "";
  if (reason === "no_volunteers") return "discussion settled (no one else spoke up)";
  if (reason === "max_rounds") return "peer-turn safety cap reached";
  if (reason === "no_eligible") return "no eligible speakers";
  return String(reason);
}

async function advanceGroupDiscussion() {
  if (!groupSessionId || !groupAdvanceBtn) return;
  if (groupTranscript.length) renderGroupTranscript(groupTranscript);
  const { signal, gen } = beginSpeechGeneration();

  groupAdvanceBtn.disabled = true;
  sendBtn.disabled = true;
  messageInput.disabled = true;
  showTypingIndicator({
    name: "Students",
    label: "Peers continuing",
  });
  setStatus("Nudging peers to continue…");

  try {
    const data = await api(`/api/group-sessions/${groupSessionId}/advance`, {
      method: "POST",
    });
    if (!isSpeechCurrent(gen) || signal.aborted) return;

    removeTypingIndicator();
    groupTranscript = data.transcript || groupTranscript;
    const replies = data.multi_round_replies || data.replies || [];
    const observerIds = data.observers || [];
    const replyIds = replies.map((r) => r.speaker_id);

    renderGroupRoster(data.students, {
      replied: replyIds,
      observers: observerIds,
      decisions: data.speak_decisions || [],
    });

    await streamGroupReplies(replies, signal, gen);
    if (!isSpeechCurrent(gen) || signal.aborted) return;

    const replyNames = replyIds
      .map((id) => groupDisplayNames[id] || capitalize(id))
      .join(" → ");
    const observerNames = observerIds
      .map((id) => groupDisplayNames[id] || capitalize(id))
      .join(", ");
    const listening = observerNames ? ` · Listening: ${observerNames}` : "";
    const stopBit = data.stop_reason
      ? ` · ${formatPeerStopReason(data.stop_reason)}`
      : "";
    setStatus(
      replies.length
        ? `Nudge: ${replyNames}${listening}${stopBit}.`
        : `No one volunteered to speak${stopBit ? ` · ${formatPeerStopReason(data.stop_reason)}` : ""}.`
    );
  } catch (err) {
    if (err?.name === "AbortError") return;
    removeTypingIndicator();
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    if (groupSessionId) {
      groupAdvanceBtn.disabled = false;
      messageInput.disabled = false;
      sendBtn.disabled = false;
    }
  }
}

async function exportGroupSession() {
  if (!groupSessionId) return;
  try {
    const data = await api(`/api/group-sessions/${groupSessionId}/export`);
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `group-session-${groupSessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus("Transcript export downloaded.");
  } catch (err) {
    setStatus(`Export failed: ${err.message}`, "error");
  }
}

async function resetGroupSession() {
  cancelSpeechStreams();
  if (groupSessionId) {
    try {
      await api(`/api/group-sessions/${groupSessionId}`, { method: "DELETE" });
    } catch (_) {
      /* ignore */
    }
  }
  groupSessionId = null;
  groupTranscript = [];
  groupDisplayNames = {};
  groupStudents = [];
  lastGroupMeta = null;
  showChatEmpty();
  groupRoster.textContent = "Prepare a session to see the roster.";
  groupExportBtn.disabled = true;
  groupResetBtn.disabled = true;
  if (groupAdvanceBtn) groupAdvanceBtn.disabled = true;
  messageInput.disabled = true;
  sendBtn.disabled = true;
  startPracticeBtn.disabled = true;
  setStatus("Group session cleared. Prepare a new one to continue.");
}

async function sendMessage(text) {
  if (mode === "group") {
    await sendGroupMessage(text);
  } else {
    await sendMessage1to1(text);
  }
}

function insertTeacherPrompt(text) {
  messageInput.value = text;
  if (!messageInput.disabled) {
    messageInput.focus();
    messageInput.setSelectionRange(text.length, text.length);
  } else {
    setStatus("Start practice first, then use a script line.");
  }
}

async function copyDemoScript() {
  const lines = [
    "1. (Start — opener with phone plans)",
    "2. What do the 0.10 and 0.30 mean in each plan?",
    "3. Can someone represent Plan A and Plan B with equations or a table?",
    "4. Discuss together — when is each plan better?",
    "5. Jordan, please watch while Alex and Sam compare approaches",
    "6. At how many texts do the plans cost the same?",
  ];
  await navigator.clipboard.writeText(lines.join("\n"));
  setStatus("Phone-plans demo script copied.");
}

async function endAndRestart() {
  if (mode === "group") {
    await resetGroupSession();
  } else {
    await endOneToOneSession();
  }
  showSetup();
  setSetupStep(1);
  setStatus("Set up a group to start practice.");
}

async function prepareCurrentMode() {
  if (mode === "group") {
    await createGroupSession();
  } else {
    await createSession();
  }
}

async function startPractice() {
  if (mode === "group") {
    await startGroupSession();
  } else {
    await startSession();
  }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

backToSetupBtn.addEventListener("click", async () => {
  if (groupSessionId) {
    await resetGroupSession();
  }
  setSetupStep(1);
});

prepareBtn.addEventListener("click", () => {
  prepareCurrentMode().catch((err) => setStatus(`Error: ${err.message}`, "error"));
});

startPracticeBtn.addEventListener("click", () => {
  startPractice().catch((err) => setStatus(`Error: ${err.message}`, "error"));
});

endSessionBtn.addEventListener("click", () => {
  endAndRestart().catch((err) => setStatus(`Error: ${err.message}`, "error"));
});

profileSelect?.addEventListener("change", () => {
  onProfileChange().catch((err) => setStatus(`Error: ${err.message}`, "error"));
});

groupPreset.addEventListener("change", syncGroupPresetUI);
groupTaskSelect.addEventListener("change", syncGroupTaskUI);

groupExportBtn.addEventListener("click", () => {
  exportGroupSession().catch((err) =>
    setStatus(`Error: ${err.message}`, "error")
  );
});
if (groupAdvanceBtn) {
  groupAdvanceBtn.addEventListener("click", () => {
    advanceGroupDiscussion().catch((err) =>
      setStatus(`Error: ${err.message}`, "error")
    );
  });
}
groupResetBtn.addEventListener("click", () => {
  resetGroupSession()
    .then(() => {
      showSetup();
      setSetupStep(1);
    })
    .catch((err) => setStatus(`Error: ${err.message}`, "error"));
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () =>
    insertTeacherPrompt(button.dataset.prompt)
  );
});
copyDemoBtn.addEventListener("click", () => {
  copyDemoScript().catch(() =>
    setStatus(
      "Could not copy automatically. Select individual script lines instead.",
      "error"
    )
  );
});

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(messageInput.value).catch((err) =>
    setStatus(`Error: ${err.message}`, "error")
  );
});

// Keyboard: Enter to send (Shift+Enter for newline)
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled && messageInput.value.trim()) {
      chatForm.requestSubmit();
    }
  }
});

syncGroupPresetUI();
syncGroupTaskUI();
setSetupStep(1);
showSetup();
setStatus("Set up a group to start practice.");
loadProfiles().catch((err) => setStatus(`Error: ${err.message}`, "error"));
