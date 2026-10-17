/**
 * app.js — DocuMind AI Frontend
 * --------------------------------
 * Handles: PDF upload (drag & drop), document list, chat UI,
 * API communication, toast notifications, and auto-resize textarea.
 */

'use strict';

// ── Constants ────────────────────────────────────────────────────────────────
const API = {
  UPLOAD:       '/api/upload',
  DOCUMENTS:    '/api/documents',
  CHAT:         '/api/chat',
  CHAT_HISTORY: '/api/chat/history',
  HEALTH:       '/health',
};

// ── DOM References ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const DOM = {
  sidebar:       $('sidebar'),
  sidebarToggle: $('sidebarToggle'),
  mobileMenuBtn: $('mobileMenuBtn'),
  overlay:       $('overlay'),
  dropzone:      $('dropzone'),
  fileInput:     $('fileInput'),
  uploadProgress:$('uploadProgress'),
  progressFill:  $('progressFill'),
  progressLabel: $('progressLabel'),
  documentList:  $('documentList'),
  docEmpty:      $('docEmpty'),
  docCount:      $('docCount'),
  chatArea:      $('chatArea'),
  welcomeScreen: $('welcomeScreen'),
  messageList:   $('messageList'),
  questionInput: $('questionInput'),
  sendBtn:       $('sendBtn'),
  clearChatBtn:  $('clearChatBtn'),
  toastContainer:$('toastContainer'),
  modelName:     $('modelName'),
};

// ── State ────────────────────────────────────────────────────────────────────
let isLoading = false;
let documents = [];

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  attachEventListeners();
  await Promise.all([fetchHealth(), fetchDocuments(), fetchChatHistory()]);
}

// ── Event Listeners ──────────────────────────────────────────────────────────
function attachEventListeners() {
  // Sidebar toggle (desktop collapse)
  DOM.sidebarToggle.addEventListener('click', toggleSidebar);
  DOM.mobileMenuBtn.addEventListener('click', openMobileSidebar);
  DOM.overlay.addEventListener('click', closeMobileSidebar);

  // Drag-and-drop upload
  DOM.dropzone.addEventListener('dragover',  e => { e.preventDefault(); DOM.dropzone.classList.add('dragover'); });
  DOM.dropzone.addEventListener('dragleave', ()  => DOM.dropzone.classList.remove('dragover'));
  DOM.dropzone.addEventListener('drop',      handleDrop);
  DOM.dropzone.addEventListener('keydown',   e => (e.key === 'Enter' || e.key === ' ') && DOM.fileInput.click());
  DOM.fileInput.addEventListener('change',   handleFileSelect);

  // Chat
  DOM.questionInput.addEventListener('keydown', handleInputKeydown);
  DOM.questionInput.addEventListener('input',   autoResizeTextarea);
  DOM.sendBtn.addEventListener('click',         sendQuestion);
  DOM.clearChatBtn.addEventListener('click',    clearChat);

  // Example questions
  document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.questionInput.value = btn.dataset.question;
      autoResizeTextarea();
      DOM.questionInput.focus();
    });
  });
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
function toggleSidebar() {
  const isCollapsed = DOM.sidebar.style.width === '0px' || DOM.sidebar.style.width === '';
  // simple mobile-first approach: just use class on mobile
  DOM.sidebar.classList.toggle('open');
}

function openMobileSidebar() {
  DOM.sidebar.classList.add('open');
  DOM.overlay.classList.remove('hidden');
}

function closeMobileSidebar() {
  DOM.sidebar.classList.remove('open');
  DOM.overlay.classList.add('hidden');
}

// ── Health Check ─────────────────────────────────────────────────────────────
async function fetchHealth() {
  try {
    const res  = await fetch(API.HEALTH);
    const data = await res.json();
    DOM.modelName.textContent = data.llm_model || 'Unknown';

    if (!data.gemini_key_configured && !data.openai_key_configured) {
      showToast('No LLM API key configured. Answers will show raw context only.', 'warning', 8000);
    }
  } catch {
    DOM.modelName.textContent = 'Offline';
  }
}

// ── Document Upload ───────────────────────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  DOM.dropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) uploadFile(file);
  e.target.value = ''; // allow re-upload of same file
}

async function uploadFile(file) {
  // Client-side validation
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    return showToast('Only PDF files are accepted.', 'error');
  }
  if (file.size > 52_428_800) {
    return showToast('File too large. Maximum size is 50 MB.', 'error');
  }
  if (file.size === 0) {
    return showToast('The selected file is empty.', 'error');
  }

  // Show progress
  DOM.uploadProgress.classList.remove('hidden');
  DOM.progressLabel.textContent = `Uploading ${file.name}…`;
  animateProgress(0, 40, 600);

  const formData = new FormData();
  formData.append('file', file);

  try {
    animateProgress(40, 75, 800);
    const res  = await fetch(API.UPLOAD, { method: 'POST', body: formData });
    const data = await res.json();

    animateProgress(75, 100, 400);

    if (data.success) {
      showToast(`"${data.document.original_name}" uploaded successfully!`, 'success');
      if (data.warning) showToast(data.warning, 'warning', 6000);
      await fetchDocuments();
    } else {
      showToast(data.error || 'Upload failed.', 'error');
    }
  } catch (err) {
    showToast('Network error during upload. Please try again.', 'error');
    console.error('Upload error:', err);
  } finally {
    setTimeout(() => {
      DOM.uploadProgress.classList.add('hidden');
      DOM.progressFill.style.width = '0';
    }, 600);
  }
}

function animateProgress(from, to, duration) {
  const start = performance.now();
  const update = ts => {
    const p = Math.min((ts - start) / duration, 1);
    DOM.progressFill.style.width = `${from + (to - from) * p}%`;
    if (p < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// ── Document List ─────────────────────────────────────────────────────────────
async function fetchDocuments() {
  try {
    const res  = await fetch(API.DOCUMENTS);
    const data = await res.json();
    if (data.success) {
      documents = data.documents;
      renderDocuments();
    }
  } catch (err) {
    console.error('Error fetching documents:', err);
  }
}

function renderDocuments() {
  const ready = documents.filter(d => d.status === 'ready');
  DOM.docCount.textContent = ready.length;

  if (documents.length === 0) {
    DOM.docEmpty.classList.remove('hidden');
    // Remove all doc items except the empty placeholder
    Array.from(DOM.documentList.querySelectorAll('.doc-item')).forEach(el => el.remove());
    return;
  }

  DOM.docEmpty.classList.add('hidden');

  // Rebuild list
  Array.from(DOM.documentList.querySelectorAll('.doc-item')).forEach(el => el.remove());

  documents.forEach(doc => {
    const li = document.createElement('li');
    li.className = 'doc-item';
    li.dataset.id = doc.id;

    const statusMap = { ready: '● Ready', failed: '✕ Failed', processing: '⏳ Processing' };
    const sizeKB = (doc.file_size / 1024).toFixed(1);

    li.innerHTML = `
      <div class="doc-icon" aria-hidden="true">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <rect x="2" y="1" width="12" height="14" rx="2" stroke="currentColor" stroke-width="1.4"/>
          <path d="M5 5h6M5 8h6M5 11h4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="doc-info">
        <p class="doc-name" title="${escHtml(doc.original_name)}">${escHtml(doc.original_name)}</p>
        <p class="doc-meta">${doc.page_count || 0} pages · ${sizeKB} KB</p>
        <span class="doc-status ${doc.status}">${statusMap[doc.status] || doc.status}</span>
      </div>
      <button class="doc-delete" data-id="${doc.id}" aria-label="Remove ${escHtml(doc.original_name)}" title="Remove document">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
      </button>
    `;
    DOM.documentList.appendChild(li);
  });

  // Attach delete handlers
  DOM.documentList.querySelectorAll('.doc-delete').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      await deleteDocument(parseInt(btn.dataset.id));
    });
  });
}

async function deleteDocument(docId) {
  const doc = documents.find(d => d.id === docId);
  const name = doc ? doc.original_name : 'this document';

  if (!confirm(`Remove "${name}" from the session?`)) return;

  try {
    const res  = await fetch(`${API.DOCUMENTS}/${docId}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      showToast(`"${name}" removed.`, 'info');
      await fetchDocuments();
    } else {
      showToast(data.error || 'Could not remove document.', 'error');
    }
  } catch {
    showToast('Network error. Please try again.', 'error');
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────
async function fetchChatHistory() {
  try {
    const res  = await fetch(API.CHAT_HISTORY);
    const data = await res.json();
    if (data.success && data.history.length > 0) {
      hideWelcome();
      data.history.forEach(msg => {
        if (msg.role === 'user') appendUserMessage(msg.content, false);
        else appendAssistantMessage(msg.content, msg.sources || [], false);
      });
      scrollToBottom();
    }
  } catch (err) {
    console.error('Error fetching history:', err);
  }
}

function handleInputKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
}

async function sendQuestion() {
  const question = DOM.questionInput.value.trim();
  if (!question || isLoading) return;

  hideWelcome();
  isLoading = true;
  DOM.sendBtn.disabled = true;
  DOM.questionInput.value = '';
  autoResizeTextarea();

  appendUserMessage(question);
  const loadingEl = appendLoadingMessage();
  scrollToBottom();

  try {
    const res  = await fetch(API.CHAT, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question }),
    });
    const data = await res.json();

    loadingEl.remove();

    if (data.success) {
      appendAssistantMessage(data.answer, data.sources || []);
    } else {
      appendAssistantMessage(
        data.error || 'An error occurred. Please try again.',
        []
      );
    }
  } catch {
    loadingEl.remove();
    appendAssistantMessage('Network error. Please check your connection and try again.', []);
    showToast('Request failed. Please try again.', 'error');
  } finally {
    isLoading = false;
    DOM.sendBtn.disabled = false;
    DOM.questionInput.focus();
    scrollToBottom();
  }
}

async function clearChat() {
  if (!confirm('Clear the entire conversation?')) return;
  try {
    await fetch(API.CHAT_HISTORY, { method: 'DELETE' });
    DOM.messageList.innerHTML = '';
    showWelcome();
    showToast('Conversation cleared.', 'info');
  } catch {
    showToast('Could not clear history.', 'error');
  }
}

// ── Message rendering ─────────────────────────────────────────────────────────
function appendUserMessage(text, animate = true) {
  const msg = createMessageEl('user', animate);
  msg.querySelector('.message-bubble').textContent = text;
  DOM.messageList.appendChild(msg);
  return msg;
}

function appendAssistantMessage(text, sources = [], animate = true) {
  const msg = createMessageEl('assistant', animate);
  const bubble = msg.querySelector('.message-bubble');

  // Simple markdown-like rendering
  bubble.innerHTML = renderMarkdown(text);

  // Sources
  if (sources && sources.length > 0) {
    const srcSection = document.createElement('div');
    srcSection.className = 'sources-section';
    srcSection.innerHTML = `<p class="sources-label">Sources</p>`;
    sources.forEach(src => {
      const pages = src.pages && src.pages.length ? `Pages: ${src.pages.join(', ')}` : '';
      srcSection.innerHTML += `
        <div class="source-card">
          <svg class="source-icon" width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <rect x="1" y="1" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.3"/>
            <path d="M4 5h6M4 7.5h6M4 10h4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
          </svg>
          <span class="source-name" title="${escHtml(src.document_name)}">${escHtml(src.document_name)}</span>
          <span class="source-pages">${pages}</span>
        </div>`;
    });
    msg.querySelector('.message-body').appendChild(srcSection);
  }

  DOM.messageList.appendChild(msg);
  return msg;
}

function appendLoadingMessage() {
  const msg = createMessageEl('assistant loading-message', false);
  msg.querySelector('.message-bubble').innerHTML = `
    <div class="typing-dots" aria-label="AI is thinking">
      <span></span><span></span><span></span>
    </div>
    <span style="font-size:13px">Searching documents…</span>`;
  DOM.messageList.appendChild(msg);
  return msg;
}

function createMessageEl(role, animate) {
  const el   = document.createElement('div');
  const baseRole = role.split(' ')[0]; // 'user' or 'assistant'
  el.className = `message ${role}${animate ? '' : ''}`;

  const initials = baseRole === 'user' ? 'You' : 'AI';
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  el.innerHTML = `
    <div class="message-avatar" aria-hidden="true">${initials}</div>
    <div class="message-body">
      <div class="message-bubble" role="${baseRole === 'assistant' ? 'article' : 'paragraph'}"></div>
      <p class="message-time" aria-label="Sent at ${now}">${now}</p>
    </div>`;
  return el;
}

// Very basic markdown → HTML (no external library needed)
function renderMarkdown(text) {
  if (!text) return '';
  let html = escHtml(text);
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Code inline
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bullet lists
  html = html.replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
  // Numbered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  // Paragraphs
  html = html.replace(/\n{2,}/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  return `<p>${html}</p>`;
}

// ── UI Helpers ────────────────────────────────────────────────────────────────
function hideWelcome() {
  DOM.welcomeScreen.classList.add('hidden');
}
function showWelcome() {
  DOM.welcomeScreen.classList.remove('hidden');
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    DOM.chatArea.scrollTo({ top: DOM.chatArea.scrollHeight, behavior: 'smooth' });
  });
}

function autoResizeTextarea() {
  DOM.questionInput.style.height = 'auto';
  DOM.questionInput.style.height = Math.min(DOM.questionInput.scrollHeight, 140) + 'px';
}

// ── Toast ─────────────────────────────────────────────────────────────────────
const TOAST_ICONS = {
  success: `<svg class="toast-icon" width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.4"/><path d="M5 8l2 2 4-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  error:   `<svg class="toast-icon" width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.4"/><path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  warning: `<svg class="toast-icon" width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2L14 13H2L8 2z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M8 6v4M8 11.5v.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  info:    `<svg class="toast-icon" width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.4"/><path d="M8 7v4M8 5v.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
};

function showToast(message, type = 'info', duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `${TOAST_ICONS[type] || TOAST_ICONS.info}<span>${escHtml(message)}</span>`;
  DOM.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Security helper ───────────────────────────────────────────────────────────
function escHtml(str) {
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
