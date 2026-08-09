// Streams an answer over server-sent events and renders it as it arrives.
//
// fetch() with a manually parsed event stream is used rather than EventSource,
// because EventSource cannot issue a POST and the question does not belong in a
// query string. The parsing this costs is a dozen lines.

const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const submit = document.getElementById("submit");
const thread = document.getElementById("thread");
const status = document.getElementById("status");

let busy = false;

function setStatus(message, isError = false) {
  status.hidden = !message;
  status.textContent = message;
  status.classList.toggle("error", isError);
  status.classList.toggle("dots", Boolean(message) && !isError);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Citation markers become links into the source list below the answer. Done on
// the accumulated text after each token rather than per token, because a marker
// can be split across two deltas and a per-token rewrite would miss it.
function renderAnswer(element, text, exchangeId) {
  element.innerHTML = escapeHtml(text).replace(
    /\[(\d+)\]/g,
    (_, n) => `<a class="cite" href="#src-${exchangeId}-${n}">${n}</a>`,
  );
}

function sourceCard(source, exchangeId) {
  const card = document.createElement("div");
  card.className = "source";
  card.id = `src-${exchangeId}-${source.marker}`;

  const meta = [];
  if (source.section) meta.push(escapeHtml(source.section));
  if (source.phase) meta.push(escapeHtml(source.phase));
  meta.push(`p.${source.page}`);

  // The scale is named beside the grade because JOSPT grades of recommendation
  // and NATA evidence categories share letters and are not interchangeable. A
  // bare "Grade A" would invite exactly that confusion.
  const grade = source.grade
    ? `<span class="grade">Grade ${escapeHtml(source.grade)} · ${escapeHtml(
        source.grade_scale || "unspecified scale",
      )}</span>`
    : "";

  const link = source.url
    ? ` · <a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">source document</a>`
    : "";

  card.innerHTML = `
    <div class="source-head">
      <span class="source-marker">[${source.marker}]</span>
      <span class="source-title">${escapeHtml(source.title)}</span>
    </div>
    <div class="source-meta">${meta.join(" · ")}${link} ${grade}</div>
    <details class="excerpt">
      <summary>Show the passage this came from</summary>
      <p>${escapeHtml(source.text || "")}</p>
    </details>`;
  return card;
}

async function ask(question) {
  if (busy) return;
  busy = true;
  submit.disabled = true;
  setStatus("Searching the corpus");

  const exchangeId = Date.now();
  const exchange = document.createElement("article");
  exchange.className = "exchange";
  exchange.innerHTML = `<p class="question">${escapeHtml(question)}</p>
    <div class="answer"></div>
    <div class="sources" hidden><h3>Sources cited</h3><div class="source-list"></div></div>`;
  // Newest first, so a new answer does not push the reader down the page.
  thread.prepend(exchange);

  const answerEl = exchange.querySelector(".answer");
  const sourcesEl = exchange.querySelector(".sources");
  const listEl = exchange.querySelector(".source-list");

  let text = "";
  let retrieved = [];

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Events are separated by a blank line. The trailing fragment is kept in
      // the buffer, since a chunk boundary can land inside an event.
      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const event = JSON.parse(line.slice(6));

        if (event.kind === "sources") {
          retrieved = event.sources;
          setStatus("Reading sources");
        } else if (event.kind === "token") {
          text += event.text;
          renderAnswer(answerEl, text, exchangeId);
          setStatus("");
        } else if (event.kind === "cited") {
          // Only the sources the answer actually cited are listed. Showing
          // everything retrieved would present material the answer never used
          // as though it backed the claims.
          const cited = event.sources.length ? event.sources : retrieved;
          sourcesEl.hidden = false;
          for (const source of cited) listEl.appendChild(sourceCard(source, exchangeId));
        } else if (event.kind === "error") {
          setStatus(event.text, true);
        }
      }
    }
    if (!text.trim()) setStatus("No answer was returned. Please try again.", true);
  } catch (error) {
    setStatus(`Could not reach MedBrain: ${error.message}`, true);
  } finally {
    busy = false;
    submit.disabled = false;
    status.classList.remove("dots");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (question) {
    ask(question);
    input.value = "";
  }
});

// Enter sends, Shift+Enter makes a new line.
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.getElementById("examples").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-q]");
  if (button) ask(button.dataset.q);
});

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    document.getElementById("health").textContent = h.ok
      ? `${h.chunks} indexed passages across the corpus.`
      : "Index is empty. Answers are unavailable.";
  })
  .catch(() => {});
