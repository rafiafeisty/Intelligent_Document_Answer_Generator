const BACKEND_URL = "http://127.0.0.1:8000";

let selectedFiles = [];
let isProcessing = false;
let isQuerying = false;
let documentStats = {
    totalDocs: 0,
    totalPages: 0,
    totalSentences: 0,
    ready: false
};

const chatMessages = document.getElementById("chat-messages");
const loadingMessage = document.getElementById("loading-message");
const queryProgress = document.getElementById("query-progress");
const askBtn = document.getElementById("ask-btn");
const questionInput = document.getElementById("question");


function updateFileInfo() {
    const fileInfo = document.getElementById("file-info");
    const uploadBtn = document.getElementById("upload-btn");

    if (selectedFiles.length === 0) {
        fileInfo.textContent = "No files selected";
        uploadBtn.disabled = true;
    } else {
        const names = selectedFiles.map(f => f.name).join(", ");
        fileInfo.innerHTML = `<strong>${selectedFiles.length} file(s)</strong><br>${names.substring(0,120)}${names.length>120?'...':''}`;
        uploadBtn.disabled = false;
    }
}

function clearFiles() {
    selectedFiles = [];
    document.getElementById("files").value = "";
    updateFileInfo();
}

async function uploadDocs() {
    if (selectedFiles.length === 0 || isProcessing) return;

    isProcessing = true;
    const uploadBtn = document.getElementById("upload-btn");
    uploadBtn.disabled = true;

    document.getElementById("upload-progress").style.display = "block";
    document.getElementById("progress-fill").style.width = "0%";
    document.getElementById("progress-percent").textContent = "0%";
    document.getElementById("progress-details").textContent = "Uploading files...";

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append("files", file));

    try {
        const res = await fetch(`${BACKEND_URL}/upload`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) throw new Error(`Server error ${res.status}`);

        let progress = 0;
        const interval = setInterval(() => {
            progress += 8;
            if (progress > 100) progress = 100;
            document.getElementById("progress-fill").style.width = progress + "%";
            document.getElementById("progress-percent").textContent = progress + "%";
            if (progress >= 100) {
                clearInterval(interval);
                document.getElementById("progress-details").textContent = "Processing complete!";
            }
        }, 180);

        const data = await res.json();
        clearInterval(interval);
        document.getElementById("progress-fill").style.width = "100%";
        document.getElementById("progress-percent").textContent = "100%";
        document.getElementById("progress-details").textContent = data.message || "Documents processed successfully";

        documentStats = {
            totalDocs: selectedFiles.length,
            totalPages: 42 * selectedFiles.length,    
            totalSentences: 320 * selectedFiles.length, 
            ready: true
        };

        document.getElementById("stat-docs").textContent = documentStats.totalDocs;
        document.getElementById("stat-pages").textContent = documentStats.totalPages;
        document.getElementById("stat-sentences").textContent = documentStats.totalSentences;
        document.getElementById("stat-ready").textContent = "Yes";
        document.getElementById("stats-container").style.display = "block";

        document.getElementById("query-status").querySelector(".status-text").textContent = "Ready";
        document.getElementById("query-status").querySelector(".status-dot").className = "status-dot ready";

    } catch (err) {
        document.getElementById("progress-details").textContent = "Error: " + err.message;
        document.getElementById("progress-details").style.color = "#ef4444";
    } finally {
        isProcessing = false;
        uploadBtn.disabled = false;
        setTimeout(() => {
            document.getElementById("upload-progress").style.display = "none";
        }, 1800);
    }
}


function addMessage(content, type = "ai") {
    const div = document.createElement("div");
    div.className = `message ${type}`;
    div.innerHTML = `<div class="message-content">${content}</div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function ask() {
    const question = questionInput.value.trim();
    if (!question) return;

    if (!documentStats.ready) {
        addMessage("Please upload and process documents first.", "system");
        return;
    }

    addMessage(question, "user");
    questionInput.value = "";

    askBtn.disabled = true;
    loadingMessage.classList.remove("hidden");
    queryProgress.classList.remove("hidden");
    document.getElementById("query-progress-fill").style.width = "0%";

    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 12;
        if (progress > 100) progress = 100;
        document.getElementById("query-progress-fill").style.width = progress + "%";
        if (progress >= 100) clearInterval(progressInterval);
    }, 220);

    try {
        const res = await fetch(`${BACKEND_URL}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });

        if (!res.ok) throw new Error(`Server error ${res.status}`);

        const data = await res.json();

        clearInterval(progressInterval);
        document.getElementById("query-progress-fill").style.width = "100%";

        let html = `<strong>Answer:</strong><br>${(data.answer || "No answer generated").replace(/\n/g, "<br>")}<br><br>`;

        if (data.sources?.length > 0) {
            html += `<strong>Sources:</strong><ul style="margin:10px 0 10px 20px; padding-left:0;">`;
            data.sources.forEach(s => {
                const fname = s.file.split(/[\\/]/).pop();
                html += `<li><strong>${fname}</strong>  (${(s.relevance_score*100).toFixed(0)}%)<br><em>${s.snippet}</em></li>`;
            });
            html += `</ul>`;
        }

        html += `<br><em>Confidence: ${(data.confidence * 100).toFixed(1)}%</em>`;

        loadingMessage.classList.add("hidden");
        queryProgress.classList.add("hidden");
        addMessage(html, "ai");

    } catch (err) {
        loadingMessage.classList.add("hidden");
        queryProgress.classList.add("hidden");
        addMessage(`<span style="color:#dc2626">Error: ${err.message}</span>`, "ai");
    } finally {
        askBtn.disabled = false;
    }
}

function clearQuestion() {
    questionInput.value = "";
}

document.getElementById("files").addEventListener("change", e => {
    selectedFiles = Array.from(e.target.files);
    updateFileInfo();
});

questionInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        ask();
    }
});