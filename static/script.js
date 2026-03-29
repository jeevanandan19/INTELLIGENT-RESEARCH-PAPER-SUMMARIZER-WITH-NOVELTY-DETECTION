document.addEventListener("DOMContentLoaded", () => {
  const uploadBox = document.getElementById("uploadBox");
  const fileLabel = document.getElementById("fileLabel");
  const fileInput = document.getElementById("fileInput");
  const form = document.getElementById("uploadForm");
  const loading = document.getElementById("loading");

  // Drag-and-drop file handler
  uploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadBox.classList.add("dragover");
  });
  uploadBox.addEventListener("dragleave", () => {
    uploadBox.classList.remove("dragover");
  });
  uploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadBox.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && (file.type === "application/pdf" || file.name.endsWith(".docx"))) {
        fileInput.files = e.dataTransfer.files;
        fileLabel.textContent = `✅ Selected: ${file.name}`;
      } else {
        alert("Please upload a PDF or Word (.docx) file!");
      }
  });

  // File selection display
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) fileLabel.textContent = `✅ Selected: ${file.name}`;
  });

  // Show loading spinner with cycling status messages
  form.addEventListener("submit", (e) => {
    const file = fileInput.files[0];
    if (file && file.size > 20 * 1024 * 1024) {
      e.preventDefault();
      alert("File is too large. Maximum allowed size is 20MB.");
      return;
    }
    loading.classList.remove("hidden");
    const analyzeBtn = document.getElementById("analyzeBtn");
    if (analyzeBtn) {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Analyzing...";
    }
    const steps = [
      "Extracting text from your paper... 📄",
      "Generating AI summary... 🧠",
      "Fetching related papers online... 🌐",
      "Computing novelty score... 📊",
      "Almost done, hang tight... ⏳"
    ];
    let i = 0;
    const msgEl = loading.querySelector(".loading-text");
    msgEl.textContent = steps[0];
    setInterval(() => {
      i = (i + 1) % steps.length;
      msgEl.textContent = steps[i];
    }, 4000);
  });
});
