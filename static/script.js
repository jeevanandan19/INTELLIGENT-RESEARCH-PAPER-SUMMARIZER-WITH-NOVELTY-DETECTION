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

  // Show loading spinner after submitting
  form.addEventListener("submit", (e) => {
    loading.classList.remove("hidden");
  });
});
