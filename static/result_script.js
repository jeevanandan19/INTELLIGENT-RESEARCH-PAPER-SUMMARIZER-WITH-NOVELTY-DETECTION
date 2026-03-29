document.addEventListener("DOMContentLoaded", () => {
  // Animate novelty bar
  const fill = document.getElementById("noveltyFill");
  if (fill) {
    const pct = parseFloat(fill.innerText);
    fill.style.width = "0%";
    setTimeout(() => { fill.style.width = pct + "%"; }, 400);
  }

  // Collapsible sections
  const coll = document.getElementsByClassName("collapsible");
  for (let i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function () {
      this.classList.toggle("active");
      const content = this.nextElementSibling;
      content.style.display = (content.style.display === "block") ? "none" : "block";
    });
  }

  // Chart.js Visualization
  const ctx = document.getElementById("similarityChart");
  if (ctx) {
    const paperTitles = JSON.parse(ctx.dataset.labels);
    const similarities = JSON.parse(ctx.dataset.values);
    const paperUrls = JSON.parse(ctx.dataset.urls);
    const paperSources = JSON.parse(ctx.dataset.sources);

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: paperTitles,
        datasets: [{
          label: "Similarity (%)",
          data: similarities,
          backgroundColor: "#1a73e8cc",
          borderColor: "#1a73e8",
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(context) {
                const idx = context.dataIndex;
                return `${paperSources[idx]} — ${context.parsed.y.toFixed(2)}% similarity`;
              }
            }
          },
          title: {
            display: true,
            text: "Similarity Comparison with Online Papers (click bar to open paper)",
            font: { size: 16 }
          }
        },
        scales: {
          x: { ticks: { font: { size: 12 }, maxRotation: 45, minRotation: 45 } },
          y: { beginAtZero: true, max: 100 }
        },
        onClick: (e, elements) => {
          if (elements.length > 0) {
            const idx = elements[0].index;
            const url = paperUrls[idx];
            if (url) window.open(url, "_blank");
          }
        }
      }
    });
  }

  // Download summary as text file
  const downloadBtn = document.getElementById("downloadBtn");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {
      const summaryEl = document.querySelector(".summary-block");
      const noveltyEl = document.querySelector(".novelty-label");
      const noveltyPct = document.getElementById("noveltyFill");
      const filename = document.title.replace("Research Paper Analysis Result", "").trim() || "summary";

      let content = "=== RESEARCH PAPER ANALYSIS ===\n\n";
      content += "STRUCTURED SUMMARY:\n" + (summaryEl ? summaryEl.innerText : "") + "\n\n";
      content += "NOVELTY SCORE: " + (noveltyPct ? noveltyPct.innerText : "") + "\n";
      content += "NOVELTY LABEL: " + (noveltyEl ? noveltyEl.innerText : "") + "\n";

      const blob = new Blob([content], { type: "text/plain" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "analysis_summary.txt";
      a.click();
      URL.revokeObjectURL(a.href);
    });
  }
});
