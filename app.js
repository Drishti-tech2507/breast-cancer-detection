// Simple demo UI logic and client-side "mock" prediction

// Features to show (short list for demo)
const FEATURE_NAMES = [
  "mean radius",
  "mean texture",
  "mean perimeter",
  "mean area",
  "mean smoothness",
  "mean compactness"
];

const inputsGrid = document.getElementById("inputsGrid");
const predictionOutput = document.getElementById("predictionOutput");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const demoSampleBtn = document.getElementById("demoSampleBtn");
const accVal = document.getElementById("accVal");
const featChartEl = document.getElementById("featChart").getContext("2d");
const previewChartEl = document.getElementById("previewChart").getContext("2d");

// create inputs
let currentValues = {};
FEATURE_NAMES.forEach((f, i) => {
  const wrapper = document.createElement("div");
  wrapper.className = "input-field";
  const label = document.createElement("label");
  label.innerText = f;
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.value = (i%2===0) ? 10 + i*2 : 12 + i*1.5;
  input.oninput = () => {
    currentValues[f] = parseFloat(input.value || 0);
    updateCharts();
  };
  wrapper.appendChild(label);
  wrapper.appendChild(input);
  inputsGrid.appendChild(wrapper);
  currentValues[f] = parseFloat(input.value);
});

// Fake model accuracy (demo)
accVal.innerText = "0.94";

// Chart: feature importance-like (initial)
let featChart = new Chart(featChartEl, {
  type: 'bar',
  data: {
    labels: FEATURE_NAMES,
    datasets: [{
      label: 'Feature value',
      data: FEATURE_NAMES.map(f => currentValues[f]),
      backgroundColor: FEATURE_NAMES.map((_,i) => i%2? 'rgba(99,102,241,0.8)' : 'rgba(99,102,241,0.5)')
    }]
  },
  options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}}
});

let previewChart = new Chart(previewChartEl, {
  type: 'radar',
  data: {
    labels: FEATURE_NAMES,
    datasets: [{
      label: 'Current inputs',
      data: FEATURE_NAMES.map(f => currentValues[f]),
      backgroundColor: 'rgba(253, 230, 138,0.3)',
      borderColor: 'rgba(253, 230, 138,1)',
      pointBackgroundColor: 'rgba(253, 230, 138,1)'
    }]
  },
  options: { responsive:true, maintainAspectRatio:false, scales:{r:{beginAtZero:true}} }
});

function updateCharts(){
  featChart.data.datasets[0].data = FEATURE_NAMES.map(f => currentValues[f]);
  featChart.update();
  previewChart.data.datasets[0].data = FEATURE_NAMES.map(f => currentValues[f]);
  previewChart.update();
}

// Mock prediction rule:
// If mean radius + mean perimeter is above threshold -> Malignant else Benign
function mockPredict(vals){
  const r = vals["mean radius"] || 0;
  const p = vals["mean perimeter"] || 0;
  const score = r * 0.6 + p * 0.4;
  // demo thresholds tuned for sample values
  return score > 25 ? { label: "Malignant", code: 0 } : { label: "Benign", code: 1 };
}

predictBtn.addEventListener("click", () => {
  // read current values (they are kept updated)
  const res = mockPredict(currentValues);
  if(res.code === 1){
    predictionOutput.className = "prediction benign";
    predictionOutput.innerHTML = `Benign ✅<div style="font-size:13px;color:var(--muted);margin-top:6px">Confidence (demo): ${(Math.random()*0.12 + 0.88).toFixed(2)}</div>`;
  } else {
    predictionOutput.className = "prediction malign";
    predictionOutput.innerHTML = `Malignant ❌<div style="font-size:13px;color:var(--muted);margin-top:6px">Confidence (demo): ${(Math.random()*0.12 + 0.75).toFixed(2)}</div>`;
  }
  // animate chart to highlight differences
  featChart.options.animation = {duration:700};
  previewChart.options.animation = {duration:700};
  updateCharts();
});

resetBtn.addEventListener("click", () => {
  // reset to default demo numbers
  document.querySelectorAll("#inputsGrid input").forEach((el,i)=> {
    el.value = (i%2===0) ? 10 + i*2 : 12 + i*1.5;
    el.dispatchEvent(new Event('input'));
  });
  predictionOutput.className = "prediction default";
  predictionOutput.innerText = "No prediction yet";
});

demoSampleBtn.addEventListener("click", () => {
  // load small sample values to inputs
  const sample = [14.2, 18.3, 92.0, 600.4, 0.096, 0.1];
  document.querySelectorAll("#inputsGrid input").forEach((el,i)=> {
    el.value = sample[i] || el.value;
    el.dispatchEvent(new Event('input'));
  });
  // show preview change
  updateCharts();
});

// Download PDF (demo): generate small PDF using html2canvas + jsPDF (fast fallback)
document.getElementById("downloadPdfBtn").addEventListener("click", () => {
  alert("This is a demo button. To enable PDF export integrate a backend or client-side library (jsPDF).");
});

// initial charts render
updateCharts();
