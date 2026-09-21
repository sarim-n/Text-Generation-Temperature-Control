// Datasets for candidate tokens and logits
const SCENARIOS = {
  sky: {
    prompt: "The sky is ",
    tokens: [
      { text: "blue", logit: 8.5 },
      { text: "clear", logit: 6.2 },
      { text: "dark", logit: 5.4 },
      { text: "cloudy", logit: 4.8 },
      { text: "falling", logit: 2.1 },
      { text: "green", logit: 1.2 },
      { text: "purple", logit: 0.8 },
      { text: "banana", logit: -2.5 }
    ]
  },
  code: {
    prompt: "def calculate_average(numbers):\n    return sum(numbers) / ",
    tokens: [
      { text: "len(numbers)", logit: 9.8 },
      { text: "count", logit: 5.1 },
      { text: "N", logit: 4.2 },
      { text: "size", logit: 3.8 },
      { text: "total", logit: 1.5 },
      { text: "0", logit: 0.2 },
      { text: "True", logit: -1.0 },
      { text: "\"average\"", logit: -3.0 }
    ]
  },
  story: {
    prompt: "Once upon a time in a distant galaxy, there was a ",
    tokens: [
      { text: "starship", logit: 7.2 },
      { text: "king", logit: 6.8 },
      { text: "robot", logit: 6.1 },
      { text: "dragon", logit: 5.5 },
      { text: "alien", logit: 4.9 },
      { text: "wizard", logit: 3.8 },
      { text: "programmer", logit: 2.4 },
      { text: "sandwich", logit: -0.5 }
    ]
  }
};

let currentScenario = 'sky';
let generatedText = '';

// DOM Elements
const tempSlider = document.getElementById('temp-slider');
const tempValText = document.getElementById('temp-val-text');
const tempCategory = document.getElementById('temp-category');

const topkSlider = document.getElementById('topk-slider');
const topkValText = document.getElementById('topk-val-text');

const toppSlider = document.getElementById('topp-slider');
const toppValText = document.getElementById('topp-val-text');

const scenarioSelect = document.getElementById('scenario-select');
const barsContainer = document.getElementById('bars-container');

const entropyVal = document.getElementById('entropy-val');
const activeCandidatesVal = document.getElementById('active-candidates');
const maxProbVal = document.getElementById('max-prob');

const promptPrefix = document.getElementById('prompt-prefix');
const generatedTokens = document.getElementById('generated-tokens');
const samplingHistory = document.getElementById('sampling-history');
const sampleBtn = document.getElementById('sample-btn');

const presetBtns = document.querySelectorAll('.preset-btn');

// Calculate Softmax with Temperature, Top-K, and Top-P
function computeDistributions() {
  const temp = parseFloat(tempSlider.value);
  const topK = parseInt(topkSlider.value);
  const topP = parseFloat(toppSlider.value);

  const scenario = SCENARIOS[currentScenario];
  let tokens = scenario.tokens.map(t => ({ ...t }));

  // Step 1: Divide logits by Temperature
  tokens.forEach(t => {
    t.scaledLogit = t.logit / temp;
  });

  // Sort descending by scaledLogit
  tokens.sort((a, b) => b.scaledLogit - a.scaledLogit);

  // Softmax on raw/scaled logits
  const maxScaledLogit = tokens[0].scaledLogit; // numerical stability
  let sumExp = 0;
  tokens.forEach(t => {
    t.exp = Math.exp(t.scaledLogit - maxScaledLogit);
    sumExp += t.exp;
  });

  tokens.forEach(t => {
    t.prob = t.exp / sumExp;
  });

  // Apply Top-K filtering
  tokens.forEach((t, index) => {
    t.passedTopK = index < topK;
  });

  // Apply Top-P (Nucleus) filtering
  let cumProb = 0;
  tokens.forEach(t => {
    if (cumProb < topP) {
      t.passedTopP = true;
      cumProb += t.prob;
    } else {
      t.passedTopP = false;
    }
  });

  // Active filter combination
  tokens.forEach(t => {
    t.active = t.passedTopK && t.passedTopP;
  });

  // Re-normalize probabilities among active tokens
  const activeTokens = tokens.filter(t => t.active);
  const activeSumProb = activeTokens.reduce((acc, t) => acc + t.prob, 0);

  tokens.forEach(t => {
    if (t.active) {
      t.finalProb = t.prob / activeSumProb;
    } else {
      t.finalProb = 0;
    }
  });

  return { tokens, activeTokens, temp, topK, topP };
}

// Update UI
function updateUI() {
  const { tokens, activeTokens, temp, topK, topP } = computeDistributions();

  // Update text labels
  tempValText.textContent = temp.toFixed(2);
  topkValText.textContent = `Top ${topK}`;
  toppValText.textContent = topP.toFixed(2);

  // Temperature category badge
  if (temp < 0.3) {
    tempCategory.textContent = "Strict / Factual";
    tempCategory.style.background = "rgba(6, 182, 212, 0.2)";
    tempCategory.style.color = "#22d3ee";
  } else if (temp <= 0.9) {
    tempCategory.textContent = "Balanced";
    tempCategory.style.background = "rgba(99, 102, 241, 0.2)";
    tempCategory.style.color = "#818cf8";
  } else if (temp <= 1.5) {
    tempCategory.textContent = "Creative";
    tempCategory.style.background = "rgba(245, 158, 11, 0.2)";
    tempCategory.style.color = "#fbbf24";
  } else {
    tempCategory.textContent = "High Entropy / Random";
    tempCategory.style.background = "rgba(244, 63, 94, 0.2)";
    tempCategory.style.color = "#fb7185";
  }

  // Calculate Entropy: H(P) = -sum(p * log2(p))
  let entropy = 0;
  activeTokens.forEach(t => {
    if (t.finalProb > 0) {
      entropy -= t.finalProb * Math.log2(t.finalProb);
    }
  });
  entropyVal.textContent = `${entropy.toFixed(2)} bits`;
  activeCandidatesVal.textContent = `${activeTokens.length} token${activeTokens.length === 1 ? '' : 's'}`;

  const maxProb = activeTokens.length > 0 ? Math.max(...activeTokens.map(t => t.finalProb)) : 0;
  maxProbVal.textContent = `${(maxProb * 100).toFixed(1)}%`;

  // Render Bar Graph
  barsContainer.innerHTML = '';
  tokens.forEach(t => {
    const row = document.createElement('div');
    row.className = `token-row ${t.active ? '' : 'filtered'}`;

    const percentage = (t.finalProb * 100).toFixed(1);

    row.innerHTML = `
      <div class="token-name" title="${t.text}">"${t.text}"</div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${percentage}%"></div>
      </div>
      <div class="token-prob">${percentage}%</div>
      <div class="token-logit">z=${t.logit.toFixed(1)}</div>
    `;
    barsContainer.appendChild(row);
  });

  promptPrefix.textContent = `"${SCENARIOS[currentScenario].prompt}"`;
}

// Token Sampling Logic
function sampleToken() {
  const { activeTokens } = computeDistributions();
  if (activeTokens.length === 0) return;

  const rand = Math.random();
  let cumulative = 0;
  let chosenToken = activeTokens[0];

  for (let t of activeTokens) {
    cumulative += t.finalProb;
    if (rand <= cumulative) {
      chosenToken = t;
      break;
    }
  }

  generatedText += ` ${chosenToken.text}`;
  generatedTokens.textContent = generatedText;

  samplingHistory.innerHTML = `Sampled token <strong style="color: #6366f1;">"${chosenToken.text}"</strong> with probability <strong>${(chosenToken.finalProb * 100).toFixed(1)}%</strong> (random roll: ${rand.toFixed(3)}).`;
}

// Event Listeners
tempSlider.addEventListener('input', updateUI);
topkSlider.addEventListener('input', updateUI);
toppSlider.addEventListener('input', updateUI);

scenarioSelect.addEventListener('change', (e) => {
  currentScenario = e.target.value;
  generatedText = '';
  generatedTokens.textContent = '';
  samplingHistory.textContent = 'Click "Sample Next Token" to pick a token based on the calculated distribution above.';
  updateUI();
});

sampleBtn.addEventListener('click', sampleToken);

presetBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    presetBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    tempSlider.value = btn.dataset.temp;
    topkSlider.value = btn.dataset.topk;
    toppSlider.value = btn.dataset.topp;
    updateUI();
  });
});

// Initial render
updateUI();
