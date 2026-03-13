'use strict';

// ── Constants ──────────────────────────────────────────────────────────────
const CELL_SIZE = 40; // 10 * 40 = 400px
const COLORS = {
  0: '#1a1a2e', // empty
  1: '#4ade80', // body
  2: '#22d3ee', // head
  3: '#f97316', // food
};
const MAX_CHART_POINTS = 500;

// ── DOM refs ───────────────────────────────────────────────────────────────
const canvas       = document.getElementById('gameCanvas');
const ctx          = canvas.getContext('2d');
const btnStart     = document.getElementById('btnStart');
const btnPause     = document.getElementById('btnPause');
const btnWatch     = document.getElementById('btnWatch');
const btnReset     = document.getElementById('btnReset');
const speedSlider  = document.getElementById('speedSlider');
const speedLabel   = document.getElementById('speedLabel');
const connBadge    = document.getElementById('connection-status');

const mEpisode = document.getElementById('mEpisode');
const mScore   = document.getElementById('mScore');
const mEpsilon = document.getElementById('mEpsilon');
const mLoss    = document.getElementById('mLoss');
const mMean    = document.getElementById('mMean');
const mStatus  = document.getElementById('mStatus');

const livScore = document.getElementById('livScore');
const livSteps = document.getElementById('livSteps');

// ── State ──────────────────────────────────────────────────────────────────
let trainingRunning = false;

// ── Chart setup ───────────────────────────────────────────────────────────
const chartCanvas = document.getElementById('scoreChart');
const chart = new Chart(chartCanvas, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'Score',
        data: [],
        borderColor: '#4ade80',
        backgroundColor: 'rgba(74,222,128,0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      },
      {
        label: 'Moyenne glissante',
        data: [],
        borderColor: '#22d3ee',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: false,
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: { color: '#94a3b8', boxWidth: 12, font: { size: 11 } },
      },
    },
    scales: {
      x: {
        ticks: { color: '#94a3b8', maxTicksLimit: 6, font: { size: 10 } },
        grid: { color: '#2d3748' },
      },
      y: {
        ticks: { color: '#94a3b8', font: { size: 10 } },
        grid: { color: '#2d3748' },
        min: 0,
      },
    },
  },
});

// ── Canvas drawing ─────────────────────────────────────────────────────────
function drawGrid(grid) {
  const n = grid.length;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const val = grid[r][c];
      ctx.fillStyle = COLORS[val] ?? COLORS[0];
      ctx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);

      // subtle grid lines
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
    }
  }
}

// Draw initial empty grid
function drawEmptyGrid(size = 10) {
  const empty = Array.from({ length: size }, () => Array(size).fill(0));
  drawGrid(empty);
}
drawEmptyGrid();

// ── Chart updates ─────────────────────────────────────────────────────────
function updateChart(scoreHistory) {
  if (!scoreHistory || scoreHistory.length === 0) return;

  // Keep only last MAX_CHART_POINTS
  const slice = scoreHistory.slice(-MAX_CHART_POINTS);
  const startEp = Math.max(1, scoreHistory.length - MAX_CHART_POINTS + 1);
  const labels = slice.map((_, i) => startEp + i);

  // Rolling mean (window = 20)
  const window = 20;
  const rollingMean = slice.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const sub = slice.slice(start, i + 1);
    return sub.reduce((a, b) => a + b, 0) / sub.length;
  });

  chart.data.labels = labels;
  chart.data.datasets[0].data = slice;
  chart.data.datasets[1].data = rollingMean;
  chart.update('none');
}

// ── Metrics update ────────────────────────────────────────────────────────
function updateMetrics(data) {
  if (data.episode !== undefined) mEpisode.textContent = data.episode;
  if (data.score   !== undefined) mScore.textContent   = data.score;
  if (data.epsilon !== undefined) mEpsilon.textContent = data.epsilon.toFixed(4);
  if (data.loss    !== undefined) {
    mLoss.textContent = data.loss === 0 ? '—' : data.loss.toFixed(5);
  }
  if (data.mean_score !== undefined) mMean.textContent = data.mean_score.toFixed(2);
}

function setStatus(status) {
  const labels = {
    running: 'Entraînement',
    paused:  'En pause',
    reset:   'Arrêté',
  };
  mStatus.textContent = labels[status] ?? status;
}

// ── Button states ─────────────────────────────────────────────────────────
function setButtonState(status) {
  trainingRunning = status === 'running';
  btnStart.disabled = trainingRunning;
  btnPause.disabled = !trainingRunning;
  setStatus(status);
}

// ── Socket.IO ─────────────────────────────────────────────────────────────
const socket = io({ transports: ['websocket', 'polling'] });

socket.on('connect', () => {
  connBadge.textContent = 'Connecté';
  connBadge.className   = 'badge badge-connected';
});

socket.on('disconnect', () => {
  connBadge.textContent = 'Déconnecté';
  connBadge.className   = 'badge badge-disconnected';
  setStatus('paused');
});

socket.on('game_state', (data) => {
  drawGrid(data.grid);
  livScore.textContent = data.score ?? 0;
  livSteps.textContent = data.steps ?? 0;
});

socket.on('metrics', (data) => {
  updateMetrics(data);
  updateChart(data.score_history);
});

socket.on('training_status', (data) => {
  setButtonState(data.status);
});

socket.on('watch_done', () => {
  btnWatch.disabled = false;
  btnWatch.textContent = '👁 Regarder';
});

// ── Controls ──────────────────────────────────────────────────────────────
btnStart.addEventListener('click', () => {
  socket.emit('start_training', {});
  setButtonState('running');
});

btnPause.addEventListener('click', () => {
  socket.emit('pause_training', {});
  setButtonState('paused');
});

btnWatch.addEventListener('click', () => {
  btnWatch.disabled = true;
  btnWatch.textContent = '⏳ En cours…';
  socket.emit('watch_game', {});
});

btnReset.addEventListener('click', () => {
  socket.emit('reset_training', {});
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.data.datasets[1].data = [];
  chart.update('none');
  drawEmptyGrid();
  livScore.textContent = 0;
  livSteps.textContent = 0;
  updateMetrics({ episode: 0, score: 0, epsilon: 1, loss: 0, mean_score: 0 });
  setButtonState('reset');
});

speedSlider.addEventListener('input', (e) => {
  const val = parseInt(e.target.value, 10);
  speedLabel.textContent = `Toutes les ${val} épisode${val > 1 ? 's' : ''}`;
  socket.emit('set_speed', { emit_every: val });
});

// ── Keep-alive ping (avoids Render free tier sleep) ───────────────────────
setInterval(() => {
  if (!trainingRunning) socket.emit('ping_keepalive');
}, 25000);
