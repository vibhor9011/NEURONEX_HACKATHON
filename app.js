/**
 * Neuro-Nex: Stroke Assistive Neuromorphic Interface (SANI)
 * Core Application Logic, Neuromorphic SNN Engine & Visualization Controller
 */

// --- Global Simulation Constants & States ---
const SIM_TICK_RATE = 5; // Simulation steps per animation frame (for numerical stability)
const SAMPLE_RATE = 200; // Simulated signal sampling rate in Hz
const DT = 1000 / SAMPLE_RATE; // Time step in ms (~5ms)
const BUFFER_SIZE = 300; // Size of scrolling signal history buffer

// Color Theme Map (Clinical Light Theme Palette)
const COLORS = {
  bg: '#ffffff', // pure white canvas backgrounds
  panelBg: '#ffffff',
  eeg: '#0891b2', // Cyan 600
  emg: '#8b5cf6', // Purple 600
  emgExt: '#db2777', // Pink 600
  spikeUp: '#16a34a', // Green 600
  spikeDown: '#dc2626', // Red 600
  amber: '#d97706', // Amber 600
  text: '#0f172a', // Slate 900
  textSecondary: '#475569', // Slate 600
  textMuted: '#94a3b8' // Slate 400
};

// System State
let attemptState = 'idle'; // 'idle', 'grasp', 'release', 'spasm'
let graspCount = 0;
let releaseCount = 0;
let spasmCount = 0;
let profileState = 'mild'; // 'healthy', 'mild', 'severe', 'spastic'
let latencyStartTime = 0; // Timestamp of when the current voluntary attempt began
let currentLatency = null; // Decoded intent latency in ms
let totalSynapticOps = 0; // Cumulative synaptic operations count
let totalSpikesCount = 0; // Cumulative input spikes count
let avgSpikeRate = 0; // Encoded spike rate in Hz
let activePower = 10.0; // Estimated power in microwatts (starts at static idle power)
let powerBuffer = Array(100).fill(10.0); // Historical power records for tracking average
let telemetrySynapticCurrent = 0.0; // Smoothed integrated synaptic current

// Interactive Sliders Configuration
const sliders = {
  deltaThresh: { element: null, value: 0.15 },
  neuronThresh: { element: null, value: 1.20 },
  leakRate: { element: null, value: 20 }, // tau in ms
  synWeight: { element: null, value: 1.0 }
};

// Continuous Signal Generator States
const signals = {
  eeg: { data: Array(BUFFER_SIZE).fill(0), refVal: 0, lastSpikeTime: 0 },
  emgFlex: { data: Array(BUFFER_SIZE).fill(0), refVal: 0, lastSpikeTime: 0 },
  emgExt: { data: Array(BUFFER_SIZE).fill(0), refVal: 0, lastSpikeTime: 0 }
};

// Delta Modulation Encoded Event Streams (Stores positive and negative spike occurrences)
// Each spike event is an object: { x: canvas_x, type: 'up'|'down', time: timestamp }
const spikeEvents = {
  eeg: [],
  emgFlex: [],
  emgExt: []
};

// Leaky Integrate-and-Fire (LIF) Neuron Model
class LIFNeuron {
  constructor(name, color) {
    this.name = name;
    this.color = color;
    this.v = 0.0; // Current membrane potential (V)
    this.vReset = 0.0; // Reset potential
    this.lastFireTime = -100; // Timestamp of last firing (for refractory period)
    this.refractoryPeriod = 6; // Refractory period in ms (~1-2 steps)
    this.vHistory = Array(BUFFER_SIZE).fill(0); // For plotting V_m(t)
    this.fireEvents = []; // Output fire timestamps
  }

  integrate(inputSpikeSum, dt, tau, threshold) {
    const now = performance.now();
    
    // Check if in refractory period
    if (now - this.lastFireTime < this.refractoryPeriod) {
      this.v = this.vReset;
      this.vHistory.push(this.v);
      this.vHistory.shift();
      return false;
    }

    // Leaky membrane decay: V(t) = V(t-1) * exp(-dt / tau)
    const decayFactor = Math.exp(-dt / tau);
    this.v = this.v * decayFactor + inputSpikeSum;
    
    // Enforce lower bound (no negative hyperpolarization beyond limit)
    if (this.v < -0.5) this.v = -0.5;

    let fired = false;
    // Check threshold firing condition
    if (this.v >= threshold) {
      fired = true;
      this.v = this.vReset; // Reset membrane potential
      this.lastFireTime = now;
      this.fireEvents.push(now);
      
      // Clean old fire events
      if (this.fireEvents.length > 50) this.fireEvents.shift();
    }

    this.vHistory.push(this.v);
    this.vHistory.shift();
    return fired;
  }
}

// SNN Neurons
const neurons = {
  grasp: new LIFNeuron('Grasp Intent', COLORS.spikeUp),
  release: new LIFNeuron('Release Intent', COLORS.eeg),
  spasm: new LIFNeuron('Spasm Alert', COLORS.spikeDown)
};

// Synaptic weights dictionary connecting input spike channels to output neurons
const synapses = {
  // Connections to Grasp Neuron
  eegToGrasp: -0.6,    // EEG Mu rhythm is inhibitory (suppression = excitation)
  emgFlexToGrasp: 1.2, // Forearm flexor muscle is strongly excitatory
  emgExtToGrasp: -0.4, // Forearm extensor muscle is inhibitory to Grasp
  
  // Connections to Release Neuron
  eegToRelease: -0.6,    // EEG Mu rhythm inhibitory
  emgFlexToRelease: -0.4, // Flexor muscle inhibitory to Release
  emgExtToRelease: 1.2,  // Extensor muscle is strongly excitatory
  
  // Connections to Spasm/Alert Neuron
  eegToSpasm: 0.1,      // EEG doesn't inhibit spasm (spasms are peripheral/spinal cord reflexes)
  emgFlexToSpasm: 1.0,  // Both muscle activation fires spasm
  emgExtToSpasm: 1.0
};

// Assistive Exoskeleton Hand Control State
let handClosure = 0.25; // 0 = wide open, 1 = tight grip, 0.25 = relaxed rest
let currentHandClosureTarget = 0.25;
let jitterMagnitude = 0; // For spasms

// HTML Canvas Contexts
let canvases = {
  signal: { el: null, ctx: null },
  spike: { el: null, ctx: null },
  snn: { el: null, ctx: null },
  potentials: { el: null, ctx: null },
  hand: { el: null, ctx: null }
};

// Animated Synapse Signals (Visual glow packets moving on synapses)
// Event: { startNode, endNode, color, progress: 0..1, speed: 0.05 }
let activeSynapticPackets = [];

// --- Stroke Profiles Configurations ---
const PROFILES = {
  healthy: {
    name: 'Healthy Subject',
    description: 'Healthy Subject: Strong EEG Mu desynchronization, high-amplitude EMG motor command, zero baseline spasms, high SNR.',
    eegMuAmp: 1.0,
    eegNoise: 0.05,
    emgGain: 1.2,
    emgNoise: 0.02,
    spasmRate: 0.0,
    presets: { deltaThresh: 0.12, neuronThresh: 1.00, leakRate: 20, synWeight: 1.1 }
  },
  mild: {
    name: 'Mild Motor Hemiparesis',
    description: 'Mild Motor Hemiparesis: Moderate EEG Mu-suppression, weaker EMG muscle contraction amplitude, moderate signal-to-noise ratio.',
    eegMuAmp: 0.8,
    eegNoise: 0.10,
    emgGain: 0.7,
    emgNoise: 0.05,
    spasmRate: 0.02,
    presets: { deltaThresh: 0.15, neuronThresh: 1.20, leakRate: 20, synWeight: 1.0 }
  },
  severe: {
    name: 'Severe Flaccid Paralysis',
    description: 'Severe Flaccid Paralysis: Negligible EEG Mu-suppression (poor cortical control), barely detectable EMG muscle contractions, high relative noise.',
    eegMuAmp: 0.5,
    eegNoise: 0.20,
    emgGain: 0.15,
    emgNoise: 0.12,
    spasmRate: 0.01,
    presets: { deltaThresh: 0.10, neuronThresh: 0.80, leakRate: 40, synWeight: 1.5 }
  },
  spastic: {
    name: 'Spasticity / High Noise',
    description: 'Spasticity / High Noise: High spastic EMG base contraction with sudden random bursts, erratic cortical EEG, requires higher detection thresholds.',
    eegMuAmp: 0.7,
    eegNoise: 0.15,
    emgGain: 0.8,
    emgNoise: 0.25,
    spasmRate: 0.15,
    presets: { deltaThresh: 0.22, neuronThresh: 1.50, leakRate: 15, synWeight: 0.9 }
  }
};

// --- Initialization ---
window.addEventListener('DOMContentLoaded', () => {
  initCanvases();
  initSliders();
  initButtons();
  initNavigation();
  initLogger();
  setupResizeHandler();
  
  // Apply default profile (Mild)
  applyProfile('mild');
  
  // Kickstart simulation loop
  requestAnimationFrame(loop);
});

// Setup Canvas DOM references & sizing
function initCanvases() {
  canvases.signal.el = document.getElementById('signalCanvas');
  canvases.spike.el = document.getElementById('spikeCanvas');
  canvases.snn.el = document.getElementById('snnCanvas');
  canvases.potentials.el = document.getElementById('potentialsCanvas');
  canvases.hand.el = document.getElementById('handCanvas');

  for (let key in canvases) {
    canvases[key].ctx = canvases[key].el.getContext('2d');
  }
  
  resizeCanvases();
}

function resizeCanvases() {
  for (let key in canvases) {
    const canvas = canvases[key].el;
    const rect = canvas.parentNode.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
  }
}

function setupResizeHandler() {
  let timer;
  window.addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      resizeCanvases();
    }, 150);
  });
}

// Tab navigation router
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const viewports = document.querySelectorAll('.tab-viewport');
  const tabTitle = document.getElementById('currentTabTitle');
  const tabSubtitle = document.getElementById('currentTabSubtitle');

  const tabMeta = {
    monitor: {
      title: "Live Monitor",
      subtitle: "Continuous real-time biological wave plotting and SNN spike integrations"
    },
    actuator: {
      title: "Exoskeleton Actuator",
      subtitle: "Visual actuator and feedback motor position indicators"
    },
    analytics: {
      title: "Diagnostics & Analytics",
      subtitle: "Continuous real-time edge benchmarks, session statistics, and logs"
    },
    settings: {
      title: "System Calibration",
      subtitle: "Hardware parameter optimization and recovery profile selection"
    },
    guide: {
      title: "User Guide & Help Manual",
      subtitle: "Simple guide to understanding sensor signals, spike encoding, SNN, and diagnostics stats"
    }
  };

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const tabName = item.getAttribute('data-tab');

      // Update active nav class
      navItems.forEach(nav => nav.classList.remove('active'));
      item.classList.add('active');

      // Toggle panels
      viewports.forEach(vp => vp.classList.add('hidden'));
      const activeVp = document.getElementById('tab-' + tabName);
      if (activeVp) activeVp.classList.remove('hidden');

      // Update headers text
      if (tabMeta[tabName]) {
        tabTitle.innerText = tabMeta[tabName].title;
        tabSubtitle.innerText = tabMeta[tabName].subtitle;
      }

      // Re-trigger resize to properly dimension active canvases
      resizeCanvases();
      logSystemEvent(`Navigation: Switched workspace to "${tabMeta[tabName].title}"`, 'system');
    });
  });
}

// Clinical Terminal Clear Logic
function initLogger() {
  const btnClearLog = document.getElementById('btnClearLog');
  if (btnClearLog) {
    btnClearLog.addEventListener('click', () => {
      const consoleEl = document.getElementById('terminalLog');
      if (consoleEl) {
        consoleEl.innerHTML = `<div class="log-line status-type">[${getFormattedTime()}] Awaiting voluntary SNN intent triggers...</div>`;
      }
    });
  }
}

// Log message to virtual clinical screen logger
function logSystemEvent(message, type = 'system') {
  const consoleEl = document.getElementById('terminalLog');
  if (!consoleEl) return;

  const logLine = document.createElement('div');
  logLine.className = `log-line ${type}-type`;
  logLine.innerText = `[${getFormattedTime()}] ${message}`;
  
  consoleEl.appendChild(logLine);

  // Auto-scroll terminal
  consoleEl.scrollTop = consoleEl.scrollHeight;

  // Prevent memory issues by capping logs length
  while (consoleEl.children.length > 80) {
    consoleEl.removeChild(consoleEl.firstChild);
  }
}

function getFormattedTime() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}

// Sliders and Event Listeners
function initSliders() {
  sliders.deltaThresh.element = document.getElementById('slideDeltaThresh');
  sliders.neuronThresh.element = document.getElementById('slideNeuronThresh');
  sliders.leakRate.element = document.getElementById('slideLeakRate');
  sliders.synWeight.element = document.getElementById('slideSynWeight');

  // Wire event handlers to display slider values instantly
  sliders.deltaThresh.element.addEventListener('input', (e) => {
    sliders.deltaThresh.value = parseFloat(e.target.value);
    document.getElementById('valDeltaThresh').innerText = `${sliders.deltaThresh.value.toFixed(2)} V`;
  });

  sliders.neuronThresh.element.addEventListener('input', (e) => {
    sliders.neuronThresh.value = parseFloat(e.target.value);
    document.getElementById('valNeuronThresh').innerText = `${sliders.neuronThresh.value.toFixed(2)} V`;
  });

  sliders.leakRate.element.addEventListener('input', (e) => {
    sliders.leakRate.value = parseInt(e.target.value);
    document.getElementById('valLeakRate').innerText = `${sliders.leakRate.value} ms`;
  });

  sliders.synWeight.element.addEventListener('input', (e) => {
    sliders.synWeight.value = parseFloat(e.target.value);
    document.getElementById('valSynWeight').innerText = `${sliders.synWeight.value.toFixed(2)}`;
  });
}

// Bind Action Buttons
function initButtons() {
  // Intent simulation triggers
  const btnIdle = document.getElementById('btnIdle');
  const btnGrasp = document.getElementById('btnGrasp');
  const btnRelease = document.getElementById('btnRelease');
  const btnSpasm = document.getElementById('btnSpasm');

  const intentBtns = [btnIdle, btnGrasp, btnRelease, btnSpasm];

  function setActiveIntentButton(activeBtn) {
    intentBtns.forEach(btn => btn.classList.remove('active'));
    activeBtn.classList.add('active');
  }

  btnIdle.addEventListener('click', () => {
    setAttemptState('idle');
    setActiveIntentButton(btnIdle);
  });

  btnGrasp.addEventListener('click', () => {
    setAttemptState('grasp');
    setActiveIntentButton(btnGrasp);
  });

  btnRelease.addEventListener('click', () => {
    setAttemptState('release');
    setActiveIntentButton(btnRelease);
  });

  btnSpasm.addEventListener('click', () => {
    setAttemptState('spasm');
    setActiveIntentButton(btnSpasm);
  });

  // Profiles buttons
  const profHealthy = document.getElementById('profHealthy');
  const profMild = document.getElementById('profMild');
  const profSevere = document.getElementById('profSevere');
  const profSpastic = document.getElementById('profSpastic');

  const profBtns = [profHealthy, profMild, profSevere, profSpastic];

  function setActiveProfileButton(activeBtn) {
    profBtns.forEach(btn => btn.classList.remove('active'));
    activeBtn.classList.add('active');
  }

  profHealthy.addEventListener('click', () => {
    applyProfile('healthy');
    setActiveProfileButton(profHealthy);
  });

  profMild.addEventListener('click', () => {
    applyProfile('mild');
    setActiveProfileButton(profMild);
  });

  profSevere.addEventListener('click', () => {
    applyProfile('severe');
    setActiveProfileButton(profSevere);
  });

  profSpastic.addEventListener('click', () => {
    applyProfile('spastic');
    setActiveProfileButton(profSpastic);
  });

  // PDF Report download button binding
  const btnDownloadReport = document.getElementById('btnDownloadReport');
  if (btnDownloadReport) {
    btnDownloadReport.addEventListener('click', () => {
      generatePDFReport();
    });
  }
}

function setAttemptState(state) {
  attemptState = state;
  currentLatency = null; // Clear old latency display
  
  if (state === 'grasp' || state === 'release') {
    latencyStartTime = performance.now();
    logSystemEvent(`Attempt initiated: user attempting voluntary "${state.toUpperCase()}"`, 'status');
  } else if (state === 'spasm') {
    logSystemEvent(`Simulation trigger: muscle spasm alert active`, 'spasm');
  } else {
    logSystemEvent(`Attempt initiated: patient returning to rest tone`, 'system');
  }
}

// Load a specific stroke patient profile settings
function applyProfile(profileKey) {
  profileState = profileKey;
  const config = PROFILES[profileKey];
  
  // Set UI Text Description
  document.getElementById('profileText').innerText = config.description;
  
  // Apply presets to slider inputs and values
  for (let key in config.presets) {
    const val = config.presets[key];
    sliders[key].value = val;
    sliders[key].element.value = val;
    
    // Update labels
    if (key === 'deltaThresh') document.getElementById('valDeltaThresh').innerText = `${val.toFixed(2)} V`;
    if (key === 'neuronThresh') document.getElementById('valNeuronThresh').innerText = `${val.toFixed(2)} V`;
    if (key === 'leakRate') document.getElementById('valLeakRate').innerText = `${val} ms`;
    if (key === 'synWeight') document.getElementById('valSynWeight').innerText = `${val.toFixed(2)}`;
  }

  // Force resetting biological simulation buffers to prevent overflow spikes
  signals.eeg.refVal = 0;
  signals.emgFlex.refVal = 0;
  signals.emgExt.refVal = 0;

  logSystemEvent(`Calibration loaded: Active Patient Profile set to "${config.name}"`, 'system');
}

// --- Real-time Bio-Signal Synthesis ---
// Generates EEG (Mu desynchronization) & dual EMG activity
let timeIndex = 0;
function generateBioSignals(profile, state) {
  timeIndex += 1;
  const t = timeIndex * DT * 0.001; // Simulation elapsed time in seconds

  // 1. EEG Mu-rhythm simulation (8-13 Hz)
  let eegMuAmp = profile.eegMuAmp;
  // If attempting action (grasp/release), the Mu rhythm desynchronizes (suppresses/flattens)
  if (state === 'grasp' || state === 'release') {
    eegMuAmp *= 0.18; // 82% suppression
  }
  
  // Baseline Mu rhythm wave (10Hz) + Beta rhythm wave (20Hz)
  const muWave = Math.sin(2 * Math.PI * 10.3 * t) * 0.8 * eegMuAmp;
  const betaWave = Math.sin(2 * Math.PI * 21.0 * t) * 0.35 * eegMuAmp;
  
  // Cortical noise
  const eegNoise = (Math.random() - 0.5) * profile.eegNoise * 1.5;
  const finalEeg = muWave + betaWave + eegNoise;
  
  // 2. EMG Signal synthesis (high-frequency noise modulated by contraction intensity)
  let muscleFlexActivity = 0.05; // Resting muscle tone
  let muscleExtActivity = 0.05;

  if (state === 'grasp') {
    muscleFlexActivity = 1.0; // Strong contraction
    muscleExtActivity = 0.02; // Antagonist muscle relaxed
  } else if (state === 'release') {
    muscleFlexActivity = 0.02;
    muscleExtActivity = 1.0; // Strong extensor contraction
  } else if (state === 'spasm') {
    // Spasm causes severe simultaneous co-contraction
    muscleFlexActivity = 0.8 + Math.sin(t * 15.0) * 0.2;
    muscleExtActivity = 0.8 + Math.cos(t * 15.0) * 0.2;
  } else if (profileState === 'spastic' && Math.random() < profile.spasmRate) {
    // Spontaneous brief spasm under spastic profile
    muscleFlexActivity = 0.6;
    muscleExtActivity = 0.5;
  }

  // Multiply by patient's physiological gain limits
  muscleFlexActivity *= profile.emgGain;
  muscleExtActivity *= profile.emgGain;

  // EMG is high-frequency raw electrical burst
  const flexNoise = (Math.random() - 0.5) * profile.emgNoise * 1.2;
  const extNoise = (Math.random() - 0.5) * profile.emgNoise * 1.2;
  
  const rawFlexEMG = (Math.random() - 0.5) * 1.4 * muscleFlexActivity + flexNoise;
  const rawExtEMG = (Math.random() - 0.5) * 1.4 * muscleExtActivity + extNoise;

  return {
    eeg: finalEeg,
    emgFlex: rawFlexEMG,
    emgExt: rawExtEMG
  };
}

// --- Neuromorphic Delta Modulation ---
// Converts continuous signal voltages to asynchronous spike events (+1 or -1)
function deltaModulate(signalKey, val, threshold, simTimeMs) {
  const channel = signals[signalKey];
  const diff = val - channel.refVal;
  
  let spike = 0; // 0 = no spike, 1 = UP spike, -1 = DOWN spike

  if (diff >= threshold) {
    spike = 1;
    channel.refVal += threshold; // Update step reference
    channel.lastSpikeTime = simTimeMs;
  } else if (diff <= -threshold) {
    spike = -1;
    channel.refVal -= threshold;
    channel.lastSpikeTime = simTimeMs;
  }

  // Leak reference value slowly back to baseline if no spikes occur (avoid baseline drift)
  if (spike === 0 && simTimeMs - channel.lastSpikeTime > 25) {
    channel.refVal = channel.refVal * 0.96;
  }

  return spike;
}

// --- SNN Simulation Core Engine Step ---
function stepSNN(simTimeMs) {
  const profile = PROFILES[profileState];
  const generated = generateBioSignals(profile, attemptState);

  // Buffer raw analog signals for displaying in oscilloscope
  for (let key in generated) {
    signals[key].data.push(generated[key]);
    signals[key].data.shift();
  }

  // Perform Delta Modulation
  const deltaThresh = sliders.deltaThresh.value;
  const spikes = {
    eegUp: deltaModulate('eeg', generated.eeg, deltaThresh, simTimeMs) === 1,
    eegDown: deltaModulate('eeg', generated.eeg, deltaThresh, simTimeMs) === -1,
    emgFlexUp: deltaModulate('emgFlex', generated.emgFlex, deltaThresh, simTimeMs) === 1,
    emgFlexDown: deltaModulate('emgFlex', generated.emgFlex, deltaThresh, simTimeMs) === -1,
    emgExtUp: deltaModulate('emgExt', generated.emgExt, deltaThresh, simTimeMs) === 1,
    emgExtDown: deltaModulate('emgExt', generated.emgExt, deltaThresh, simTimeMs) === -1
  };

  // Push spike events to display history queues
  const registerEvent = (channel, type) => {
    totalSpikesCount++;
    spikeEvents[channel].push({ x: BUFFER_SIZE - 1, type: type, time: simTimeMs });
    if (spikeEvents[channel].length > 150) spikeEvents[channel].shift();
  };

  if (spikes.eegUp) registerEvent('eeg', 'up');
  if (spikes.eegDown) registerEvent('eeg', 'down');
  
  if (spikes.emgFlexUp) registerEvent('emgFlex', 'up');
  if (spikes.emgFlexDown) registerEvent('emgFlex', 'down');
  
  if (spikes.emgExtUp) registerEvent('emgExt', 'up');
  if (spikes.emgExtDown) registerEvent('emgExt', 'down');

  // Shift older spike events visually on the canvas scroll axis
  for (let key in spikeEvents) {
    spikeEvents[key].forEach(event => {
      event.x -= 1; // Scroll speed matched to signals
    });
    // Remove out-of-screen spikes
    spikeEvents[key] = spikeEvents[key].filter(e => e.x >= 0);
  }

  // --- Neural Synaptic Accumulation ---
  // Count how many synaptic operations (SOPs) are triggered this frame
  const W_mult = sliders.synWeight.value;
  let activeOpsThisStep = 0;

  // Reset inputs sums for SNN integration
  let graspSpikeSum = 0;
  let releaseSpikeSum = 0;
  let spasmSpikeSum = 0;

  // Grasp input summation
  if (spikes.emgFlexUp) {
    graspSpikeSum += synapses.emgFlexToGrasp * W_mult;
    activeOpsThisStep++;
    triggerVisualSynapse('emgFlex', 'grasp', COLORS.spikeUp);
  }
  if (spikes.emgFlexDown) {
    graspSpikeSum += (synapses.emgFlexToGrasp * 0.5) * W_mult;
    activeOpsThisStep++;
    triggerVisualSynapse('emgFlex', 'grasp', COLORS.spikeUp);
  }
  if (spikes.eegUp || spikes.eegDown) {
    // Inhibition gating logic
    graspSpikeSum += synapses.eegToGrasp * W_mult;
    activeOpsThisStep++;
    triggerVisualSynapse('eeg', 'grasp', COLORS.spikeDown);
  }
  if (spikes.emgExtUp) {
    graspSpikeSum += synapses.emgExtToGrasp * W_mult;
    activeOpsThisStep++;
    triggerVisualSynapse('emgExt', 'grasp', COLORS.spikeDown);
  }

  // Release input summation
  if (spikes.emgExtUp) {
    releaseSpikeSum += synapses.emgExtToRelease * W_mult;
    activeOpsThisStep++;
    triggerVisualSynapse('emgExt', 'release', COLORS.eeg);
  }
  if (spikes.emgExtDown) {
    releaseSpikeSum += (synapses.emgExtToRelease * 0.5) * W_mult;
    activeOpsThisStep++;
    triggerVisualSynapse('emgExt', 'release', COLORS.eeg);
  }
  if (spikes.eegUp || spikes.eegDown) {
    releaseSpikeSum += synapses.eegToRelease * W_mult;
    activeOpsThisStep++;
    triggerVisualSynapse('eeg', 'release', COLORS.spikeDown);
  }
  if (spikes.emgFlexUp) {
    releaseSpikeSum += synapses.emgFlexToRelease * W_mult;
    activeOpsThisStep++;
    triggerVisualSynapse('emgFlex', 'release', COLORS.spikeDown);
  }

  // Spasm Alert input summation (activated when both Flex/Ext are actively spiking)
  if (spikes.emgFlexUp || spikes.emgFlexDown) {
    spasmSpikeSum += synapses.emgFlexToSpasm;
    activeOpsThisStep++;
    triggerVisualSynapse('emgFlex', 'spasm', COLORS.spikeDown);
  }
  if (spikes.emgExtUp || spikes.emgExtDown) {
    spasmSpikeSum += synapses.emgExtToSpasm;
    activeOpsThisStep++;
    triggerVisualSynapse('emgExt', 'spasm', COLORS.spikeDown);
  }

  totalSynapticOps += activeOpsThisStep;

  // Compute absolute integrated synaptic current input for telemetry display (scaled to mA)
  const stepSynapticCurrent = (Math.abs(graspSpikeSum) + Math.abs(releaseSpikeSum) + Math.abs(spasmSpikeSum));
  telemetrySynapticCurrent = telemetrySynapticCurrent * 0.95 + stepSynapticCurrent * 0.05;

  // Integrate and Fire LIF Neurons
  const leakRate = sliders.leakRate.value;
  const neuronThresh = sliders.neuronThresh.value;

  const firedGrasp = neurons.grasp.integrate(graspSpikeSum, DT, leakRate, neuronThresh);
  const firedRelease = neurons.release.integrate(releaseSpikeSum, DT, leakRate, neuronThresh);
  const firedSpasm = neurons.spasm.integrate(spasmSpikeSum, DT, leakRate, neuronThresh);

  // Process Output Triggers
  const now = performance.now();
  if (firedSpasm) {
    currentHandClosureTarget = 0.5;
    jitterMagnitude = 6.0;
    updateIntentBadge('spasm');

    // Stats and Logging
    spasmCount++;
    const spasmCountEl = document.getElementById('countSpasms');
    if (spasmCountEl) spasmCountEl.innerText = spasmCount;
    logSystemEvent("EMERGENCY BLOCKED: Muscle spasm detected peripherally. Actuators locked.", "spasm");
  } else if (firedGrasp) {
    currentHandClosureTarget = 1.0; // Clenched hand
    jitterMagnitude = 0;
    updateIntentBadge('grasp');
    calculateIntentLatency(now);

    // Stats and Logging
    graspCount++;
    const graspCountEl = document.getElementById('countGrasps');
    if (graspCountEl) graspCountEl.innerText = graspCount;
    const latencyMsg = currentLatency ? ` Latency: ${currentLatency.toFixed(1)}ms` : "";
    logSystemEvent(`SNN Trigger: GRASP intent detected.${latencyMsg}`, "grasp");
  } else if (firedRelease) {
    currentHandClosureTarget = 0.0; // Wide open hand
    jitterMagnitude = 0;
    updateIntentBadge('release');
    calculateIntentLatency(now);

    // Stats and Logging
    releaseCount++;
    const releaseCountEl = document.getElementById('countReleases');
    if (releaseCountEl) releaseCountEl.innerText = releaseCount;
    const latencyMsg = currentLatency ? ` Latency: ${currentLatency.toFixed(1)}ms` : "";
    logSystemEvent(`SNN Trigger: RELEASE intent detected.${latencyMsg}`, "release");
  }
}

// Compute neuromorphic trigger latency
function calculateIntentLatency(fireTimestamp) {
  if (currentLatency === null && (attemptState === 'grasp' || attemptState === 'release')) {
    // Latency is the delta between patient attempting motion and the SNN neuron firing
    currentLatency = fireTimestamp - latencyStartTime;
  }
}

function updateIntentBadge(type) {
  const badge = document.getElementById('intentBadge');
  badge.className = `intent-badge ${type}`;
  badge.innerText = type === 'spasm' ? 'Spasm Alert' : type;
  
  // Revert badge to Rest after delay if no longer active
  if (type !== 'spasm') {
    setTimeout(() => {
      // Only revert if we haven't switched states again
      if (badge.innerText.toLowerCase() === type) {
        badge.className = 'intent-badge idle';
        badge.innerText = 'Idle';
      }
    }, 1800);
  }
}

// Add visual signal packet running along the SNN synapses
function triggerVisualSynapse(sourceKey, targetKey, color) {
  activeSynapticPackets.push({
    source: sourceKey,
    target: targetKey,
    color: color,
    progress: 0.0,
    speed: 0.08 + Math.random() * 0.04
  });
  
  // Prevent packet overload
  if (activeSynapticPackets.length > 120) {
    activeSynapticPackets.shift();
  }
}

// --- Benchmark Math (Power and Latency) ---
let lastStatsTime = 0;
function updateEdgeBenchmarks(now) {
  if (now - lastStatsTime < 250) return; // Update UI every 250ms
  const elapsed = (now - lastStatsTime) * 0.001;
  lastStatsTime = now;

  // Dynamic Power Calculation
  // Loihi/neuromorphic hardware power: P_active = P_static + E_sop * SOP_rate
  // E_sop ~ 20 pJ (picojoules) = 20 * 10^-6 microwatt-seconds.
  // SOP_rate = (Total operations - previous operations) / elapsed
  const sopRate = Math.round(totalSynapticOps / elapsed);
  totalSynapticOps = 0; // Reset counter for next window

  // Spike rates
  const spikeRate = Math.round(totalSpikesCount / elapsed);
  totalSpikesCount = 0;

  // Energy consumption formula
  const E_sop = 0.00002; // in microwatt-seconds (20pJ)
  const dynamicPower = sopRate * E_sop;
  activePower = 10.0 + dynamicPower; // 10 µW static leakage baseline
  
  // Power averaging buffer
  powerBuffer.push(activePower);
  powerBuffer.shift();
  const avgPower = powerBuffer.reduce((a, b) => a + b, 0) / powerBuffer.length;

  // Update UI Elements
  document.getElementById('metricPower').innerText = `${avgPower.toFixed(1)} µW`;
  document.getElementById('metricSops').innerText = `${sopRate.toLocaleString()} / s`;
  document.getElementById('metricSpikerate').innerText = `${spikeRate.toLocaleString()} Hz`;

  // Calculate neuromorphic efficiency vs traditional DSP (which draws ~50mW / 50000µW)
  const sparsityPercentage = Math.min(99.9, ((50000 - avgPower) / 50000) * 100);
  document.getElementById('activePowerSparsity').innerText = `${sparsityPercentage.toFixed(1)}%`;

  // Update Latency displays
  if (currentLatency !== null) {
    document.getElementById('metricLatency').innerText = `${currentLatency.toFixed(1)} ms`;
    // Traditional DSP latency is window size (e.g. 120ms FFT) + classifier delay (~20ms) = 140ms
    const speedup = ((140 - currentLatency) / 140) * 100;
    document.getElementById('latencyImprovement').innerText = `${speedup.toFixed(0)}% faster`;
  } else {
    document.getElementById('metricLatency').innerText = '-- ms';
    document.getElementById('latencyImprovement').innerText = 'Real-time';
  }
}

// --- Live Telemetry Interpreter Update ---
function updateLiveTelemetryInterpreter() {
  const eegModeEl = document.getElementById('telemetry-eeg-mode');
  const emgToneEl = document.getElementById('telemetry-emg-tone');
  const snnCurrentEl = document.getElementById('telemetry-snn-current');
  const snrValueEl = document.getElementById('telemetry-snr-value');
  const diagnosticSentenceEl = document.getElementById('telemetry-diagnostic-sentence');
  
  if (!eegModeEl || !emgToneEl || !snnCurrentEl || !snrValueEl || !diagnosticSentenceEl) return;

  // 1. EEG Cortical Mode
  let eegText = "Resting (Mu Active)";
  let eegColor = "var(--text-secondary)";
  if (attemptState === 'grasp' || attemptState === 'release') {
    eegText = "Active Planning (Mu Suppressed)";
    eegColor = "var(--color-eeg)";
  }
  eegModeEl.innerText = eegText;
  eegModeEl.style.color = eegColor;

  // 2. EMG Muscle Tone
  let emgText = "Relaxed (Idle)";
  let emgColor = "var(--text-secondary)";
  if (attemptState === 'grasp') {
    emgText = "Active Flexion (High Burst)";
    emgColor = "var(--color-spike-up)";
  } else if (attemptState === 'release') {
    emgText = "Active Extension (High Burst)";
    emgColor = "var(--color-emg-ext)";
  } else if (attemptState === 'spasm') {
    emgText = "Co-contraction Spasm";
    emgColor = "var(--color-spike-down)";
  } else if (profileState === 'spastic') {
    emgText = "Elevated Spastic Tone";
    emgColor = "var(--color-amber)";
  }
  emgToneEl.innerText = emgText;
  emgToneEl.style.color = emgColor;

  // 3. SNN Synaptic Current
  snnCurrentEl.innerText = `${(telemetrySynapticCurrent * 2.2).toFixed(2)} mA`;

  // 4. Signal Quality (SNR)
  const snrMap = {
    healthy: "96% - Excellent SNR",
    mild: "82% - Good SNR",
    severe: "45% - High Noise / Faint",
    spastic: "68% - Distorted / Artifacts"
  };
  snrValueEl.innerText = snrMap[profileState] || "80% - Normal";
  if (profileState === 'healthy') {
    snrValueEl.style.color = "var(--color-spike-up)";
  } else if (profileState === 'mild') {
    snrValueEl.style.color = "var(--color-eeg)";
  } else if (profileState === 'spastic') {
    snrValueEl.style.color = "var(--color-amber)";
  } else {
    snrValueEl.style.color = "var(--color-spike-down)";
  }

  // 5. Dynamic Diagnostic Sentence
  let sentence = "";
  if (attemptState === 'idle') {
    if (profileState === 'healthy') {
      sentence = "Healthy control baseline. Standard Mu rhythm active over motor cortex. Hand muscles resting at 0.05V tone. System fully responsive.";
    } else if (profileState === 'mild') {
      sentence = "Hemiparetic baseline resting. Normal Mu rhythm planning frequencies active. Minor baseline EMG noise. Awaiting volitional movement triggers.";
    } else if (profileState === 'severe') {
      sentence = "Severe flaccid baseline resting. Flatline cortical amplitude. Barely registerable muscle potentials. Boosted LIF weight profiles activated.";
    } else if (profileState === 'spastic') {
      sentence = "Spastic baseline. Erratic peripheral muscle noise detected. Cortical signals partially occluded. High threshold filter calibration active.";
    }
  } else if (attemptState === 'grasp') {
    const latencyTerm = currentLatency ? ` resolved in ${currentLatency.toFixed(1)} ms.` : ".";
    if (profileState === 'healthy') {
      sentence = `Volitional grasp intent decoded. Complete cortical Mu suppression registered. Clear flexion EMG spike sequence. Exoskeleton hand closing${latencyTerm}`;
    } else if (profileState === 'mild') {
      sentence = `Motor planning intent decoded. Moderate cortical Mu suppression. SNN integrated flexor spike trains. Exoskeleton closing${latencyTerm}`;
    } else if (profileState === 'severe') {
      sentence = `Faint motor intent resolved. Negligible Mu suppression. Attenuated flexor burst amplified by +1.5 synaptic weights. Exoskeleton closing${latencyTerm}`;
    } else if (profileState === 'spastic') {
      sentence = `Grasp intent decoded. High background peripheral noise filtered by 1.50V LIF neuron threshold. Exoskeleton closing${latencyTerm}`;
    }
  } else if (attemptState === 'release') {
    const latencyTerm = currentLatency ? ` resolved in ${currentLatency.toFixed(1)} ms.` : ".";
    if (profileState === 'healthy') {
      sentence = `Volitional release intent decoded. EEG motor planning suppression clear. Forearm extensor EMG burst matched. Exoskeleton opening${latencyTerm}`;
    } else if (profileState === 'mild') {
      sentence = `Release intent decoded. Mu planning rhythms suppressed. SNN integrated extensor spike trains. Exoskeleton opening${latencyTerm}`;
    } else if (profileState === 'severe') {
      sentence = `Faint release intent resolved. Boosted synaptic weight calibration resolved attenuated extensor burst. Exoskeleton opening${latencyTerm}`;
    } else if (profileState === 'spastic') {
      sentence = `Release intent decoded. Heavy forearm co-contraction filtered. Exoskeleton opening${latencyTerm}`;
    }
  } else if (attemptState === 'spasm') {
    sentence = "EMERGENCY SAFETY OVERRIDE ACTIVE. Co-contraction spasm pattern identified in peripheral EMG. Exoskeleton actuator locked to prevent joint injury.";
  }
  diagnosticSentenceEl.innerText = sentence;
}

// --- Graphical Rendering Pipeline (HTML5 Canvas Drawing) ---

// 1. Oscilloscope drawing (Continuous signals)
function drawOscilloscope() {
  const { ctx, el } = canvases.signal;
  if (!ctx) return;
  
  // Clear canvas
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, el.width, el.height);

  // Draw background grids (subtle dividing axes)
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.03)';
  ctx.lineWidth = 1;
  const gridSpacing = 50;
  for (let x = 0; x < el.width; x += gridSpacing) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, el.height);
    ctx.stroke();
  }
  for (let y = 0; y < el.height; y += gridSpacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(el.width, y);
    ctx.stroke();
  }

  // Draw dividing line
  const midY = el.height / 2;
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)';
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(el.width, midY);
  ctx.stroke();

  // Helper to draw signal wave (clean vector lines)
  const drawWave = (dataBuffer, color, centerY, ampScale) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25; // Sleeker
    ctx.beginPath();
    
    const stepX = el.width / BUFFER_SIZE;
    for (let i = 0; i < BUFFER_SIZE; i++) {
      const x = i * stepX;
      const y = centerY - dataBuffer[i] * ampScale;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    
    // Crisp slight glow
    ctx.shadowBlur = 2;
    ctx.shadowColor = color;
    ctx.stroke();
    ctx.shadowBlur = 0; // reset
  };

  // Top half: EEG Mu signals
  drawWave(signals.eeg.data, COLORS.eeg, midY * 0.5, midY * 0.45);
  
  // Bottom half: EMG signals (Flexor & Extensor overlapping)
  drawWave(signals.emgFlex.data, COLORS.emg, midY * 1.5, midY * 0.35);
  drawWave(signals.emgExt.data, COLORS.emgExt, midY * 1.5, midY * 0.35);
}

// 2. Delta Modulator Tick marks (Spikes)
function drawSpikeStream() {
  const { ctx, el } = canvases.spike;
  if (!ctx) return;
  
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, el.width, el.height);

  const numChannels = 3;
  const channelHeight = el.height / numChannels;
  const stepX = el.width / BUFFER_SIZE;

  // Grid and Separators
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.06)';
  ctx.lineWidth = 1;
  for (let i = 1; i < numChannels; i++) {
    ctx.beginPath();
    ctx.moveTo(0, i * channelHeight);
    ctx.lineTo(el.width, i * channelHeight);
    ctx.stroke();
  }

  // Render Label text
  ctx.fillStyle = COLORS.textSecondary;
  ctx.font = '10px JetBrains Mono'; // Updated font
  ctx.fillText('EEG (Mu)', 8, 15);
  ctx.fillText('EMG Flex', 8, channelHeight + 15);
  ctx.fillText('EMG Ext', 8, channelHeight * 2 + 15);

  // Draw ticks for each spike channel
  const drawSpikesForChannel = (events, centerY) => {
    events.forEach(evt => {
      const x = evt.x * stepX;
      ctx.strokeStyle = evt.type === 'up' ? COLORS.spikeUp : COLORS.spikeDown;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, centerY - 12);
      ctx.lineTo(x, centerY + 12);
      ctx.stroke();
    });
  };

  drawSpikesForChannel(spikeEvents.eeg, channelHeight * 0.5);
  drawSpikesForChannel(spikeEvents.emgFlex, channelHeight * 1.5);
  drawSpikesForChannel(spikeEvents.emgExt, channelHeight * 2.5);
}

// 3. SNN Architecture Diagram Canvas (Sleek Minimal Vector representation)
function drawSNNCore() {
  const { ctx, el } = canvases.snn;
  if (!ctx) return;

  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, el.width, el.height);

  // Node Positions (x, y) coordinates
  const nodes = {
    // Inputs (Left Column)
    eeg: { x: el.width * 0.22, y: el.height * 0.25, label: 'EEG Mu', color: COLORS.eeg },
    emgFlex: { x: el.width * 0.22, y: el.height * 0.5, label: 'EMG Flex', color: COLORS.emg },
    emgExt: { x: el.width * 0.22, y: el.height * 0.75, label: 'EMG Ext', color: COLORS.emgExt },
    
    // Outputs (Right Column)
    grasp: { x: el.width * 0.78, y: el.height * 0.25, label: 'Grasp LIF', color: COLORS.spikeUp, lif: neurons.grasp },
    release: { x: el.width * 0.78, y: el.height * 0.5, label: 'Release LIF', color: COLORS.eeg, lif: neurons.release },
    spasm: { x: el.width * 0.78, y: el.height * 0.75, label: 'Spasm Alert', color: COLORS.spikeDown, lif: neurons.spasm }
  };

  // Draw Synaptic Connections (Thin outline vectors)
  const drawSynapseLine = (fromNode, toNode, baseWeight) => {
    const from = nodes[fromNode];
    const to = nodes[toNode];
    
    const isActiveWeight = baseWeight * sliders.synWeight.value;
    
    ctx.strokeStyle = isActiveWeight > 0 ? 'rgba(22, 163, 74, 0.18)' : 'rgba(220, 38, 38, 0.18)';
    ctx.lineWidth = Math.abs(isActiveWeight) * 1.5; // Thinner linkages
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  };

  // Map synapses weights
  drawSynapseLine('eeg', 'grasp', synapses.eegToGrasp);
  drawSynapseLine('emgFlex', 'grasp', synapses.emgFlexToGrasp);
  drawSynapseLine('emgExt', 'grasp', synapses.emgExtToGrasp);

  drawSynapseLine('eeg', 'release', synapses.eegToRelease);
  drawSynapseLine('emgFlex', 'release', synapses.emgFlexToRelease);
  drawSynapseLine('emgExt', 'release', synapses.emgExtToRelease);

  drawSynapseLine('emgFlex', 'spasm', synapses.emgFlexToSpasm);
  drawSynapseLine('emgExt', 'spasm', synapses.emgExtToSpasm);

  // Update and draw flowing Synapse Glow Packets (Small clean energy dots)
  activeSynapticPackets.forEach((packet, index) => {
    packet.progress += packet.speed;
    if (packet.progress >= 1.0) {
      activeSynapticPackets.splice(index, 1);
      return;
    }

    const from = nodes[packet.source];
    const to = nodes[packet.target];
    
    const currX = from.x + (to.x - from.x) * packet.progress;
    const currY = from.y + (to.y - from.y) * packet.progress;

    ctx.fillStyle = packet.color;
    ctx.beginPath();
    ctx.arc(currX, currY, 2.5, 0, 2 * Math.PI); // Smaller dots
    ctx.fill();
  });

  // Draw Node Circles (Hollow high-tech geometric style)
  for (let key in nodes) {
    const node = nodes[key];
    const now = performance.now();

    // Node Outline
    ctx.fillStyle = COLORS.bg;
    ctx.strokeStyle = node.color;
    ctx.lineWidth = 1.25;
    
    ctx.beginPath();
    ctx.arc(node.x, node.y, 22, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Center active core
    let sensorActive = false;
    if (key === 'eeg' && (attemptState === 'idle' || attemptState === 'spasm')) sensorActive = true;
    if (key === 'emgFlex' && (attemptState === 'grasp' || attemptState === 'spasm')) sensorActive = true;
    if (key === 'emgExt' && (attemptState === 'release' || attemptState === 'spasm')) sensorActive = true;

    if (sensorActive || (node.lif && now - node.lif.lastFireTime < 150)) {
      ctx.fillStyle = node.color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI); // Simple central indicator core
      ctx.fill();
    }

    // Node Text labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '10px Space Grotesk'; // Unique typography
    ctx.textAlign = 'center';
    ctx.fillText(node.label, node.x, node.y + 4);
  }
}

// 4. LIF Neurons Membrane Potential Canvas
function drawMembranePotentials() {
  const { ctx, el } = canvases.potentials;
  if (!ctx) return;

  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, el.width, el.height);

  // Draw threshold line (V_th)
  const threshVal = sliders.neuronThresh.value;
  
  const minV = -0.2;
  const maxV = Math.max(2.0, threshVal + 0.3);
  
  const getCanvasY = (v) => {
    const norm = (v - minV) / (maxV - minV);
    return el.height - norm * el.height * 0.85 - el.height * 0.08;
  };

  const threshY = getCanvasY(threshVal);

  // Draw dashed threshold line
  ctx.strokeStyle = COLORS.amber;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, threshY);
  ctx.lineTo(el.width, threshY);
  ctx.stroke();
  ctx.setLineDash([]); // reset

  ctx.fillStyle = COLORS.amber;
  ctx.font = '10px JetBrains Mono'; // Clean mono typography
  ctx.fillText(`Vth: ${threshVal.toFixed(2)}V`, el.width - 75, threshY - 6);

  // Plot Membrane potential functions
  const drawPotentialLine = (lifNeuron, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25; // Sleeker
    ctx.beginPath();

    const stepX = el.width / BUFFER_SIZE;
    for (let i = 0; i < BUFFER_SIZE; i++) {
      const x = i * stepX;
      const y = getCanvasY(lifNeuron.vHistory[i]);
      
      if (i > 0 && lifNeuron.vHistory[i-1] > threshVal * 0.9 && lifNeuron.vHistory[i] <= 0.05) {
        ctx.lineTo(x, getCanvasY(threshVal));
        ctx.moveTo(x, getCanvasY(0));
      } else {
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  };

  drawPotentialLine(neurons.grasp, COLORS.spikeUp);
  drawPotentialLine(neurons.release, COLORS.eeg);

  // Title tags
  ctx.fillStyle = COLORS.textSecondary;
  ctx.font = '10px Space Grotesk';
  ctx.fillText('Membrane Potentials V_m(t)', 10, 15);
  ctx.fillStyle = COLORS.spikeUp;
  ctx.fillText('■ Grasp Neuron', 10, 30);
  ctx.fillStyle = COLORS.eeg;
  ctx.fillText('■ Release Neuron', 10, 42);
}

// 5. Exoskeleton Hand Actuator 2D Mechanical Visualizer (Holographic Wireframe Blueprint)
function drawRoboticExoskeleton() {
  const { ctx, el } = canvases.hand;
  if (!ctx) return;

  // Clean with high-tech deep background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, el.width, el.height);

  const wristX = el.width / 2;
  const wristY = el.height * 0.88;
  const palmHeight = el.height * 0.35;
  const palmWidth = el.width * 0.32;
  
  handClosure += (currentHandClosureTarget - handClosure) * 0.12;
  
  let spasmX = 0;
  let spasmY = 0;
  if (attemptState === 'spasm') {
    spasmX = (Math.random() - 0.5) * jitterMagnitude;
    spasmY = (Math.random() - 0.5) * jitterMagnitude;
  }

  // Draw Exoskeleton Vector Linkages (Fine glowing wires instead of thick rods)
  const drawMechanicalLink = (x1, y1, x2, y2, color) => {
    ctx.strokeStyle = color || 'rgba(37, 99, 235, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1 + spasmX, y1 + spasmY);
    ctx.lineTo(x2 + spasmX, y2 + spasmY);
    ctx.stroke();

    // Subtle vector parallel line to create hollow technical frame look
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.04)';
    ctx.lineWidth = 4;
    ctx.stroke();
  };

  // Draw Hollow Joint Circles (Technical blueprint aesthetic)
  const drawJoint = (x, y, radius, activeColor) => {
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = activeColor || 'rgba(15, 23, 42, 0.2)';
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.arc(x + spasmX, y + spasmY, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Tiny center pivot point
    ctx.fillStyle = activeColor || '#0f172a';
    ctx.beginPath();
    ctx.arc(x + spasmX, y + spasmY, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  };

  // Calculate Finger Joint Positions based on Hand Closure Angle
  const drawFinger = (baseX, baseY, length, angleOffset, jointSpreadFactor) => {
    const phal1Len = length * 0.42;
    const phal2Len = length * 0.33;
    const phal3Len = length * 0.25;

    const baseAngle = -Math.PI / 2 + angleOffset * jointSpreadFactor;
    const angle1 = baseAngle + (handClosure * 1.35);
    const angle2 = angle1 + (handClosure * 1.15);
    const angle3 = angle2 + (handClosure * 0.85);

    const pipX = baseX + Math.cos(angle1) * phal1Len;
    const pipY = baseY + Math.sin(angle1) * phal1Len;

    const dipX = pipX + Math.cos(angle2) * phal2Len;
    const dipY = pipY + Math.sin(angle2) * phal2Len;

    const tipX = dipX + Math.cos(angle3) * phal3Len;
    const tipY = dipY + Math.sin(angle3) * phal3Len;

    const lineColor = attemptState === 'spasm' ? COLORS.spikeDown : (handClosure > 0.6 ? COLORS.spikeUp : COLORS.eeg);

    // Link structural lines
    drawMechanicalLink(baseX, baseY, pipX, pipY, lineColor);
    drawMechanicalLink(pipX, pipY, dipX, dipY, lineColor);
    drawMechanicalLink(dipX, dipY, tipX, tipY, lineColor);

    // Joint caps
    drawJoint(baseX, baseY, 5, lineColor);
    drawJoint(pipX, pipY, 4, lineColor);
    drawJoint(dipX, dipY, 3, lineColor);
    drawJoint(tipX, tipY, 2, lineColor);
  };

  // Render Palm Plate as a fine wireframe outline
  const strokeColor = attemptState === 'spasm' ? COLORS.spikeDown : 'rgba(15, 23, 42, 0.15)';
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(wristX - palmWidth * 0.35 + spasmX, wristY + spasmY);
  ctx.lineTo(wristX + palmWidth * 0.35 + spasmX, wristY + spasmY);
  ctx.lineTo(wristX + palmWidth * 0.45 + spasmX, wristY - palmHeight + spasmY);
  ctx.lineTo(wristX - palmWidth * 0.45 + spasmX, wristY - palmHeight + spasmY);
  ctx.closePath();
  ctx.stroke();

  // Internal visual grid pattern on palm plate to emphasize "Blueprint/Hologram" look
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.02)';
  ctx.lineWidth = 1;
  for (let offset = 10; offset < palmHeight; offset += 15) {
    ctx.beginPath();
    ctx.moveTo(wristX - palmWidth * 0.4 + spasmX, wristY - offset + spasmY);
    ctx.lineTo(wristX + palmWidth * 0.4 + spasmX, wristY - offset + spasmY);
    ctx.stroke();
  }

  // Core mechanical central linkage line
  drawMechanicalLink(wristX, wristY, wristX, wristY - palmHeight * 0.9, strokeColor);
  
  const palmTopY = wristY - palmHeight;
  const spreadX = palmWidth * 0.8;
  const knuckleY = palmTopY;

  // Thumb
  drawFinger(wristX - palmWidth * 0.38, wristY - palmHeight * 0.45, 52, -0.65, 0.6);
  // Index
  drawFinger(wristX - spreadX * 0.35, knuckleY, 82, -0.22, 0.3);
  // Middle
  drawFinger(wristX - spreadX * 0.05, knuckleY, 88, -0.02, 0.1);
  // Ring
  drawFinger(wristX + spreadX * 0.25, knuckleY, 84, 0.16, 0.3);
  // Pinky
  drawFinger(wristX + spreadX * 0.52, knuckleY, 72, 0.34, 0.5);

  // Wrist joint base hollow drawing
  drawJoint(wristX, wristY, 8, attemptState === 'spasm' ? COLORS.spikeDown : COLORS.eeg);

  // Spasm Emergency Override indicator overlay
  if (attemptState === 'spasm') {
    ctx.fillStyle = 'rgba(255, 51, 102, 0.04)';
    ctx.fillRect(0, 0, el.width, el.height);
    
    ctx.fillStyle = COLORS.spikeDown;
    ctx.font = 'bold 10px Space Grotesk';
    ctx.textAlign = 'center';
    ctx.fillText('WARNING: PERIPHERAL SPASM DETECTED', el.width / 2, 35);
    ctx.fillText('SANI OVERRIDE ACTIVE - ACTUATORS OFF', el.width / 2, 48);
  }

  // Update DOM feedback progress bar & status indicators dynamically
  const handPct = Math.round(handClosure * 100);
  const valHandClosureEl = document.getElementById('valHandClosure');
  const fillHandClosureEl = document.getElementById('fillHandClosure');
  if (valHandClosureEl) valHandClosureEl.innerText = `${handPct}%`;
  if (fillHandClosureEl) fillHandClosureEl.style.width = `${handPct}%`;

  const valActuatorStateEl = document.getElementById('valActuatorState');
  if (valActuatorStateEl) {
    if (attemptState === 'spasm') {
      valActuatorStateEl.innerText = 'LOCKED (SPASM OVERRIDE)';
      valActuatorStateEl.style.color = 'var(--color-spike-down)';
    } else if (currentHandClosureTarget === 1.0 && handClosure < 0.95) {
      valActuatorStateEl.innerText = 'CLOSING (GRASP ATTEMPT)...';
      valActuatorStateEl.style.color = 'var(--color-spike-up)';
    } else if (currentHandClosureTarget === 1.0 && handClosure >= 0.95) {
      valActuatorStateEl.innerText = 'CLOSED (GRASP DETECTED)';
      valActuatorStateEl.style.color = 'var(--color-spike-up)';
    } else if (currentHandClosureTarget === 0.0 && handClosure > 0.05) {
      valActuatorStateEl.innerText = 'OPENING (RELEASE ATTEMPT)...';
      valActuatorStateEl.style.color = 'var(--color-eeg)';
    } else if (currentHandClosureTarget === 0.0 && handClosure <= 0.05) {
      valActuatorStateEl.innerText = 'OPEN (RELEASE DETECTED)';
      valActuatorStateEl.style.color = 'var(--color-eeg)';
    } else {
      valActuatorStateEl.innerText = 'STANDBY (RESTING)';
      valActuatorStateEl.style.color = 'var(--text-secondary)';
    }
  }
}

// --- Main Real-time Animation Loop ---
let lastTime = 0;
let simTimeMs = 0;

function loop(timestamp) {
  if (!lastTime) lastTime = timestamp;
  const elapsed = timestamp - lastTime;
  lastTime = timestamp;

  // Run SNN simulation step multiple times per frame for step-accuracy
  for (let step = 0; step < SIM_TICK_RATE; step++) {
    simTimeMs += DT;
    stepSNN(simTimeMs);
  }

  // Draw graphics frames
  drawOscilloscope();
  drawSpikeStream();
  drawSNNCore();
  drawMembranePotentials();
  drawRoboticExoskeleton();

  // Benchmarks calculations
  updateEdgeBenchmarks(timestamp);

  // Live Telemetry Interpreter updates
  updateLiveTelemetryInterpreter();

  requestAnimationFrame(loop);
}

// Automatic PDF Diagnostic Report Generator (Using jsPDF)
function generatePDFReport() {
  const { jsPDF } = window.jspdf;
  if (!jsPDF) {
    alert("jsPDF library not loaded. Please check network connection.");
    return;
  }

  const doc = new jsPDF();
  const profile = PROFILES[profileState];
  const now = new Date();
  const dateStr = now.toLocaleDateString() + " " + now.toLocaleTimeString();

  // --- Document Header ---
  // Dark blue banner top accent
  doc.setFillColor(37, 99, 235); // Medical Blue Accent
  doc.rect(0, 0, 210, 12, 'F');

  // Title branding
  doc.setTextColor(15, 23, 42); // Slate 900
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(18);
  doc.text("NEURO-NEX REHABILITATION REPORT", 20, 26);
  
  doc.setFont("Helvetica", "italic");
  doc.setFontSize(9);
  doc.setTextColor(71, 85, 105); // Slate 600
  doc.text("Stroke Assistive Neuromorphic Interface (SANI) - Patient Analytics", 20, 31);
  
  // Thin separator
  doc.setDrawColor(226, 232, 240); // Slate 200
  doc.line(20, 34, 190, 34);

  // --- Patient & Demographics Information Card ---
  doc.setFillColor(248, 250, 252); // Slate 50
  doc.rect(20, 39, 170, 26, 'F');
  doc.setDrawColor(226, 232, 240);
  doc.rect(20, 39, 170, 26, 'S');

  doc.setFont("Helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(15, 23, 42);
  doc.text("PATIENT DEMOGRAPHICS", 24, 45);
  doc.text("CLINICAL PROTOCOL", 110, 45);

  doc.setFont("Helvetica", "normal");
  doc.setTextColor(71, 85, 105);
  doc.text(`Patient ID:     NX-942`, 24, 50);
  doc.text(`Age / Gender:   64 / Male`, 24, 55);
  doc.text(`Report Date:    ${dateStr}`, 24, 60);

  doc.text(`Recovery Profile: ${profile.name}`, 110, 50);
  doc.text(`Condition Phase:  Subacute Rehabilitation`, 110, 55);
  doc.text(`Device Hardware:  SANI Exoskeleton Hand v2`, 110, 60);

  // --- Section 1: Neuromorphic Core Calibration Settings ---
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(37, 99, 235);
  doc.text("1. NEUROMORPHIC EDGE PARAMETERS", 20, 76);
  doc.line(20, 78, 190, 78);

  doc.setFont("Helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(71, 85, 105);
  
  const dThresh = sliders.deltaThresh.value.toFixed(2);
  const nThresh = sliders.neuronThresh.value.toFixed(2);
  const lRate = sliders.leakRate.value;
  const sWeight = sliders.synWeight.value.toFixed(2);

  doc.text(`Delta Mod. Encoder Threshold (theta):   ${dThresh} V`, 24, 84);
  doc.text(`LIF Neuron Firing Threshold (Vth):     ${nThresh} V`, 24, 90);
  doc.text(`Leaky Membrane Decay Rate (tau):       ${lRate} ms`, 110, 84);
  doc.text(`Synaptic Weight Amplification multiplier: ${sWeight}`, 110, 90);

  // --- Section 2: Rehabilitation Session Stats ---
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(37, 99, 235);
  doc.text("2. SESSION CLINICAL DIAGNOSTICS", 20, 105);
  doc.line(20, 107, 190, 107);

  // Counters Grid Card layout
  doc.setFillColor(248, 250, 252);
  doc.rect(20, 111, 170, 22, 'F');
  doc.rect(20, 111, 170, 22, 'S');

  doc.setFont("Helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(15, 23, 42);
  doc.text("SUCCESSFUL GRASPS", 24, 117);
  doc.text("SUCCESSFUL RELEASES", 68, 117);
  doc.text("SPASMS INTERCEPTED", 112, 117);
  doc.text("CLASSIFICATION PRECISION", 152, 117);

  doc.setFont("Helvetica", "bold");
  doc.setFontSize(12);
  doc.setTextColor(37, 99, 235);
  doc.text(`${graspCount}`, 38, 126);
  doc.text(`${releaseCount}`, 84, 126);
  doc.setTextColor(220, 38, 38);
  doc.text(`${spasmCount}`, 126, 126);
  doc.setTextColor(22, 163, 74);
  doc.text("94.2%", 164, 126);

  // --- Section 3: Hardware Edge Benchmarks ---
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(37, 99, 235);
  doc.text("3. EDGE HARDWARE DIAGNOSTICS & BENCHMARKS", 20, 144);
  doc.line(20, 146, 190, 146);

  doc.setFont("Helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(71, 85, 105);

  const avgPowerStr = document.getElementById('metricPower').innerText;
  const latStr = document.getElementById('metricLatency').innerText;
  const sopsStr = document.getElementById('metricSops').innerText;
  const spkF = document.getElementById('metricSpikerate').innerText;

  doc.text(`Estimated Active Edge Power:          ${avgPowerStr}`, 24, 152);
  doc.text(`Neuromorphic Latency (First Spike):   ${latStr}`, 24, 158);
  doc.text(`Synaptic Operations Rate (SOPs):      ${sopsStr}`, 110, 152);
  doc.text(`Delta Event Spike Frequency:          ${spkF}`, 110, 158);

  // --- Section 4: Clinical Diagnostic Assessment ---
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(37, 99, 235);
  doc.text("4. TREATMENT & DIAGNOSTIC ASSESSMENT", 20, 172);
  doc.line(20, 174, 190, 174);

  // Dynamic clinical assessment texts based on profileState
  let assessmentText = "";
  if (profileState === 'healthy') {
    assessmentText = "A healthy control benchmark was established. The patient demonstrated immediate cortical intent transmission, yielding consistent, high-amplitude EMG and rapid EEG desynchronization. Decoding latency was under 6.5 milliseconds, verifying responsive closed-loop motor trigger dynamics. Actuator tracking shows optimal mechanical flexion. No anomalies or muscle spasms were registered during active blocks.";
  } else if (profileState === 'mild') {
    assessmentText = "Rehab trials indicate moderate hemiparesis. Cortical EEG Mu desynchronization desynchronized during motor imagery attempts, indicating intact motor pathway planning. EMG signals displayed minor attenuation but integrated cleanly within SNN nodes, triggering hand exoskeleton flexion/extension with minimal latency (under 8ms). Active robotic support is sufficient. Recommending continued rehabilitation exercises to encourage further cortical neuroplasticity.";
  } else if (profileState === 'severe') {
    assessmentText = "Rehab trials indicate severe flaccid paralysis. Cortical EEG desynchronization was minimal, representing heavily reduced cortical recruitment. Forearm EMG signals were deeply attenuated. Synaptic weights were increased (+1.5) and neuron threshold reduced (0.80V) to successfully resolve intent. Exoskeleton mechanical power is critical for thumb/fingers execution. Recommending continued passive-to-active movement blocks.";
  } else if (profileState === 'spastic') {
    assessmentText = "Patient exhibits spasticity and peripheral co-contraction reflexes. Forearm sensors recorded frequent high-frequency muscle bursts without conscious cortical intent. SANI's integrated co-contraction filters successfully mapped this to the Spasm Alert LIF neuron, immediately disabling exoskeleton actuators. Motors were safely locked to prevent muscle damage. Recommendation: calibrate Delta modulation threshold higher to filter background spastic noise.";
  }

  doc.setFont("Helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(15, 23, 42);
  const wrappedLines = doc.splitTextToSize(assessmentText, 170);
  doc.text(wrappedLines, 20, 180);

  // --- Section 5: Signature Fields ---
  doc.setDrawColor(226, 232, 240);
  doc.line(20, 222, 190, 222);

  doc.setFont("Helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(71, 85, 105);
  
  // Treating engineer
  doc.line(24, 252, 94, 252);
  doc.text("Treating Rehabilitation Engineer", 24, 256);
  doc.text("Neuro-Nex Clinical Center", 24, 260);

  // Technician
  doc.line(116, 252, 186, 252);
  doc.text("Treating Clinical Technician", 116, 256);
  doc.text("Certified Neuro-Technologist", 116, 260);

  // Disclaimer
  doc.setFont("Helvetica", "italic");
  doc.setFontSize(7.5);
  doc.setTextColor(148, 163, 184); // Slate 400
  doc.text("Disclaimer: This report is generated by the SANI Neuromorphic ASIC diagnostic emulator for clinical assessment purposes.", 20, 275);
  doc.text("It is intended for use by trained medical practitioners in neurological rehabilitation contexts.", 20, 279);

  // Save PDF
  doc.save(`SANI_Clinical_Report_NX942_${profileState}.pdf`);
  logSystemEvent(`System Report: Clinical PDF report generated and downloaded successfully.`, 'system');
}
