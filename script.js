// ==========================================
// CIRCUITLENS LOGIC
// ==========================================

const LABELS = {
    0: { code: 'X1', name: 'AC Voltage Source' },
    1: { code: 'X10', name: 'Fuse' },
    2: { code: 'X11', name: 'Resistor' },
    3: { code: 'X12', name: 'Variable Resistor' },
    4: { code: 'X13', name: 'Inductor' },
    5: { code: 'X14', name: 'Variable Inductor' },
    6: { code: 'X15', name: 'Polarized Capacitor' },
    7: { code: 'X16', name: 'Capacitor' },
    8: { code: 'X17', name: 'Variable Capacitor' },
    9: { code: 'X2', name: 'Current Source' },
    10: { code: 'X3', name: 'Dependent Voltage Source' },
    11: { code: 'X4', name: 'Dependent Current Source' },
    12: { code: 'X5', name: 'DC Voltage Source' },
    13: { code: 'X6', name: 'Cell' },
    14: { code: 'X7', name: 'Battery' },
    15: { code: 'X8', name: 'Ground' },
    16: { code: 'X9', name: 'Switch' }
};

let session;
let isCameraRunning = false;
let isProcessing = false;
let currentModelName = 'MobileNet V3'; 
const MODEL_PATH_BASE = './models/';

// DOM Elements
const videoElement = document.getElementById('webcam');
const statusDot = document.getElementById('model-status');
const resultOverlay = document.getElementById('result-overlay');
const fileInput = document.getElementById('file-input');
const dropArea = document.getElementById('drop-area');
const imageWrapper = document.getElementById('image-wrapper');
const uploadedImg = document.getElementById('uploaded-image');
const resetBtn = document.getElementById('reset-upload-btn');
const predModel = document.getElementById('pred-model');

// --- DROPDOWN & RE-INFERENCE LOGIC ---
const dropdown = document.getElementById('custom-dropdown');
const selectedText = document.getElementById('selected-model-text');
const options = document.querySelectorAll('.option');

dropdown.addEventListener('click', () => dropdown.classList.toggle('open'));

options.forEach(option => {
    option.addEventListener('click', async (e) => { // Marked as async
        const val = e.target.getAttribute('data-value');
        const text = e.target.innerText;
        
        // Update UI Text
        selectedText.innerText = text;
        currentModelName = text;
        
        // Handle selection state
        options.forEach(o => o.classList.remove('selected'));
        e.target.classList.add('selected');
        
        // 1. Load the new model
        await loadModel(val);

        // 2. CHECK: If an image is currently uploaded, re-run inference immediately
        if (!imageWrapper.classList.contains('hidden') && uploadedImg.src) {
            // Visual feedback that we are re-scanning
            document.getElementById('pred-label').innerText = "Updating...";
            document.getElementById('pred-model').innerText = "Loading...";
            
            // Run inference with the new session
            await runInference(uploadedImg);
        }
    });
});

// Close dropdown if clicked outside
document.addEventListener('click', (e) => {
    if (!dropdown.contains(e.target)) dropdown.classList.remove('open');
});


// 1. LOAD MODEL
async function loadModel(modelFileName) {
    statusDot.classList.remove('active');
    statusDot.title = "Loading...";
    try {
        // Create new session
        session = await ort.InferenceSession.create(`${MODEL_PATH_BASE}${modelFileName}.onnx`, { executionProviders: ['wasm'] });
        statusDot.classList.add('active');
        statusDot.title = "Ready";
    } catch (e) {
        console.error("Model Error:", e);
        alert("Failed to load model: " + modelFileName);
    }
}
// Initial Load
loadModel('mobilenet_v3'); 


// 2. VIEW SWITCHING
window.switchTab = (tab) => {
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view-mode').forEach(v => v.classList.remove('active'));
    resultOverlay.classList.add('hidden');

    if (tab === 'camera') {
        document.getElementById('btn-cam').classList.add('active');
        document.getElementById('camera-view').classList.add('active');
    } else {
        document.getElementById('btn-upload').classList.add('active');
        document.getElementById('upload-view').classList.add('active');
        stopCamera();
    }
};


// 3. CAMERA
async function startCamera() {
    if (isCameraRunning) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        videoElement.srcObject = stream;
        isCameraRunning = true;
        document.getElementById('toggle-cam-btn').innerHTML = '<i class="fa-solid fa-stop"></i> Stop';
        requestAnimationFrame(processCameraFrame);
    } catch (err) {
        alert("Camera Error: " + err);
    }
}

function stopCamera() {
    const stream = videoElement.srcObject;
    if (stream) stream.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
    isCameraRunning = false;
    document.getElementById('toggle-cam-btn').innerHTML = '<i class="fa-solid fa-play"></i> Start Camera';
}

document.getElementById('toggle-cam-btn').addEventListener('click', () => {
    if (isCameraRunning) stopCamera(); else startCamera();
});


// 4. INFERENCE LOOP (CAMERA)
async function processCameraFrame() {
    if (!isCameraRunning) return;
    if (!isProcessing) {
        isProcessing = true;
        await runInference(videoElement);
        setTimeout(() => { isProcessing = false; }, 200);
    }
    requestAnimationFrame(processCameraFrame);
}


// 5. UPLOAD & RESET
dropArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
            uploadedImg.src = evt.target.result;
            dropArea.style.display = 'none';
            imageWrapper.classList.remove('hidden'); 
            
            // Run inference once image loads
            uploadedImg.onload = () => runInference(uploadedImg);
        };
        reader.readAsDataURL(file);
    }
});

resetBtn.addEventListener('click', () => {
    uploadedImg.src = '';
    imageWrapper.classList.add('hidden'); 
    dropArea.style.display = 'block';     
    resultOverlay.classList.add('hidden');
    fileInput.value = '';
});


// 6. INFERENCE CORE
async function preprocessImage(imageSource) {
    const width = 224, height = 224;
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageSource, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height).data;
    
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const float32Data = new Float32Array(3 * width * height);
    
    for (let i = 0; i < width * height; i++) {
        float32Data[i] = ((imageData[i * 4] / 255) - mean[0]) / std[0];
        float32Data[i + width * height] = ((imageData[i * 4 + 1] / 255) - mean[1]) / std[1];
        float32Data[i + 2 * width * height] = ((imageData[i * 4 + 2] / 255) - mean[2]) / std[2];
    }
    return new ort.Tensor('float32', float32Data, [1, 3, width, height]);
}

async function runInference(imageSource) {
    if (!session) return;
    const startTime = performance.now();
    try {
        const inputTensor = await preprocessImage(imageSource);
        const results = await session.run({ input: inputTensor });
        const probs = softmax(results.output.data);
        const maxIdx = probs.indexOf(Math.max(...probs));
        updateUI(maxIdx, (probs[maxIdx] * 100).toFixed(1), (performance.now() - startTime).toFixed(0));
    } catch (e) { console.error(e); }
}

function softmax(arr) {
    return arr.map(val => Math.exp(val) / arr.map(Math.exp).reduce((a, b) => a + b));
}

function updateUI(classIdx, confidence, time) {
    const label = LABELS[classIdx];
    resultOverlay.classList.remove('hidden');
    document.getElementById('pred-label').innerText = label.name;
    document.getElementById('pred-code').innerText = label.code;
    predModel.innerText = currentModelName;
    document.getElementById('conf-score').innerText = confidence;
    document.getElementById('time-taken').innerText = time;
    
    const title = document.getElementById('pred-label');
    title.style.color = confidence < 50 ? 'var(--text-sub)' : 'var(--accent-dark)';
    if(confidence < 50) title.innerText = "Unsure...";
}