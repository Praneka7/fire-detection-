/**
 * PYROGUARD - FIRE SURVEILLANCE & EARLY WARNING DASHBOARD
 * Frontend Engine & Web Audio Alert Synthesizer
 */

(() => {
    'use strict';

    // State
    let soundEnabled = true;
    let audioCtx = null;
    let sirenOsc = null;
    let sirenGain = null;
    let sirenInterval = null;
    let currentMode = 'processed';
    let previousAlertCount = 0;
    let configDebounceTimer = null;
    let currentModalFilename = '';
    let isDeviceWebcamActive = false;
    let webcamStream = null;
    let webcamProcessing = false;
    let webcamInterval = null;

    // DOM Elements
    const elements = {
        clockDisplay: document.getElementById('clockDisplay'),
        systemPill: document.getElementById('systemPill'),
        systemStatusText: document.getElementById('systemStatusText'),
        soundToggleBtn: document.getElementById('soundToggleBtn'),
        soundIconOn: document.getElementById('soundIconOn'),
        soundIconOff: document.getElementById('soundIconOff'),
        soundLabel: document.getElementById('soundLabel'),
        emergencyBanner: document.getElementById('emergencyBanner'),
        cameraStream: document.getElementById('cameraStream'),
        localVideo: document.getElementById('localVideo'),
        localCanvas: document.getElementById('localCanvas'),
        cloudCameraPrompt: document.getElementById('cloudCameraPrompt'),
        startWebcamCta: document.getElementById('startWebcamCta'),
        webcamToggleBtn: document.getElementById('webcamToggleBtn'),
        webcamIconStart: document.getElementById('webcamIconStart'),
        webcamIconStop: document.getElementById('webcamIconStop'),
        webcamBtnLabel: document.getElementById('webcamBtnLabel'),
        streamLoading: document.getElementById('streamLoading'),
        fpsDisplay: document.getElementById('fpsDisplay'),
        modeProcessedBtn: document.getElementById('modeProcessedBtn'),
        modeCleanBtn: document.getElementById('modeCleanBtn'),
        modeMaskBtn: document.getElementById('modeMaskBtn'),
        manualSnapBtn: document.getElementById('manualSnapBtn'),
        reloadStreamBtn: document.getElementById('reloadStreamBtn'),
        fullscreenBtn: document.getElementById('fullscreenBtn'),
        videoStage: document.getElementById('videoStage'),
        streamTag: document.getElementById('streamTag'),
        alertStatusCard: document.getElementById('alertStatusCard'),
        mainStatusText: document.getElementById('mainStatusText'),
        bufferCountDisplay: document.getElementById('bufferCountDisplay'),
        confirmProgressBar: document.getElementById('confirmProgressBar'),
        regionsCountDisplay: document.getElementById('regionsCountDisplay'),
        flameAreaDisplay: document.getElementById('flameAreaDisplay'),
        totalAlertsDisplay: document.getElementById('totalAlertsDisplay'),
        lastAlertDisplay: document.getElementById('lastAlertDisplay'),
        minAreaSlider: document.getElementById('minAreaSlider'),
        minAreaBadge: document.getElementById('minAreaBadge'),
        confirmFramesSlider: document.getElementById('confirmFramesSlider'),
        confirmFramesBadge: document.getElementById('confirmFramesBadge'),
        cooldownSlider: document.getElementById('cooldownSlider'),
        cooldownBadge: document.getElementById('cooldownBadge'),
        resetDefaultsBtn: document.getElementById('resetDefaultsBtn'),
        snapshotGrid: document.getElementById('snapshotGrid'),
        snapshotsCount: document.getElementById('snapshotsCount'),
        refreshGalleryBtn: document.getElementById('refreshGalleryBtn'),
        clearGalleryBtn: document.getElementById('clearGalleryBtn'),
        snapshotModal: document.getElementById('snapshotModal'),
        modalImage: document.getElementById('modalImage'),
        modalMeta: document.getElementById('modalMeta'),
        modalCloseBtn: document.getElementById('modalCloseBtn'),
        modalOverlay: document.getElementById('modalOverlay'),
        modalDeleteBtn: document.getElementById('modalDeleteBtn'),
        modalDownloadBtn: document.getElementById('modalDownloadBtn'),
        toastContainer: document.getElementById('toastContainer'),
    };

    // =========================================================
    // WEB AUDIO ALARM SYNTHESIZER
    // =========================================================
    function initAudio() {
        if (!audioCtx) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (AudioContextClass) {
                audioCtx = new AudioContextClass();
            }
        }
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
    }

    function startSiren() {
        if (!soundEnabled || sirenOsc || !audioCtx) return;

        try {
            sirenOsc = audioCtx.createOscillator();
            sirenGain = audioCtx.createGain();

            sirenOsc.type = 'sawtooth';
            sirenOsc.frequency.setValueAtTime(880, audioCtx.currentTime);

            sirenGain.gain.setValueAtTime(0.15, audioCtx.currentTime);

            sirenOsc.connect(sirenGain);
            sirenGain.connect(audioCtx.destination);
            sirenOsc.start();

            // Modulate pitch between 880Hz and 1300Hz
            let high = true;
            sirenInterval = setInterval(() => {
                if (!sirenOsc || !audioCtx) return;
                const freq = high ? 1300 : 880;
                sirenOsc.frequency.setTargetAtTime(freq, audioCtx.currentTime, 0.08);
                high = !high;
            }, 180);
        } catch (e) {
            console.warn('Audio alarm could not start:', e);
        }
    }

    function stopSiren() {
        if (sirenInterval) {
            clearInterval(sirenInterval);
            sirenInterval = null;
        }
        if (sirenOsc) {
            try {
                sirenOsc.stop();
                sirenOsc.disconnect();
            } catch (e) {}
            sirenOsc = null;
        }
    }

    function toggleSound() {
        initAudio();
        soundEnabled = !soundEnabled;
        elements.soundToggleBtn.classList.toggle('active', soundEnabled);
        elements.soundIconOn.classList.toggle('hidden', !soundEnabled);
        elements.soundIconOff.classList.toggle('hidden', soundEnabled);
        elements.soundLabel.textContent = soundEnabled ? 'SIREN ON' : 'MUTED';
        if (!soundEnabled) {
            stopSiren();
        }
        showToast(soundEnabled ? 'Alarm siren enabled' : 'Alarm siren muted');
    }

    // =========================================================
    // TELEMETRY & POLLING ENGINE
    // =========================================================
    async function pollStatus() {
        try {
            const res = await fetch('/api/status');
            if (!res.ok) throw new Error('Status request failed');
            const data = await res.json();
            updateDashboard(data);
        } catch (err) {
            elements.systemPill.classList.remove('alert-active');
            elements.systemStatusText.textContent = 'OFFLINE';
            elements.systemPill.style.color = '#ef4444';
        }
    }

    function updateDashboard(data) {
        // Clock
        if (data.server_time) {
            elements.clockDisplay.textContent = data.server_time;
        }

        // System pill & FPS
        elements.systemStatusText.textContent = data.is_alert ? 'FIRE ALARM ACTIVE' : 'SYSTEM ONLINE';
        elements.systemPill.classList.toggle('alert-active', data.is_alert);
        elements.fpsDisplay.textContent = (data.fps || 0).toFixed(1);

        // Alert state visual cues
        elements.emergencyBanner.classList.toggle('hidden', !data.is_alert);
        elements.alertStatusCard.classList.toggle('state-alert', data.is_alert);
        elements.mainStatusText.textContent = data.is_alert ? '🔥 FIRE ALERT DETECTED' : 'MONITORING SECURE';

        // Confirmation progress meter
        const countText = `${data.candidate_frames} / ${data.confirm_frames} FRAMES`;
        elements.bufferCountDisplay.textContent = countText;
        elements.confirmProgressBar.style.width = `${data.confirm_percent}%`;

        // Telemetry cards
        elements.regionsCountDisplay.textContent = data.regions_count;
        elements.flameAreaDisplay.innerHTML = `${data.total_flame_area.toLocaleString()} <small>px²</small>`;
        elements.totalAlertsDisplay.textContent = data.total_alerts;
        elements.lastAlertDisplay.textContent = data.last_alert_timestamp || 'None Recorded';

        // Cloud standby prompt banner
        if (elements.cloudCameraPrompt) {
            if (data.is_cloud_standby && !isDeviceWebcamActive) {
                elements.cloudCameraPrompt.classList.remove('hidden');
            } else {
                elements.cloudCameraPrompt.classList.add('hidden');
            }
        }

        // Sound Siren Trigger
        if (data.is_alert) {
            initAudio();
            startSiren();
        } else {
            stopSiren();
        }

        // New snapshots auto-detection
        if (data.total_alerts > previousAlertCount) {
            previousAlertCount = data.total_alerts;
            loadSnapshots();
            if (data.is_alert) {
                showToast('🚨 New fire incident snapshot captured!');
            }
        }
    }

    // =========================================================
    // CLIENT DEVICE WEBCAM (HTML5 USERMEDIA)
    // =========================================================
    async function startDeviceWebcam() {
        initAudio();
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                showToast('Webcam not supported or browser requires HTTPS.');
                return;
            }

            elements.streamLoading.classList.remove('hidden');
            showToast('Opening device camera...');

            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 960 },
                    height: { ideal: 540 },
                    facingMode: 'user',
                },
                audio: false,
            });

            elements.localVideo.srcObject = webcamStream;
            await elements.localVideo.play();

            isDeviceWebcamActive = true;
            if (elements.cloudCameraPrompt) {
                elements.cloudCameraPrompt.classList.add('hidden');
            }
            elements.webcamToggleBtn.classList.add('active');
            elements.webcamIconStart.classList.add('hidden');
            elements.webcamIconStop.classList.remove('hidden');
            elements.webcamBtnLabel.textContent = 'Stop Device Camera';
            elements.streamTag.textContent = 'DEVICE WEBCAM // OPTICAL SENSOR';
            elements.streamLoading.classList.add('hidden');
            showToast('Device webcam active & analyzing for fire!');

            const canvas = elements.localCanvas;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });

            clearInterval(webcamInterval);
            webcamInterval = setInterval(async () => {
                if (!isDeviceWebcamActive || webcamProcessing) return;
                if (!elements.localVideo.videoWidth) return;

                webcamProcessing = true;
                try {
                    const targetW = 640;
                    const targetH = Math.round((elements.localVideo.videoHeight / elements.localVideo.videoWidth) * targetW) || 360;
                    canvas.width = targetW;
                    canvas.height = targetH;
                    ctx.drawImage(elements.localVideo, 0, 0, targetW, targetH);

                    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                    const res = await fetch('/api/process_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image: dataUrl,
                            mode: currentMode,
                        }),
                    });

                    if (res.ok) {
                        const result = await res.json();
                        if (result.image && isDeviceWebcamActive) {
                            elements.cameraStream.src = result.image;
                        }
                        updateDashboard(result);
                    }
                } catch (err) {
                    console.error('Frame processing error:', err);
                } finally {
                    webcamProcessing = false;
                }
            }, 85);

        } catch (err) {
            elements.streamLoading.classList.add('hidden');
            console.error('Camera access error:', err);
            showToast(`Camera permission error: ${err.message || 'Access denied'}`);
        }
    }

    function stopDeviceWebcam() {
        isDeviceWebcamActive = false;
        clearInterval(webcamInterval);
        if (webcamStream) {
            webcamStream.getTracks().forEach((track) => track.stop());
            webcamStream = null;
        }
        if (elements.localVideo) {
            elements.localVideo.srcObject = null;
        }
        elements.webcamToggleBtn.classList.remove('active');
        elements.webcamIconStart.classList.remove('hidden');
        elements.webcamIconStop.classList.add('hidden');
        elements.webcamBtnLabel.textContent = 'Activate My Webcam';
        elements.streamTag.textContent = 'LIVE FEED // CAM-0';

        reloadStream();
        showToast('Switched back to optical server stream');
    }

    function toggleDeviceWebcam() {
        if (isDeviceWebcamActive) {
            stopDeviceWebcam();
        } else {
            startDeviceWebcam();
        }
    }

    // =========================================================
    // SENSITIVITY CONFIGURATION CONTROLS
    // =========================================================
    function setupSliders() {
        elements.minAreaSlider.addEventListener('input', (e) => {
            elements.minAreaBadge.textContent = `${e.target.value} px`;
            queueConfigUpdate();
        });

        elements.confirmFramesSlider.addEventListener('input', (e) => {
            elements.confirmFramesBadge.textContent = `${e.target.value} frames`;
            queueConfigUpdate();
        });

        elements.cooldownSlider.addEventListener('input', (e) => {
            elements.cooldownBadge.textContent = `${e.target.value} s`;
            queueConfigUpdate();
        });

        elements.resetDefaultsBtn.addEventListener('click', () => {
            elements.minAreaSlider.value = 1200;
            elements.minAreaBadge.textContent = '1200 px';
            elements.confirmFramesSlider.value = 8;
            elements.confirmFramesBadge.textContent = '8 frames';
            elements.cooldownSlider.value = 10;
            elements.cooldownBadge.textContent = '10 s';
            queueConfigUpdate();
            showToast('Sensitivity reset to defaults');
        });
    }

    function queueConfigUpdate() {
        clearTimeout(configDebounceTimer);
        configDebounceTimer = setTimeout(async () => {
            try {
                await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        min_area: parseInt(elements.minAreaSlider.value, 10),
                        confirm_frames: parseInt(elements.confirmFramesSlider.value, 10),
                        cooldown: parseFloat(elements.cooldownSlider.value),
                    }),
                });
            } catch (err) {
                console.error('Config update failed:', err);
            }
        }, 250);
    }

    // =========================================================
    // STREAM VIEWPORT & CONTROLS
    // =========================================================
    function switchMode(mode) {
        currentMode = mode;
        elements.modeProcessedBtn.classList.toggle('active', mode === 'processed');
        elements.modeCleanBtn.classList.toggle('active', mode === 'clean');
        elements.modeMaskBtn.classList.toggle('active', mode === 'mask');

        let feedUrl = '/api/video_feed';
        let tag = 'LIVE FEED // CAM-0';
        if (mode === 'mask') {
            feedUrl = '/api/mask_feed';
            tag = 'COLOR FLAME MASK // DEBUG';
        } else if (mode === 'clean') {
            feedUrl = '/api/clean_feed';
            tag = 'CLEAN RAW FEED // 720P HD';
        }

        elements.cameraStream.src = `${feedUrl}?t=${Date.now()}`;
        elements.streamTag.textContent = tag;
        showToast(mode === 'clean' ? 'Clean HD camera feed active' : mode === 'mask' ? 'Flame mask debug view active' : 'Detection HUD overlay active');
    }

    function reloadStream() {
        let currentSrc = '/api/video_feed';
        if (currentMode === 'mask') {
            currentSrc = '/api/mask_feed';
        } else if (currentMode === 'clean') {
            currentSrc = '/api/clean_feed';
        }
        elements.cameraStream.src = `${currentSrc}?t=${Date.now()}`;
        showToast('Video stream refreshed');
    }

    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            elements.videoStage.requestFullscreen().catch((err) => {
                showToast(`Fullscreen error: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    }

    async function triggerManualSnapshot() {
        try {
            const res = await fetch('/api/manual_snapshot', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                showToast(`📸 Snapshot saved: ${data.filename}`);
                loadSnapshots();
            } else {
                showToast('Snapshot failed: No frame');
            }
        } catch (e) {
            showToast('Snapshot error');
        }
    }

    // =========================================================
    // SNAPSHOT GALLERY & MODAL
    // =========================================================
    async function loadSnapshots() {
        try {
            const res = await fetch('/api/snapshots');
            const data = await res.json();
            const snapshots = data.snapshots || [];

            elements.snapshotsCount.textContent = `${snapshots.length} Captures`;

            if (snapshots.length === 0) {
                elements.snapshotGrid.innerHTML = '<div class="empty-state">No incident snapshots recorded yet.</div>';
                return;
            }

            elements.snapshotGrid.innerHTML = snapshots.map((s) => `
                <div class="snapshot-card" data-url="${s.url}" data-time="${s.timestamp}" data-name="${s.filename}">
                    <img src="${s.url}" alt="${s.filename}" loading="lazy">
                    <div class="snapshot-info">
                        <span class="snap-time">${s.timestamp.split(' ')[1]}</span>
                        <span>${s.is_manual ? 'Manual' : 'Fire Alert'} (${s.size_kb} KB)</span>
                    </div>
                </div>
            `).join('');

            // Card click listener
            elements.snapshotGrid.querySelectorAll('.snapshot-card').forEach((card) => {
                card.addEventListener('click', () => {
                    openModal(card.dataset.url, card.dataset.time, card.dataset.name);
                });
            });
        } catch (err) {
            console.error('Failed to load snapshots:', err);
        }
    }

    function openModal(url, timestamp, filename) {
        currentModalFilename = filename;
        elements.modalImage.src = url;
        elements.modalTitle.textContent = filename;
        elements.modalMeta.textContent = `Recorded: ${timestamp}`;
        elements.modalDownloadBtn.href = url;
        elements.modalDownloadBtn.setAttribute('download', filename);
        elements.snapshotModal.classList.remove('hidden');
    }

    function closeModal() {
        elements.snapshotModal.classList.add('hidden');
        elements.modalImage.src = '';
        currentModalFilename = '';
    }

    async function clearGallery() {
        if (!confirm('Are you sure you want to clean and delete all saved snapshots from the dashboard?')) {
            return;
        }
        try {
            const res = await fetch('/api/clear_snapshots', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                showToast(`Cleaned ${data.cleared_count} snapshots from dashboard`);
                loadSnapshots();
            } else {
                showToast('Failed to clean snapshots');
            }
        } catch (err) {
            showToast('Error cleaning snapshots');
        }
    }

    async function deleteCurrentSnapshot() {
        if (!currentModalFilename) return;
        try {
            const res = await fetch(`/api/snapshot/${encodeURIComponent(currentModalFilename)}`, { method: 'DELETE' });
            const data = await res.json();
            if (data.success) {
                showToast(`Deleted snapshot: ${currentModalFilename}`);
                closeModal();
                loadSnapshots();
            } else {
                showToast('Failed to delete snapshot');
            }
        } catch (err) {
            showToast('Error deleting snapshot');
        }
    }

    // =========================================================
    // TOAST SYSTEM
    // =========================================================
    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3200);
    }

    // =========================================================
    // INITIALIZATION
    // =========================================================
    function init() {
        // Event listeners
        elements.soundToggleBtn.addEventListener('click', toggleSound);
        document.body.addEventListener('click', () => initAudio(), { once: true });

        elements.modeProcessedBtn.addEventListener('click', () => switchMode('processed'));
        elements.modeCleanBtn.addEventListener('click', () => switchMode('clean'));
        elements.modeMaskBtn.addEventListener('click', () => switchMode('mask'));
        elements.reloadStreamBtn.addEventListener('click', reloadStream);
        elements.fullscreenBtn.addEventListener('click', toggleFullscreen);
        elements.manualSnapBtn.addEventListener('click', triggerManualSnapshot);

        elements.refreshGalleryBtn.addEventListener('click', () => {
            loadSnapshots();
            showToast('Gallery refreshed');
        });

        elements.clearGalleryBtn.addEventListener('click', clearGallery);
        elements.modalDeleteBtn.addEventListener('click', deleteCurrentSnapshot);

        elements.modalCloseBtn.addEventListener('click', closeModal);
        elements.modalOverlay.addEventListener('click', closeModal);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !elements.snapshotModal.classList.contains('hidden')) {
                closeModal();
            }
        });

        if (elements.startWebcamCta) {
            elements.startWebcamCta.addEventListener('click', startDeviceWebcam);
        }
        if (elements.webcamToggleBtn) {
            elements.webcamToggleBtn.addEventListener('click', toggleDeviceWebcam);
        }

        setupSliders();
        loadSnapshots();

        // Start polling telemetry loop (every 320ms)
        pollStatus();
        setInterval(pollStatus, 320);
    }

    document.addEventListener('DOMContentLoaded', init);
})();
