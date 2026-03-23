const scoreText = document.getElementById('score-text');
const progressFill = document.getElementById('progress-fill');
const appStatus = document.getElementById('app-status');
const refImg = document.getElementById('reference-img');
const noRefPlaceholder = document.getElementById('no-ref-placeholder');
const matchCelebration = document.getElementById('match-celebration');
const historyContainer = document.getElementById('history-container');

// Sound Fallback (Try to play sound if permitted by browser)
const matchSound = document.getElementById('match-sound');

let isCurrentlyMatched = false;

// Poll Status API every 150ms
setInterval(async () => {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // Update Reference Image state
        if (data.has_reference) {
            if (refImg.src !== data.ref_url && !refImg.src.includes(data.ref_url)) {
                refImg.src = data.ref_url;
                refImg.style.display = 'block';
                noRefPlaceholder.style.display = 'none';
                appStatus.textContent = 'Tracking...';
                appStatus.style.color = 'var(--accent)';
                appStatus.style.borderColor = 'var(--accent)';
            }
        } else {
            refImg.style.display = 'none';
            noRefPlaceholder.style.display = 'flex';
            appStatus.textContent = 'Awaiting Reference';
            appStatus.style.color = '#ef4444';
            appStatus.style.borderColor = '#ef4444';
        }

        // Only update live score if not matched
        if (!data.matched && !isCurrentlyMatched) {
            updateScoreUI(data.score);
        }

        // Trigger Match Sequence
        if (data.matched && !isCurrentlyMatched) {
            handleMatch(data.score);
        }

    } catch (err) {
        console.error("API error", err);
        appStatus.textContent = 'Offline';
        appStatus.style.color = 'gray';
        appStatus.style.borderColor = 'gray';
    }
}, 150);

function updateScoreUI(score) {
    const percentage = Math.min(Math.round(score * 100), 100);
    scoreText.textContent = `${percentage}%`;
    progressFill.style.width = `${percentage}%`;

    // Dynamic styling based on score
    if (percentage > 90) {
        progressFill.style.filter = "drop-shadow(0 0 10px var(--accent))";
        scoreText.style.color = "var(--accent)";
    } else if (percentage > 70) {
        progressFill.style.filter = "drop-shadow(0 0 5px #fbbf24)";
        scoreText.style.color = "#fbbf24";
    } else {
        progressFill.style.filter = "none";
        scoreText.style.color = "white";
    }
}

function handleMatch(score) {
    isCurrentlyMatched = true;
    updateScoreUI(score); 
    
    // Play sound (browser policies might block this without prior interaction)
    try {
        matchSound.play().catch(e => console.log("Audio playback prevented by browser"));
    } catch(e){}

    // Show Celebration UI
    matchCelebration.classList.add('active');
    
    // Add to history
    updateHistory();
}

function updateHistory() {
    const ts = new Date().getTime();
    historyContainer.innerHTML = `
        <div class="history-item slideUp">
            <img src="/static/latest_match.jpg?t=${ts}" alt="Matched Pose">
            <div class="label">SUCCESS</div>
        </div>
    `;
}

async function captureReference() {
    const btn = document.querySelector('.btn-capture');
    btn.innerHTML = '📸 Capturing...';
    btn.disabled = true;

    try {
        await fetch('/api/capture', { method: 'POST' });
        // Give it a moment to process backend
        setTimeout(() => {
            btn.innerHTML = '📸 Capture Live Pose as Reference';
            btn.disabled = false;
        }, 1000);
    } catch(err) {
        alert("Failed to capture");
        btn.innerHTML = '📸 Capture Live Pose as Reference';
        btn.disabled = false;
    }
}

async function resetMatch() {
    try {
        await fetch('/api/reset', { method: 'POST' });
        matchCelebration.classList.remove('active');
        isCurrentlyMatched = false;
    } catch(err) {
        console.error("Failed to reset");
    }
}
