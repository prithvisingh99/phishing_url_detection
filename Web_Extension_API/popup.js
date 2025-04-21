document.addEventListener('DOMContentLoaded', function () {
    const predictButton = document.getElementById("predictButton");
    const urlInput = document.getElementById("urlInput");
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.getElementById("loading");

    predictButton.addEventListener("click", () => predict(urlInput.value.trim()));
    urlInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") predict(urlInput.value.trim());
    });

    async function predict(url) {
        const resultDiv = document.getElementById("result");
        const loadingDiv = document.getElementById("loading");
    
        // Clear previous results
        resultDiv.innerHTML = '';
        loadingDiv.style.display = 'block';
    
        try {
            const response = await fetch("http://localhost:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url })
            });
    
            const data = await response.json();
            
            if (!response.ok) {
                showError(data.error || "Unknown server error");
                return;
            }
    
            updateUI(data);
            
        } catch (error) {
            console.error("Network error:", error);
            showError("Unable to connect to the server");
        } finally {
            loadingDiv.style.display = 'none';
        }
    }
    
    function updateUI(data) {
        const resultDiv = document.getElementById("result");
        
        const card = document.createElement("div");
        card.className = `result-card ${data.prediction}-card`;
        
        card.innerHTML = `
            <div class="status-icon">${data.prediction === 'phishing' ? '⚠️' : '✅'}</div>
            <div class="status-details">
                <h3>${data.prediction.toUpperCase()}</h3>
                <p>Confidence: ${data.confidence}%</p>
            </div>
        `;
        
        resultDiv.appendChild(card);
    }

    function showError(message) {
        resultDiv.innerHTML = `
            <div class="error-card">
                <div class="error-icon">❌</div>
                <p>${message}</p>
            </div>
        `;
    }
});
