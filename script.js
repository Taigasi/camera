const API_URL = "http://localhost:5000/dados";

async function atualizarInterface() {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) throw new Error("API Offline");
        
        const data = await response.json();

        // 1. Atualizar Score
        const scoreEl = document.getElementById("score-val");
        const scoreCircleEl = document.querySelector(".score-circle");
        scoreEl.innerText = data.score_geral;
        
        const scoreColor = data.score_geral > 70 ? "var(--success)" : "var(--danger)";
        scoreEl.style.color = scoreColor;
        scoreCircleEl.style.borderColor = scoreColor;

        // 2. Atualizar Postura
        const postEl = document.getElementById("postura-val");
        const dotPost = document.getElementById("dot-postura");
        postEl.innerText = data.postura_status;
        
        if (data.postura_status === "Excelente" || data.postura_status === "Boa") {
            dotPost.className = "status-dot good";
            postEl.style.color = "var(--success)";
        } else {
            dotPost.className = "status-dot bad";
            postEl.style.color = "var(--danger)";
        }

        // 3. Atualizar Emoção
        document.getElementById("emo-val").innerText = data.emocao_dominante;

        // 4. Atualizar Contato Visual
        const eyeFillEl = document.getElementById("eye-fill");
        const contatoVisualPct = data.contato_visual;

        eyeFillEl.style.width = `${contatoVisualPct}%`;

        let eyeFillColor = "var(--success)";
        if (contatoVisualPct < 50) {
            eyeFillColor = "var(--danger)";
        } else if (contatoVisualPct < 80) {
            eyeFillColor = "#f59e0b"; // Laranja
        }
        eyeFillEl.style.background = eyeFillColor;


        // 5. Feedback Inteligente
        const feedBox = document.getElementById("feedback-txt");
        let feedbackHTML = "";
        let feedbackColor = "#f0fdf4"; 
        let feedbackBorder = "var(--success)";

        if (data.atencao_alerta) {
            feedbackHTML = "<strong>Alerta de Foco:</strong> Mantenha o foco na câmera (entrevistador). Seu contato visual está baixo.";
            feedbackColor = "#fef2f2"; // Vermelho claro
            feedbackBorder = "var(--danger)";
        } else if (data.postura_status !== "Excelente") {
            feedbackHTML = "<strong>Atenção:</strong> Sua postura está muito relaxada. Tente se endireitar para transmitir mais confiança.";
            feedbackColor = "#fffbeb"; // Amarelo claro
            feedbackBorder = "#f59e0b";
        } else {
            feedbackHTML = "<strong>Ótimo!</strong> Você está transmitindo confiança e profissionalismo.";
        }

        feedBox.innerHTML = feedbackHTML;
        feedBox.style.borderLeftColor = feedbackBorder;
        feedBox.style.background = feedbackColor;

    } catch (error) {
        console.error("Erro ao conectar:", error);
        document.getElementById("score-val").innerText = "OFF";
    }
}

// Atualiza a cada 500ms (meio segundo)
setInterval(atualizarInterface, 500);