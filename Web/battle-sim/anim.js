document.addEventListener("DOMContentLoaded", () => {
    const moodOptions = document.getElementById("mood-options");

    moodOptions.addEventListener("change", (event) => {
        if (event.target.name === "mood") {
            const selectedMood = event.target.value;
            console.log(`Настрій: ${selectedMood}`);
            showMoodMessage(selectedMood);
            sendMoodToServer(selectedMood);
        }
    });
});

function showMoodMessage(mood) {
    const messageBox = document.createElement("div");
    messageBox.className = "mood-message";
    messageBox.innerText = `Ви обрали: ${mood}`;
    document.body.appendChild(messageBox);

    setTimeout(() => {
        messageBox.classList.add("fade-out");
        messageBox.addEventListener("transitionend", () => messageBox.remove());
    }, 2000);
}

function sendMoodToServer(mood) {
    fetch("/api/mood", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ mood }),
    }).then((response) => {
        if (!response.ok) {
            console.error("Не вдалося відправити настрій на сервер");
        }
    });
}
