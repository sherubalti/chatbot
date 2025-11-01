document.addEventListener("DOMContentLoaded", () => {
    const sidebar = document.getElementById("sidebar");
    const menuBtn = document.getElementById("menu-btn");
    const sendBtn = document.getElementById("send-button");
    const voiceBtn = document.getElementById("voice-button");
    const userInput = document.getElementById("user-input");
    const chatMessages = document.getElementById("chat-messages");
    const historyList = document.getElementById("history-list");
    const clearHistoryBtn = document.getElementById("clear-history");
    const chatContainer = document.getElementById("chat-container");

    let conversationHistory = JSON.parse(localStorage.getItem("collegeBotHistory")) || [];
    let isBotTyping = false;
    let currentStreamingMessage = null;
    let recognition = null;
    let currentFeedbackMessage = null;
    let feedbackChart = null;
    let feedbackStates = new Map(); // Track feedback state per message

    // Voice recognition setup
    function initializeVoiceRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                voiceBtn.classList.add('listening');
                voiceBtn.innerHTML = '🔴';
                userInput.placeholder = "Listening... Speak now";
            };

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                userInput.value = transcript;
                voiceBtn.classList.remove('listening');
                voiceBtn.innerHTML = '🎤';
                userInput.placeholder = "Type your message...";
                
                // Auto-send the voice input
                if (transcript.trim() && !isBotTyping) {
                    setTimeout(() => {
                        sendVoiceMessage(transcript);
                    }, 500);
                }
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                voiceBtn.classList.remove('listening');
                voiceBtn.innerHTML = '🎤';
                userInput.placeholder = "Type your message...";
                
                if (event.error === 'not-allowed') {
                    alert('Microphone access denied. Please allow microphone permissions.');
                }
            };

            recognition.onend = () => {
                voiceBtn.classList.remove('listening');
                voiceBtn.innerHTML = '🎤';
                userInput.placeholder = "Type your message...";
            };
        } else {
            voiceBtn.style.display = 'none';
            console.warn('Speech recognition not supported in this browser');
        }
    }

    // Text-to-speech function
    function speakText(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.9;
            utterance.pitch = 1;
            utterance.volume = 0.8;
            speechSynthesis.speak(utterance);
        }
    }

    // ------------------ Helpers ------------------
    function updateHistoryDisplay() {
        historyList.innerHTML = "";
        if (!conversationHistory.length) {
            historyList.innerHTML = `<div class="history-item">No history</div>`;
            return;
        }
        conversationHistory.slice().reverse().forEach(c => {
            const item = document.createElement("div");
            item.className = "history-item";
            item.textContent = c.messages[0].text.substring(0, 30) + "...";
            historyList.appendChild(item);
        });
    }

    function saveToHistory(userMessage, botResponse) {
        const newConv = {
            id: Date.now(),
            messages: [
                { sender: "user", text: userMessage },
                { sender: "bot", text: botResponse }
            ]
        };
        conversationHistory.push(newConv);
        // Keep only last 50 conversations to prevent memory issues
        if (conversationHistory.length > 50) {
            conversationHistory = conversationHistory.slice(-50);
        }
        localStorage.setItem("collegeBotHistory", JSON.stringify(conversationHistory));
        updateHistoryDisplay();
    }

    function setInputState(disabled) {
        sendBtn.disabled = disabled;
        userInput.disabled = disabled;
        voiceBtn.disabled = disabled;
        
        if (disabled) {
            sendBtn.style.opacity = "0.6";
            voiceBtn.style.opacity = "0.6";
            userInput.placeholder = "Bot is responding...";
        } else {
            sendBtn.style.opacity = "1";
            voiceBtn.style.opacity = "1";
            userInput.placeholder = "Type your message...";
            userInput.focus();
        }
    }

    function cleanResponseText(text) {
        if (!text || typeof text !== 'string') {
            return "I'm sorry, I couldn't generate a proper response. Please try again.";
        }
        
        let cleaned = text
            .replace(/([A-Z])\1{10,}/gi, '')
            .replace(/(.)\1{15,}/g, '')
            .replace(/\|{5,}/g, '')
            .replace(/h{5,}o*t{5,}/gi, '')
            .trim();
        
        if (!cleaned || cleaned.length < 5 || cleaned.match(/^[^a-zA-Z0-9]*$/)) {
            return "I'm sorry, I couldn't generate a proper response. Please try again.";
        }
        
        return cleaned;
    }

    function showThinkingIndicator() {
        const thinkingBubble = document.createElement("div");
        thinkingBubble.className = "message bot-message thinking-indicator";
        thinkingBubble.innerHTML = `
            <div class="thinking-container">
                <span class="thinking-text">Thinking</span>
                <div class="thinking-animation">
                    <span class="thinking-dot"></span>
                    <span class="thinking-dot"></span>
                    <span class="thinking-dot"></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(thinkingBubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return thinkingBubble;
    }

    function addMessage(text, className, isStreaming = false, question = null, answer = null) {
        const msg = document.createElement("div");
        msg.className = `message ${className}`;
        const messageId = Date.now();
        msg.setAttribute('data-message-id', messageId);
        
        const cleanText = cleanResponseText(text);
        
        if (!isStreaming) {
            msg.textContent = cleanText;
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Add feedback buttons to bot messages after they're fully displayed
            if (className === 'bot-message') {
                setTimeout(() => {
                    addFeedbackButtons(msg, question || getLastUserMessage(), cleanText, messageId);
                }, 100);
            }
            
            return msg;
        }
        
        // Streaming effect
        const textContainer = document.createElement("span");
        const cursor = document.createElement("span");
        cursor.className = "streaming-cursor";
        cursor.textContent = "|";
        
        msg.appendChild(textContainer);
        msg.appendChild(cursor);
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        currentStreamingMessage = {
            element: textContainer,
            cursor: cursor,
            fullText: cleanText,
            currentIndex: 0,
            timeout: null,
            question: question || getLastUserMessage(),
            answer: cleanText,
            messageId: messageId
        };
        
        return msg;
    }

    function getLastUserMessage() {
        const userMessages = document.querySelectorAll('.user-message');
        return userMessages.length > 0 ? userMessages[userMessages.length - 1].textContent : "User query";
    }

    function startStreamingEffect() {
        if (!currentStreamingMessage) return;
        
        const { element, cursor, fullText, question, answer, messageId } = currentStreamingMessage;
        let index = 0;
        
        if (currentStreamingMessage.timeout) {
            clearTimeout(currentStreamingMessage.timeout);
        }
        
        function typeCharacter() {
            if (index < fullText.length) {
                const char = fullText.charAt(index);
                element.textContent += char;
                cursor.style.marginLeft = "1px";
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                index++;
                currentStreamingMessage.currentIndex = index;
                
                const speed = Math.random() * 40 + 20;
                currentStreamingMessage.timeout = setTimeout(typeCharacter, speed);
            } else {
                cursor.remove();
                
                // Add feedback buttons after streaming completes
                addFeedbackButtons(currentStreamingMessage.element.parentElement, question, answer, messageId);
                
                currentStreamingMessage = null;
                isBotTyping = false;
                setInputState(false);
            }
        }
        
        currentStreamingMessage.timeout = setTimeout(typeCharacter, 300);
    }

    function stopStreamingEffect() {
        if (currentStreamingMessage && currentStreamingMessage.timeout) {
            clearTimeout(currentStreamingMessage.timeout);
            
            if (currentStreamingMessage.element) {
                currentStreamingMessage.element.textContent = currentStreamingMessage.fullText;
                if (currentStreamingMessage.cursor) {
                    currentStreamingMessage.cursor.remove();
                }
                
                // Add feedback buttons even when streaming is stopped
                addFeedbackButtons(
                    currentStreamingMessage.element.parentElement, 
                    currentStreamingMessage.question, 
                    currentStreamingMessage.answer,
                    currentStreamingMessage.messageId
                );
            }
            
            currentStreamingMessage = null;
            isBotTyping = false;
            setInputState(false);
        }
    }

    function extractResponseText(data) {
        if (!data) return "No response received from the server.";
        
        if (typeof data === 'string') return data;
        
        if (data.response) {
            if (typeof data.response === 'string') return data.response;
            if (typeof data.response === 'object' && data.response.response) {
                return data.response.response;
            }
        }
        
        return "I'm sorry, I couldn't process the response properly. Please try again.";
    }

    // ------------------ Feedback System ------------------
    function initializeFeedbackSystem() {
        // Load statistics when page loads
        loadFeedbackStatistics();
    }

    function loadFeedbackStatistics() {
        fetch('/feedback/stats')
            .then(response => response.json())
            .then(data => {
                updateStatsDisplay(data.stats);
                if (data.timeline.length > 0) {
                    renderFeedbackChart(data.timeline);
                }
            })
            .catch(error => console.error('Error loading statistics:', error));
    }

    function updateStatsDisplay(stats) {
        const totalLikes = document.getElementById('total-likes');
        const totalDislikes = document.getElementById('total-dislikes');
        const totalComments = document.getElementById('total-comments');
        
        if (totalLikes) totalLikes.textContent = stats.total_likes;
        if (totalDislikes) totalDislikes.textContent = stats.total_dislikes;
        if (totalComments) totalComments.textContent = stats.total_comments;
    }

    function renderFeedbackChart(timelineData) {
        const chartCanvas = document.getElementById('feedback-chart');
        if (!chartCanvas) return;
        
        const ctx = chartCanvas.getContext('2d');
        
        if (feedbackChart) {
            feedbackChart.destroy();
        }
        
        const dates = timelineData.map(item => item.date);
        const likes = timelineData.map(item => item.likes);
        const dislikes = timelineData.map(item => item.dislikes);
        
        feedbackChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Likes',
                        data: likes,
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Dislikes',
                        data: dislikes,
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }

    function addFeedbackButtons(messageElement, question, answer, messageId) {
        // Check if feedback buttons already exist
        if (messageElement.querySelector('.feedback-container')) {
            return;
        }
        
        // Initialize feedback state for this message
        if (!feedbackStates.has(messageId)) {
            feedbackStates.set(messageId, {
                liked: false,
                disliked: false,
                question: question,
                answer: answer
            });
        }
        
        const feedbackState = feedbackStates.get(messageId);
        
        const feedbackContainer = document.createElement('div');
        feedbackContainer.className = 'feedback-container';
        feedbackContainer.setAttribute('data-message-id', messageId);
        
        feedbackContainer.innerHTML = `
            <div class="feedback-prompt">Was this response helpful?</div>
            <div class="feedback-buttons">
                <button class="feedback-btn like-btn ${feedbackState.liked ? 'active' : ''}" 
                        data-feedback="like" 
                        title="Like this response"
                        ${feedbackState.liked ? 'disabled' : ''}>
                    <i class="far fa-thumbs-up"></i> Like
                </button>
                <button class="feedback-btn dislike-btn ${feedbackState.disliked ? 'active' : ''}" 
                        data-feedback="dislike" 
                        title="Dislike this response"
                        ${feedbackState.disliked ? 'disabled' : ''}>
                    <i class="far fa-thumbs-down"></i> Dislike
                </button>
                <button class="feedback-btn comment-btn" data-feedback="comment" title="Add a comment">
                    <i class="far fa-comment"></i> Comment
                </button>
            </div>
        `;
        
        messageElement.appendChild(feedbackContainer);
        
        // Add event listeners
        const likeBtn = feedbackContainer.querySelector('.like-btn');
        const dislikeBtn = feedbackContainer.querySelector('.dislike-btn');
        const commentBtn = feedbackContainer.querySelector('.comment-btn');
        
        likeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (!feedbackState.liked && !feedbackState.disliked) {
                handleFeedback('like', question, answer, messageId);
            }
        });
        
        dislikeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (!feedbackState.liked && !feedbackState.disliked) {
                handleFeedback('dislike', question, answer, messageId);
            }
        });
        
        commentBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openCommentModal(question, answer, messageId);
        });
    }

    function handleFeedback(type, question, answer, messageId) {
        const feedbackState = feedbackStates.get(messageId);
        if (!feedbackState || feedbackState.liked || feedbackState.disliked) {
            return; // Already submitted feedback for this message
        }
        
        const like = type === 'like';
        const dislike = type === 'dislike';
        
        // Update UI immediately
        const feedbackContainer = document.querySelector(`.feedback-container[data-message-id="${messageId}"]`);
        if (feedbackContainer) {
            const likeBtn = feedbackContainer.querySelector('.like-btn');
            const dislikeBtn = feedbackContainer.querySelector('.dislike-btn');
            
            if (like) {
                likeBtn.classList.add('active');
                likeBtn.disabled = true;
                dislikeBtn.disabled = true;
                feedbackState.liked = true;
            } else if (dislike) {
                dislikeBtn.classList.add('active');
                likeBtn.disabled = true;
                dislikeBtn.disabled = true;
                feedbackState.disliked = true;
            }
        }
        
        fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                answer: answer,
                like: like,
                dislike: dislike
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showFeedbackMessage('Thank you for your feedback!');
                loadFeedbackStatistics(); // Refresh stats
            }
        })
        .catch(error => {
            console.error('Error saving feedback:', error);
            showFeedbackMessage('Error saving feedback. Please try again.');
            // Revert UI state on error
            if (feedbackContainer) {
                const likeBtn = feedbackContainer.querySelector('.like-btn');
                const dislikeBtn = feedbackContainer.querySelector('.dislike-btn');
                likeBtn.classList.remove('active');
                dislikeBtn.classList.remove('active');
                likeBtn.disabled = false;
                dislikeBtn.disabled = false;
                feedbackState.liked = false;
                feedbackState.disliked = false;
            }
        });
    }

    function openCommentModal(question, answer, messageId) {
        currentFeedbackMessage = { question, answer, messageId };
        const commentModal = document.getElementById('comment-modal');
        if (commentModal) {
            commentModal.classList.remove('hidden');
            document.getElementById('comment-text').focus();
        }
    }

    function closeCommentModal() {
        const commentModal = document.getElementById('comment-modal');
        if (commentModal) {
            commentModal.classList.add('hidden');
            document.getElementById('comment-text').value = '';
            currentFeedbackMessage = null;
        }
    }

    function submitComment() {
        const commentText = document.getElementById('comment-text').value.trim();
        
        if (!commentText) {
            alert('Please enter a comment');
            return;
        }
        
        if (!currentFeedbackMessage) {
            alert('No message selected for comment');
            return;
        }
        
        fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: currentFeedbackMessage.question,
                answer: currentFeedbackMessage.answer,
                comments: commentText
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showFeedbackMessage('Thank you for your comment!');
                closeCommentModal();
                loadFeedbackStatistics();
            }
        })
        .catch(error => {
            console.error('Error saving comment:', error);
            showFeedbackMessage('Error saving comment. Please try again.');
        });
    }

    function showFeedbackMessage(message) {
        // Create a temporary notification
        const notification = document.createElement('div');
        notification.className = 'feedback-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            z-index: 3000;
            animation: slideIn 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            font-size: 14px;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    // ------------------ UI wiring ------------------
    menuBtn.addEventListener("click", () => {
        sidebar.classList.toggle("hidden");
        chatContainer.classList.toggle("sidebar-open");
    });

    sendBtn.addEventListener("click", () => sendMessage());
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !isBotTyping) sendMessage();
    });

    voiceBtn.addEventListener("click", () => {
        if (recognition && !isBotTyping) {
            recognition.start();
        }
    });

    clearHistoryBtn.addEventListener("click", () => {
        if (confirm("Clear all history?")) {
            conversationHistory = [];
            localStorage.removeItem("collegeBotHistory");
            updateHistoryDisplay();
        }
    });

    chatMessages.addEventListener('click', () => {
        if (currentStreamingMessage) {
            stopStreamingEffect();
        }
    });

    // Initialize voice recognition
    initializeVoiceRecognition();
    updateHistoryDisplay();
    initializeFeedbackSystem();

    // Add feedback buttons to welcome message
    const welcomeMessage = document.querySelector('.bot-message');
    if (welcomeMessage) {
        const welcomeMessageId = Date.now();
        welcomeMessage.setAttribute('data-message-id', welcomeMessageId);
        addFeedbackButtons(welcomeMessage, "Welcome", "Hello! I'm College Bot. How can I assist you today?", welcomeMessageId);
    }

    // Modal event listeners
    const statsBtn = document.getElementById('stats-btn');
    if (statsBtn) {
        statsBtn.addEventListener('click', () => {
            const statsModal = document.getElementById('stats-modal');
            if (statsModal) {
                statsModal.classList.remove('hidden');
                loadFeedbackStatistics();
            }
        });
    }

    const closeStats = document.getElementById('close-stats');
    if (closeStats) {
        closeStats.addEventListener('click', () => {
            const statsModal = document.getElementById('stats-modal');
            if (statsModal) {
                statsModal.classList.add('hidden');
            }
        });
    }

    const closeComment = document.getElementById('close-comment');
    if (closeComment) {
        closeComment.addEventListener('click', closeCommentModal);
    }

    const cancelComment = document.getElementById('cancel-comment');
    if (cancelComment) {
        cancelComment.addEventListener('click', closeCommentModal);
    }

    const submitCommentBtn = document.getElementById('submit-comment');
    if (submitCommentBtn) {
        submitCommentBtn.addEventListener('click', submitComment);
    }

    // Close modals when clicking outside
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.add('hidden');
            }
        });
    });

    // ------------------ Send flow ------------------
    async function sendMessage() {
        if (isBotTyping) return;
        
        const text = userInput.value.trim();
        if (!text) return;

        // Add user message
        addMessage(text, "user-message", false);
        userInput.value = "";
        
        // Process the message
        await processUserMessage(text);
    }

    // Separate function for voice messages to ensure proper flow
    async function sendVoiceMessage(text) {
        if (isBotTyping) return;
        if (!text.trim()) return;

        // Add user message
        addMessage(text, "user-message", false);
        userInput.value = "";
        
        // Process the message
        await processUserMessage(text);
    }

    async function processUserMessage(text) {
        // Disable input during bot response
        isBotTyping = true;
        setInputState(true);

        const thinkingIndicator = showThinkingIndicator();

        try {
            await new Promise(resolve => setTimeout(resolve, 800));

            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });
            
            if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
            
            const data = await res.json();
            thinkingIndicator.remove();

            let responseText = extractResponseText(data);
            responseText = cleanResponseText(responseText);

            // Add bot message with streaming
            addMessage(responseText, "bot-message", true, text, responseText);
            startStreamingEffect();
            
            // Speak the response automatically for voice interactions
            setTimeout(() => speakText(responseText), 1000);

            // Save to history
            const estimatedTime = responseText.length * 40 + 1000;
            setTimeout(() => {
                if (!currentStreamingMessage) {
                    saveToHistory(text, responseText);
                }
            }, estimatedTime);

        } catch (err) {
            if (thinkingIndicator && thinkingIndicator.parentNode) {
                thinkingIndicator.remove();
            }
            stopStreamingEffect();

            const errorMsg = "Sorry, I encountered an error. Please try again.";
            addMessage(errorMsg, "bot-message", false, text, errorMsg);
            saveToHistory(text, errorMsg);
            
            isBotTyping = false;
            setInputState(false);
            console.error("Chat error:", err);
        }
    }

    // Add CSS styles for feedback system
    const style = document.createElement('style');
    style.textContent = `
        .feedback-container {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #E2E8F0;
        }
        
        .feedback-prompt {
            font-size: 0.8rem;
            color: #718096;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .feedback-buttons {
            display: flex;
            gap: 10px;
            margin-top: 5px;
            flex-wrap: wrap;
        }
        
        .feedback-btn {
            padding: 6px 14px;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            background: #FFFFFF;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 6px;
            color: #4A5568;
        }
        
        .feedback-btn:hover:not(:disabled) {
            transform: translateY(-1px);
            border-color: #4A5568;
            color: #4A5568;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .feedback-btn:disabled {
            cursor: not-allowed;
            opacity: 0.7;
        }
        
        .like-btn.active {
            background: #28a745 !important;
            color: white !important;
            border-color: #28a745 !important;
        }
        
        .dislike-btn.active {
            background: #dc3545 !important;
            color: white !important;
            border-color: #dc3545 !important;
        }
        
        .like-btn:hover:not(:disabled) {
            border-color: #28a745;
            color: #28a745;
        }
        
        .dislike-btn:hover:not(:disabled) {
            border-color: #dc3545;
            color: #dc3545;
        }
        
        .comment-btn:hover:not(:disabled) {
            border-color: #007bff;
            color: #007bff;
        }
        
        .feedback-notification {
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Ensure modal styles */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
        }
        
        .modal.hidden {
            display: none;
        }
        
        .modal-content {
            background: white;
            border-radius: 10px;
            width: 90%;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        
        .streaming-cursor {
            animation: blink 1.2s infinite;
            color: #007bff;
            font-weight: bold;
            margin-left: 2px;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        
        .thinking-indicator {
            color: #666;
            font-style: italic;
        }
        
        .thinking-container {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .thinking-animation {
            display: flex;
            gap: 3px;
        }
        
        .thinking-dot {
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background-color: #666;
            animation: think-bounce 1.4s infinite ease-in-out;
        }
        
        .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes think-bounce {
            0%, 80%, 100% { 
                transform: translateY(0);
                opacity: 0.3;
            }
            40% { 
                transform: translateY(-5px);
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);
});