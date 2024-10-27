import React, { useState, useEffect } from "react";
import axios from "axios";
import icon from "../assets/icon.png";

const AskQuestion = ({ documentId }) => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [chatHistory, setChatHistory] = useState([]);

  useEffect(() => {
    const fetchChatHistory = async () => {
      try {
        const response = await axios.get(
          `http://127.0.0.1:8000/document/${documentId}/questions`
        );
        setChatHistory(response.data);
      } catch (error) {
        console.error("Error fetching chat history", error);
      }
    };

    if (documentId) fetchChatHistory();
  }, [documentId]);

  const handleAskQuestion = async () => {
    try {
      const response = await axios.post("http://localhost:8000/ask_question", {
        document_id: documentId,
        question: question,
      });
      const newAnswer = response.data.answer;
      setAnswer(newAnswer);
      setChatHistory((prevHistory) => [
        ...prevHistory,
        { question, answer: newAnswer },
      ]);
      setQuestion("");
    } catch (error) {
      console.error("Error fetching answer", error);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.chatBox}>
        {chatHistory.map((chat, index) => (
          <div key={index} style={styles.chatMessage}>
            <p style={styles.question}>{chat.question}</p>
            <div style={styles.answerContainer}>
              <img alt="icon" src={icon} style={styles.icon} />
              <p style={styles.answer}>{chat.answer}</p>
            </div>
          </div>
        ))}
      </div>

      <div style={styles.questionContainer}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
          style={styles.input}
        />
        {documentId && (
          <button onClick={handleAskQuestion} style={styles.button}>
            Ask
          </button>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
    height: "100vh",
    padding: "10px",
  },
  chatBox: {
    flex: 1,
    overflowY: "auto",
    paddingBottom: "80px",
    marginBottom: "10px",
  },
  chatMessage: {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    marginBottom: "10px",
  },
  question: {
    alignSelf: "flex-end",
    backgroundColor: "#e0f7fa",
    padding: "10px",
    borderRadius: "10px",
    marginBottom: "5px",
    maxWidth: "70%",
    textAlign: "right",
  },
  answerContainer: {
    display: "flex",
    alignItems: "center",
    alignSelf: "flex-start",
    maxWidth: "70%",
    gap: "8px",
  },
  icon: {
    width: "24px",
    height: "24px",
  },
  answer: {
    backgroundColor: "#fff3e0",
    padding: "10px",
    borderRadius: "10px",
  },
  questionContainer: {
    position: "fixed",
    bottom: "20px",
    left: "20px",
    right: "20px",
    display: "flex",
    padding: "10px",
    backgroundColor: "#f9f9f9",
    boxShadow: "0px -2px 5px rgba(0, 0, 0, 0.1)",
    borderRadius: "8px",
    maxWidth: "800px",
    margin: "0 auto",
  },
  input: {
    flex: 1,
    padding: "8px",
    marginRight: "10px",
    borderRadius: "4px",
    border: "1px solid #ccc",
  },
  button: {
    padding: "8px 16px",
    borderRadius: "4px",
    backgroundColor: "#4CAF50",
    color: "#fff",
    border: "none",
    cursor: "pointer",
  },
};

export default AskQuestion;
