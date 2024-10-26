import React, { useState } from "react";
import axios from "axios";

const AskQuestion = ({ documentId }) => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleAskQuestion = async () => {
    try {
      const response = await axios.post("http://localhost:8000/ask_question", {
        document_id: documentId,
        question: question,
      });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error("Error fetching answer", error);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question..."
      />
      <button onClick={handleAskQuestion}>Ask</button>
      {answer && <p>Answer: {answer}</p>}
    </div>
  );
};

export default AskQuestion;
