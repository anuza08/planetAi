import React, { useState } from "react";
import UploadPDF from "./components/UploadPDF";
import AskQuestion from "./components/AskQuestion";

const App = () => {
  const [documentId, setDocumentId] = useState(null);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <UploadPDF onUploadSuccess={setDocumentId} />
      <div style={{ flex: 1 }} />
      <AskQuestion documentId={documentId} />
    </div>
  );
};

export default App;
