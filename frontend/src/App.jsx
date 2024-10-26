import React, { useState } from "react";
import UploadPDF from "./components/UploadPDF";
import AskQuestion from "./components/AskQuestion";

const App = () => {
  const [documentId, setDocumentId] = useState(null);

  return (
    <div>
      <h1>PDF Q&A</h1>
      <UploadPDF onUploadSuccess={setDocumentId} />
      {documentId && <AskQuestion documentId={documentId} />}
    </div>
  );
};

export default App;
