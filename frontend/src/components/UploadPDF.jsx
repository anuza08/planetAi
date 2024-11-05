import React, { useState } from "react";
import axios from "axios";
import icon from "../assets/icon.png";

const UploadPDF = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    debugger;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("title", file.name);
    console.log("file name", file.name);

    try {
      const response = await axios.post(
        "http://localhost:8000/upload_pdf",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      onUploadSuccess(response.data.document_id);
    } catch (error) {
      console.error("Error uploading PDF", error);
    }
  };

  return (
    <div>
      <div style={styles.navbar}>
        <div style={styles.navLeft}>
          <img alt="logo" src={icon} />
          <h2>planet</h2>
        </div>
        <div style={styles.navRight}>
          {file && <span style={styles.fileName}>{file.name}</span>}
          {!file && (
            <label htmlFor="fileInput" style={styles.uploadLabel}>
              Upload PDF
            </label>
          )}
          <input
            id="fileInput"
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
          {file && (
            <button onClick={handleUpload} style={styles.uploadButton}>
              Upload
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

const styles = {
  navbar: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px 20px",
    backgroundColor: "#ffff",
    color: "black",
    boxShadow: "0px 4px 8px rgba(0, 0, 0, 0.1)",
  },
  navLeft: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
  },

  navRight: {
    display: "flex",
    alignItems: "center",
  },
  uploadLabel: {
    marginRight: "10px",
    padding: "10px",
    backgroundColor: "#4CAF50",
    color: "white",
    borderRadius: "5px",
    cursor: "pointer",
  },
  uploadButton: {
    padding: "10px",
    backgroundColor: "#4CAF50",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    marginLeft: "10px",
  },
  fileName: {
    color: "#4CAF50",
    marginRight: "10px",
    fontWeight: "bold",
  },
};

export default UploadPDF;
