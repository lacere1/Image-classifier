import React, { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preds, setPreds] = useState([]);

  const onFile = (e) => {
    const f = e.target.files?.[0];
    setError(null);
    setPreds([]);
    if (f) {
      setFile(f);
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setFile(null);
      setPreview(null);
    }
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    setPreds([]);

    try {
      const fd = new FormData();
      fd.append("image", file);
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setPreds(data.predictions || []);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: 16 }}>
      <h1 style={{ marginBottom: 8 }}>Image Classifier</h1>
      <p style={{ color: "#555", marginTop: 0 }}>
        Upload an image and get the top 5 predictions.
      </p>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: 12, marginTop: 16 }}>
        <input type="file" accept="image/*" onChange={onFile} />
        <button
          disabled={!file || loading}
          style={{
            padding: "10px 16px",
            borderRadius: 8,
            border: "1px solid #ccc",
            cursor: !file || loading ? "not-allowed" : "pointer",
            width: 140
          }}
        >
          {loading ? "Classifying..." : "Analyze Image"}
        </button>
      </form>

      {preview && (
        <div style={{ marginTop: 16 }}>
          <img
            src={preview}
            alt="preview"
            style={{ maxWidth: "100%", borderRadius: 8, border: "1px solid #eee" }}
          />
        </div>
      )}

      {error && (
        <div style={{ marginTop: 16, color: "#b00020" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {preds.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h2>Top‑5 Predictions</h2>
          <ol>
            {preds.map((p, idx) => (
              <li key={idx}>
                {p.label} — {(p.score * 100).toFixed(1)}%
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
