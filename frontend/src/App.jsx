import React, { useEffect, useState } from "react";

// Empty base = same origin, which is how the bundled nginx proxy serves it.
// Set VITE_API_BASE=http://localhost:8000 when running `npm run dev` directly.
const API_BASE = import.meta.env.VITE_API_BASE || "";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [info, setInfo] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/healthz`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setInfo(null));
  }, []);

  const onFile = (e) => {
    const f = e.target.files?.[0];
    setError(null);
    setResult(null);
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
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
    setResult(null);

    try {
      const fd = new FormData();
      // The API accepts "image" or "file"; "image" is the documented name.
      fd.append("image", file);

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const pct = (v) => `${(v * 100).toFixed(1)}%`;

  return (
    <div style={S.page}>
      <h1 style={{ marginBottom: 4 }}>LandmarkLens</h1>
      <p style={S.sub}>
        Recognise London transit signage and landmarks from a photo, with a
        Grad-CAM heatmap showing what the model actually looked at.
      </p>
      {info && (
        <p style={S.meta}>
          {info.backbone} &middot; {info.backend} backend &middot;{" "}
          {info.num_classes} classes
          {info.trained?.val_top1 != null &&
            ` · val top-1 ${pct(info.trained.val_top1)}`}
        </p>
      )}

      <form onSubmit={onSubmit} style={S.form}>
        <input type="file" accept="image/*" onChange={onFile} />
        <button disabled={!file || loading} style={S.button(!file || loading)}>
          {loading ? "Analysing…" : "Identify location"}
        </button>
      </form>

      {error && (
        <div style={S.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={S.card}>
          <div style={S.headline}>{result.label}</div>
          <div style={S.confidence}>{pct(result.confidence)} confidence</div>

          <dl style={S.dl}>
            <dt style={S.dt}>location_name</dt>
            <dd style={S.dd}>
              <code>{result.location_name}</code>
            </dd>
            <dt style={S.dt}>routable to</dt>
            <dd style={S.dd}>
              {result.resolvable
                ? "yes — a specific place"
                : "no — a kind of place; the user must say which one"}
            </dd>
            {result.timing_ms && (
              <>
                <dt style={S.dt}>inference</dt>
                <dd style={S.dd}>{result.timing_ms.inference} ms</dd>
              </>
            )}
          </dl>

          {result.low_confidence && (
            <div style={S.warn}>
              Below the {pct(result.confidence_floor)} confidence floor — treat
              this as a guess.
            </div>
          )}

          <h3 style={S.h3}>Top 3</h3>
          <ol style={{ marginTop: 4 }}>
            {(result.predictions || []).map((p) => (
              <li key={p.class}>
                {p.label} — {pct(p.confidence)}
              </li>
            ))}
          </ol>
        </div>
      )}

      <div style={S.images}>
        {preview && (
          <figure style={S.figure}>
            <img src={preview} alt="uploaded" style={S.img} />
            <figcaption style={S.caption}>Your photo</figcaption>
          </figure>
        )}
        {result?.heatmap_url && (
          <figure style={S.figure}>
            <img
              src={`${API_BASE}${result.heatmap_url}`}
              alt="Grad-CAM heatmap"
              style={S.img}
            />
            <figcaption style={S.caption}>
              Grad-CAM — warm areas drove the prediction
            </figcaption>
          </figure>
        )}
      </div>
    </div>
  );
}

const S = {
  page: {
    maxWidth: 780,
    margin: "40px auto",
    padding: 16,
    fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
  },
  sub: { color: "#555", marginTop: 0 },
  meta: { color: "#777", fontSize: 13, marginTop: -8 },
  form: { display: "flex", gap: 12, alignItems: "center", marginTop: 16, flexWrap: "wrap" },
  button: (disabled) => ({
    padding: "10px 16px",
    borderRadius: 8,
    border: "1px solid #ccc",
    background: disabled ? "#f2f2f2" : "#fff",
    cursor: disabled ? "not-allowed" : "pointer",
  }),
  error: { marginTop: 16, color: "#b00020" },
  card: {
    marginTop: 20,
    padding: 16,
    border: "1px solid #e6e6e6",
    borderRadius: 12,
    background: "#fafafa",
  },
  headline: { fontSize: 22, fontWeight: 600 },
  confidence: { color: "#555", marginTop: 2 },
  dl: { display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 12px", marginTop: 12 },
  dt: { color: "#777", fontSize: 13 },
  dd: { margin: 0, fontSize: 14 },
  warn: {
    marginTop: 12,
    padding: "8px 10px",
    borderRadius: 8,
    background: "#fff6e0",
    border: "1px solid #f0d9a0",
    fontSize: 13,
  },
  h3: { marginBottom: 0, marginTop: 16, fontSize: 15 },
  images: { display: "flex", gap: 16, flexWrap: "wrap", marginTop: 20 },
  figure: { margin: 0, flex: "1 1 300px" },
  img: { maxWidth: "100%", borderRadius: 8, border: "1px solid #eee", display: "block" },
  caption: { fontSize: 12, color: "#777", marginTop: 4 },
};
