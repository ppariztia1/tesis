import { useState, useMemo } from "react";
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, BarChart, Bar, Legend,
} from "recharts";

/* ──────────────────────────────────────────────────────────────────────────
   Laboratorio de sensibilidad — Modelo Regret-Grid con aversión a la pérdida
   Mueve los parámetros y mira cómo cambian las curvas de regret y los
   histogramas (β simétrico vs asimétrico) en tiempo real.
   Nota: el RNG es propio de JS (no el PCG64 de numpy), así que los valores
   absolutos difieren un poco del script en Python; el comportamiento es fiel.
   ────────────────────────────────────────────────────────────────────────── */

// --- RNG reproducible (mulberry32) + normal (Box-Muller) ---
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeNormal(rng) {
  let spare = null;
  return (m, sd) => {
    if (spare !== null) { const v = spare; spare = null; return m + sd * v; }
    let u, v, r;
    do { u = 2 * rng() - 1; v = 2 * rng() - 1; r = u * u + v * v; } while (r === 0 || r >= 1);
    const c = Math.sqrt((-2 * Math.log(r)) / r);
    spare = v * c;
    return m + sd * (u * c);
  };
}

function computeModel(p) {
  const { aHat, aSig, apHat, apSig, lam, r, costo, pmin, pmax, nP, nS, seed, floor } = p;
  const P = new Float64Array(nP), v = new Float64Array(nP), vsym = new Float64Array(nP);
  for (let j = 0; j < nP; j++) {
    P[j] = pmin + (j * (pmax - pmin)) / (nP - 1);
    v[j] = Math.max(0, r - P[j]) - lam * Math.max(0, P[j] - r);
    vsym[j] = Math.max(0, r - P[j]) - 1 * Math.max(0, P[j] - r);
  }
  const rng = mulberry32(seed >>> 0);
  const nrm = makeNormal(rng);
  const A = new Float64Array(nS), AP = new Float64Array(nS);
  for (let k = 0; k < nS; k++) { A[k] = nrm(aHat, aSig); AP[k] = Math.max(floor, nrm(apHat, apSig)); }

  const sum = new Float64Array(nP);
  const mx = new Float64Array(nP).fill(-Infinity);
  const mn = new Float64Array(nP).fill(Infinity);
  const meanProfit = new Float64Array(nP);     // mundo actual (λ)
  const meanProfitSym = new Float64Array(nP);  // mundo simétrico (λ=1)
  const sumDem = new Float64Array(nP);
  const row = new Float64Array(nP);

  const K = Math.min(14, nS);                  // escenarios de muestra para las nubes
  const sampDem = Array.from({ length: K }, () => new Float64Array(nP));
  const sampProf = Array.from({ length: K }, () => new Float64Array(nP));

  for (let k = 0; k < nS; k++) {
    let orac = -Infinity;
    for (let j = 0; j < nP; j++) {
      let d = A[k] + AP[k] * v[j]; if (d < 0) d = 0;
      const pf = d * (P[j] - costo);
      row[j] = pf; if (pf > orac) orac = pf; meanProfit[j] += pf; sumDem[j] += d;
      if (k < K) { sampDem[k][j] = d; sampProf[k][j] = pf; }
      let ds = A[k] + AP[k] * vsym[j]; if (ds < 0) ds = 0;
      meanProfitSym[j] += ds * (P[j] - costo);
    }
    for (let j = 0; j < nP; j++) {
      const rg = orac - row[j];
      sum[j] += rg; if (rg > mx[j]) mx[j] = rg; if (rg < mn[j]) mn[j] = rg;
    }
  }

  const esp = new Float64Array(nP);
  let je = 0, jw = 0, js = 0;
  for (let j = 0; j < nP; j++) {
    esp[j] = sum[j] / nS;
    if (esp[j] < esp[je]) je = j;
    if (mx[j] < mx[jw]) jw = j;
    if (meanProfitSym[j] > meanProfitSym[js]) js = j;
  }

  // Curva para el gráfico (submuestreada si nP es grande, para que dibuje liviano)
  const stride = Math.max(1, Math.floor(nP / 220));
  const curve = [];
  for (let j = 0; j < nP; j += stride) {
    curve.push({ p: P[j], esperado: esp[j], peor: mx[j], mejor: mn[j], band: [mn[j], mx[j]] });
  }
  if (curve[curve.length - 1].p !== P[nP - 1])
    curve.push({ p: P[nP - 1], esperado: esp[nP - 1], peor: mx[nP - 1], mejor: mn[nP - 1], band: [mn[nP - 1], mx[nP - 1]] });

  // Curvas de demanda y profit (nubes de escenarios + curva media / estimada)
  const stride2 = Math.max(1, Math.floor(nP / 90));
  const demCurve = [], profCurve = [];
  for (let j = 0; j < nP; j += stride2) {
    const dRow = { p: P[j], media: sumDem[j] / nS, est: Math.max(0, aHat + apHat * v[j]) };
    const pRow = { p: P[j], media: meanProfit[j] / nS };
    for (let i = 0; i < K; i++) { dRow["s" + i] = sampDem[i][j]; pRow["s" + i] = sampProf[i][j]; }
    demCurve.push(dRow); profCurve.push(pRow);
  }

  // Histograma: distribución a precio COMÚN = p_g* (mundo actual), λ=1 vs λ actual
  const jStar = je;
  const demSym = new Float64Array(nS), demAsym = new Float64Array(nS);
  const profSym = new Float64Array(nS), profAsym = new Float64Array(nS);
  const vc = v[jStar], vs = vsym[jStar], marg = P[jStar] - costo;
  for (let k = 0; k < nS; k++) {
    let da = A[k] + AP[k] * vc; if (da < 0) da = 0;
    let dsm = A[k] + AP[k] * vs; if (dsm < 0) dsm = 0;
    demAsym[k] = da; demSym[k] = dsm;
    profAsym[k] = da * marg; profSym[k] = dsm * marg;
  }

  const profAtStar = meanProfit[je] / nS;        // profit real en la decisión correcta
  const profAtSymDec = meanProfit[js] / nS;      // profit real si decides con λ=1 (corner)
  const costIgnore = profAtStar - profAtSymDec;

  return {
    curve, demCurve, profCurve, sampleK: K,
    pgStar: P[je], pgWorst: P[jw], pgStarSym: P[js], step: P[1] - P[0],
    pmax, costIgnore, profAtStar,
    hist: { demSym, demAsym, profSym, profAsym, priceStar: P[jStar] },
  };
}

function binData(arrA, arrB, nbins) {
  let lo = Infinity, hi = -Infinity;
  for (const x of arrA) { if (x < lo) lo = x; if (x > hi) hi = x; }
  for (const x of arrB) { if (x < lo) lo = x; if (x > hi) hi = x; }
  if (hi <= lo) hi = lo + 1;
  const w = (hi - lo) / nbins;
  const out = [];
  for (let b = 0; b < nbins; b++) out.push({ mid: lo + (b + 0.5) * w, sim: 0, asim: 0 });
  for (const x of arrA) { let b = Math.floor((x - lo) / w); if (b >= nbins) b = nbins - 1; if (b < 0) b = 0; out[b].sim++; }
  for (const x of arrB) { let b = Math.floor((x - lo) / w); if (b >= nbins) b = nbins - 1; if (b < 0) b = 0; out[b].asim++; }
  return out;
}

const fmtCLP = (v) => "$" + Math.round(v).toLocaleString("es-CL");
const fmtM = (v) => "$" + (v / 1e6).toFixed(1) + "M";

const DEFAULTS = {
  aHat: 200, aSig: 20, apHat: 0.008, apSig: 0.003, lam: 2.25,
  r: 12000, costo: 3000, pmin: 5000, pmax: 20000,
  nP: 200, nS: 700, seed: 42, floor: 0.000001,
};

const COL = {
  ink: "#1c1917", muted: "#78716c", line: "#e7e5e4", paper: "#fcfcfb",
  panel: "#ffffff", esp: "#dc2626", peor: "#ea580c", mejor: "#15803d",
  band: "#fb923c", star: "#ca8a04", worst: "#15803d", sim: "#94a3b8", asim: "#4f46e5",
};

function Slider({ label, hint, value, min, max, step, onChange, format }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <span style={{ fontSize: 12.5, color: COL.ink, fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: 12.5, color: COL.ink, fontFamily: "ui-monospace, Menlo, monospace" }}>
          {format ? format(value) : value}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: COL.ink }} />
      {hint && <div style={{ fontSize: 10.5, color: COL.muted, marginTop: 2, lineHeight: 1.3 }}>{hint}</div>}
    </div>
  );
}

function GroupTitle({ children }) {
  return (
    <div style={{
      fontSize: 10.5, letterSpacing: 1.5, textTransform: "uppercase",
      color: COL.muted, fontWeight: 700, margin: "20px 0 10px", borderBottom: `1px solid ${COL.line}`, paddingBottom: 5,
    }}>{children}</div>
  );
}

function Stat({ label, value, color, note }) {
  return (
    <div style={{ flex: "1 1 0", minWidth: 130 }}>
      <div style={{ fontSize: 10.5, letterSpacing: 0.6, textTransform: "uppercase", color: COL.muted, fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 23, fontFamily: "ui-monospace, Menlo, monospace", fontWeight: 700, color: color || COL.ink, lineHeight: 1.25 }}>{value}</div>
      {note && <div style={{ fontSize: 10.5, color: COL.muted, marginTop: 1 }}>{note}</div>}
    </div>
  );
}

export default function App() {
  const [p, setP] = useState(DEFAULTS);
  const [metric, setMetric] = useState("profit"); // 'profit' | 'demanda'
  const set = (k) => (val) => setP((s) => ({ ...s, [k]: val }));

  const m = useMemo(() => computeModel({
    ...p, nP: Math.round(p.nP), nS: Math.round(p.nS), seed: Math.round(p.seed),
  }), [p]);

  const histRaw = metric === "profit"
    ? [m.hist.profSym, m.hist.profAsym] : [m.hist.demSym, m.hist.demAsym];
  const histData = useMemo(() => binData(histRaw[0], histRaw[1], 28), [histRaw, metric]);

  const cornerHit = Math.abs(m.pgStarSym - m.pmax) < m.step * 1.5;

  return (
    <div style={{ background: COL.paper, color: COL.ink, padding: "22px 24px 40px", minHeight: "100%",
      fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif" }}>
      <style>{`input[type=range]{height:4px;border-radius:4px;background:${COL.line};outline:none;}
        input[type=range]::-webkit-slider-thumb{appearance:none;width:14px;height:14px;border-radius:50%;background:${COL.ink};cursor:pointer;}`}</style>

      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        {/* Encabezado */}
        <div style={{ marginBottom: 6 }}>
          <div style={{ fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: COL.muted, fontWeight: 700 }}>
            Análisis de sensibilidad · Trabajo de título
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 800, margin: "2px 0 0", letterSpacing: -0.3 }}>
            Regret-Grid con aversión a la pérdida
          </h1>
        </div>

        {/* Lectura en vivo */}
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap", background: COL.panel,
          border: `1px solid ${COL.line}`, borderRadius: 10, padding: "16px 18px", margin: "16px 0 18px" }}>
          <Stat label="p_g*  (esperado)" value={fmtCLP(m.pgStar)} color={COL.star} note="= máx profit esperado" />
          <Stat label="p_g_worst  (minimax)" value={fmtCLP(m.pgWorst)} color={COL.worst} note="precio robusto peor caso" />
          <Stat label="p_g* sin aversión (λ=1)" value={fmtCLP(m.pgStarSym)} color={COL.sim}
            note={cornerHit ? "⚠ solución de esquina (tope de grilla)" : "óptimo interior"} />
          <Stat label="Costo de ignorar aversión" value={fmtCLP(m.costIgnore)} color={COL.asim}
            note={`${(100 * m.costIgnore / Math.max(1, m.profAtStar)).toFixed(0)}% del profit · paso grilla ${fmtCLP(m.step)}`} />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 22, alignItems: "start" }}>
          {/* Panel de controles */}
          <div style={{ background: COL.panel, border: `1px solid ${COL.line}`, borderRadius: 10, padding: "8px 18px 20px" }}>
            <GroupTitle>Demanda</GroupTitle>
            <Slider label="α̂ — nivel base" value={p.aHat} min={50} max={400} step={1} onChange={set("aHat")} />
            <Slider label="σ_α — incert. del nivel" value={p.aSig} min={0} max={60} step={1} onChange={set("aSig")} />
            <Slider label="α̂_p — sensibilidad al precio" value={p.apHat} min={0.001} max={0.02} step={0.0005}
              onChange={set("apHat")} format={(v) => v.toFixed(4)} />
            <Slider label="σ_αp — incert. de la pendiente" value={p.apSig} min={0} max={0.008} step={0.0002}
              onChange={set("apSig")} format={(v) => v.toFixed(4)}
              hint="Si es 0, p_g_worst colapsa sobre p_g*: la separación del minimax vive acá." />

            <GroupTitle>Comportamiento (Kahneman)</GroupTitle>
            <Slider label="λ — aversión a la pérdida" value={p.lam} min={1} max={4} step={0.05}
              onChange={set("lam")} format={(v) => v.toFixed(2)}
              hint="λ=1 → simétrico (sin aversión). Empírico K-T ≈ 2.25." />
            <Slider label="r — precio de referencia" value={p.r} min={8000} max={16000} step={100} onChange={set("r")} format={fmtCLP} />

            <GroupTitle>Costo y grilla</GroupTitle>
            <Slider label="costo unitario" value={p.costo} min={1000} max={6000} step={100} onChange={set("costo")} format={fmtCLP} />
            <Slider label="precio mínimo" value={p.pmin} min={2000} max={10000} step={500} onChange={set("pmin")} format={fmtCLP} />
            <Slider label="precio máximo" value={p.pmax} min={15000} max={30000} step={500} onChange={set("pmax")} format={fmtCLP} />
            <Slider label="N° de precios (resolución)" value={p.nP} min={60} max={400} step={20} onChange={set("nP")}
              hint="Más puntos = óptimos más finos, recálculo más lento." />

            <GroupTitle>Simulación</GroupTitle>
            <Slider label="piso del clip de α_p" value={p.floor} min={0} max={0.006} step={0.0001}
              onChange={set("floor")} format={(v) => v.toFixed(4)}
              hint="Súbelo para eliminar escenarios casi-planos. Ese artefacto (piso ≈ 0) es el que domina el peor caso." />
            <Slider label="N° de escenarios" value={p.nS} min={200} max={1000} step={100} onChange={set("nS")} />
            <Slider label="semilla" value={p.seed} min={0} max={200} step={1} onChange={set("seed")} />

            <button onClick={() => setP(DEFAULTS)}
              style={{ marginTop: 18, width: "100%", padding: "9px 0", borderRadius: 8, border: `1px solid ${COL.ink}`,
                background: COL.ink, color: "#fff", fontSize: 12.5, fontWeight: 600, cursor: "pointer" }}>
              Volver a valores de la tesis
            </button>
          </div>

          {/* Gráficos */}
          <div>
            {/* Demanda y Profit por precio */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, marginBottom: 18 }}>
              <div style={{ background: COL.panel, border: `1px solid ${COL.line}`, borderRadius: 10, padding: "14px 14px 6px" }}>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 2 }}>Demanda por precio</div>
                <div style={{ fontSize: 11.5, color: COL.muted, marginBottom: 8 }}>
                  14 escenarios (azul) · demanda estimada (naranjo) · quiebre en r
                </div>
                <ResponsiveContainer width="100%" height={224}>
                  <ComposedChart data={m.demCurve} margin={{ top: 6, right: 10, bottom: 4, left: 4 }}>
                    <CartesianGrid stroke={COL.line} strokeDasharray="3 3" />
                    <XAxis dataKey="p" type="number" domain={[p.pmin, p.pmax]} tickFormatter={fmtCLP}
                      tick={{ fontSize: 10, fill: COL.muted }} />
                    <YAxis tick={{ fontSize: 10, fill: COL.muted }} width={34} />
                    {Array.from({ length: m.sampleK }).map((_, i) => (
                      <Line key={i} dataKey={"s" + i} stroke="#93c5fd" dot={false} strokeWidth={0.7}
                        strokeOpacity={0.5} isAnimationActive={false} legendType="none" />
                    ))}
                    <Line dataKey="est" stroke={COL.peor} dot={false} strokeWidth={2.4} isAnimationActive={false} legendType="none" />
                    <ReferenceLine x={p.r} stroke="#a78bfa" strokeWidth={1.2} strokeDasharray="3 3" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              <div style={{ background: COL.panel, border: `1px solid ${COL.line}`, borderRadius: 10, padding: "14px 14px 6px" }}>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 2 }}>Profit por precio</div>
                <div style={{ fontSize: 11.5, color: COL.muted, marginBottom: 8 }}>
                  14 escenarios (rosa) · profit medio (rojo) · máximo en p_g*
                </div>
                <ResponsiveContainer width="100%" height={224}>
                  <ComposedChart data={m.profCurve} margin={{ top: 6, right: 10, bottom: 4, left: 4 }}>
                    <CartesianGrid stroke={COL.line} strokeDasharray="3 3" />
                    <XAxis dataKey="p" type="number" domain={[p.pmin, p.pmax]} tickFormatter={fmtCLP}
                      tick={{ fontSize: 10, fill: COL.muted }} />
                    <YAxis tickFormatter={fmtM} tick={{ fontSize: 10, fill: COL.muted }} width={42} />
                    {Array.from({ length: m.sampleK }).map((_, i) => (
                      <Line key={i} dataKey={"s" + i} stroke="#fca5a5" dot={false} strokeWidth={0.7}
                        strokeOpacity={0.5} isAnimationActive={false} legendType="none" />
                    ))}
                    <Line dataKey="media" stroke="#b91c1c" dot={false} strokeWidth={2.6} isAnimationActive={false} legendType="none" />
                    <ReferenceLine x={m.pgStar} stroke={COL.star} strokeWidth={1.6} strokeDasharray="5 3" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Regret por precio */}
            <div style={{ background: COL.panel, border: `1px solid ${COL.line}`, borderRadius: 10, padding: "14px 16px 6px", marginBottom: 18 }}>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 2 }}>Regret por precio</div>
              <div style={{ fontSize: 11.5, color: COL.muted, marginBottom: 8 }}>
                banda mejor–peor caso · líneas de decisión en p_g* (oro) y p_g_worst (verde)
              </div>
              <ResponsiveContainer width="100%" height={290}>
                <ComposedChart data={m.curve} margin={{ top: 6, right: 12, bottom: 4, left: 8 }}>
                  <CartesianGrid stroke={COL.line} strokeDasharray="3 3" />
                  <XAxis dataKey="p" type="number" domain={[p.pmin, p.pmax]} tickFormatter={fmtCLP}
                    tick={{ fontSize: 11, fill: COL.muted }} />
                  <YAxis tickFormatter={fmtM} tick={{ fontSize: 11, fill: COL.muted }} width={48} />
                  <Tooltip isAnimationActive={false}
                    labelFormatter={(v) => "Precio " + fmtCLP(v)}
                    formatter={(val, name) => [fmtCLP(val), name]} />
                  <Area dataKey="band" stroke="none" fill={COL.band} fillOpacity={0.13} isAnimationActive={false} name="rango" />
                  <Line dataKey="peor" stroke={COL.peor} dot={false} strokeWidth={1.8} isAnimationActive={false} name="peor caso" />
                  <Line dataKey="mejor" stroke={COL.mejor} dot={false} strokeWidth={1.6} strokeDasharray="3 3" isAnimationActive={false} name="mejor caso" />
                  <Line dataKey="esperado" stroke={COL.esp} dot={false} strokeWidth={2.6} isAnimationActive={false} name="esperado" />
                  <ReferenceLine x={m.pgStar} stroke={COL.star} strokeWidth={1.6} strokeDasharray="5 3" />
                  <ReferenceLine x={m.pgWorst} stroke={COL.worst} strokeWidth={1.6} strokeDasharray="5 3" />
                  <ReferenceLine x={p.r} stroke="#a78bfa" strokeWidth={1} strokeDasharray="2 3" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Histograma sensibilidad β */}
            <div style={{ background: COL.panel, border: `1px solid ${COL.line}`, borderRadius: 10, padding: "14px 16px 8px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 2 }}>
                    Histograma: β simétrico (λ=1) vs asimétrico (λ={p.lam.toFixed(2)})
                  </div>
                  <div style={{ fontSize: 11.5, color: COL.muted }}>
                    mismos escenarios, evaluados al precio común p_g* = {fmtCLP(m.hist.priceStar)}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  {["profit", "demanda"].map((opt) => (
                    <button key={opt} onClick={() => setMetric(opt)}
                      style={{ padding: "5px 11px", borderRadius: 7, fontSize: 12, cursor: "pointer", fontWeight: 600,
                        border: `1px solid ${metric === opt ? COL.ink : COL.line}`,
                        background: metric === opt ? COL.ink : "#fff", color: metric === opt ? "#fff" : COL.muted }}>
                      {opt === "profit" ? "Profit" : "Demanda"}
                    </button>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={histData} margin={{ top: 6, right: 12, bottom: 4, left: 8 }} barCategoryGap="8%">
                  <CartesianGrid stroke={COL.line} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="mid" type="number" domain={["dataMin", "dataMax"]}
                    tickFormatter={(v) => (metric === "profit" ? fmtM(v) : Math.round(v))}
                    tick={{ fontSize: 11, fill: COL.muted }} />
                  <YAxis tick={{ fontSize: 11, fill: COL.muted }} width={36}
                    label={{ value: "N° escenarios", angle: -90, position: "insideLeft", style: { fontSize: 10.5, fill: COL.muted } }} />
                  <Tooltip isAnimationActive={false}
                    labelFormatter={(v) => (metric === "profit" ? fmtCLP(v) : "Demanda " + Math.round(v))} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="sim" fill={COL.sim} name="β simétrico (λ=1)" isAnimationActive={false} />
                  <Bar dataKey="asim" fill={COL.asim} name={`β asimétrico (λ=${p.lam.toFixed(2)})`} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}