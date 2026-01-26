"use client";

import React, { useMemo, useState } from "react";
import PriceChart from "./components/PriceChart";

import Papa from "papaparse";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Area,
  CartesianGrid,
} from "recharts";
import { apiPost } from "../lib/api";

type ForecastResp = {
  method: "ml" | "arima";
  horizon: number;
  forecast: number[];
  low: number[];
  high: number[];
  vol_est: number;
  last_price: number;
};

type BSResp = {
  price: number;
  greeks: { delta: number; gamma: number; vega: number; theta: number; rho: number };
};

type IVResp = { implied_vol: number };

type MCResp = {
  S0: number;
  terminal_mean: number;
  terminal_p05: number;
  terminal_p95: number;
  payoff_mean?: number;
  payoff_p95?: number;
};

type FetchResp = { symbol: string; dates: string[]; prices: number[]; error?: string };

type PortfolioRiskResp = {
  summary: {
    VaR: number;
    CVaR: number;
    mean: number;
    vol: number;
  };
  rolling: {
    VaR: number[];
    CVaR: number[];
  };
  correlation: number[][];
  stress: Record<string, {
    mean: number;
    vol: number;
    VaR: number;
  }>;
};


function Card(props: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div style={styles.card}>
      <div style={{ marginBottom: 12 }}>
        <div style={styles.cardTitle}>{props.title}</div>
        {props.subtitle && <div style={styles.cardSub}>{props.subtitle}</div>}
      </div>
      {props.children}
    </div>
  );
}

function Stat(props: { label: string; value: string }) {
  return (
    <div style={styles.stat}>
      <div style={styles.statLabel}>{props.label}</div>
      <div style={styles.statValue}>{props.value}</div>
    </div>
  );
}

function pill(active: boolean) {
  return {
    ...styles.pill,
    ...(active ? styles.pillActive : styles.pillInactive),
  } as const;
}

function toCSVPrices(prices: number[]) {
  return prices.map((x) => String(x)).join(",");
}

export default function Home() {
  // Store dates separately for chart labeling (Phase 2)
  const [dates, setDates] = useState<string[]>([]);
  const [symbol, setSymbol] = useState<string>("AAPL");

  // Keep prices as comma text (feeds existing pipeline)
  const [pricesText, setPricesText] = useState(
    "100,101,102,101,103,104,105,106,108,107,109,110,111,112,113,114,113,115,116,118,117,119,120,121,122,121,123,124,125,126,127,128,129,130"
  );

  const prices = useMemo(() => {
    return pricesText
      .split(",")
      .map((x) => Number.parseFloat(x.trim()))
      .filter((x) => Number.isFinite(x));
  }, [pricesText]);

  const [tab, setTab] = useState<"data" | "forecast" | "options" | "risk">("data");

  // Forecast inputs
  const [method, setMethod] = useState<"ml" | "arima">("ml");
  const [horizon, setHorizon] = useState<number>(10);

  // Option inputs
  const [S, setS] = useState<number>(130);
  const [K, setK] = useState<number>(130);
  const [T, setT] = useState<number>(1);
  const [r, setR] = useState<number>(0.03);
  const [sigma, setSigma] = useState<number>(0.25);
  const [type, setType] = useState<"call" | "put">("call");

  // Outputs
  const [forecast, setForecast] = useState<ForecastResp | null>(null);
  const [bs, setBs] = useState<BSResp | null>(null);
  const [iv, setIv] = useState<IVResp | null>(null);
  const [mc, setMc] = useState<MCResp | null>(null);

  const [busy, setBusy] = useState<string>("");
  const [err, setErr] = useState<string>("");

  const canForecast = prices.length >= 30;

  const chartData = useMemo(() => {
    // If dates exist, label x-axis with index but tooltip shows date.
    const rows: any[] = prices.map((p, i) => ({
      t: i,
      date: dates[i] ?? String(i),
      history: p,
    }));

    if (forecast) {
      const base = prices.length - 1;
      for (let i = 0; i < forecast.forecast.length; i++) {
        rows.push({
          t: base + i + 1,
          date: dates[base + i + 1] ?? `+${i + 1}`,
          forecast: forecast.forecast[i],
          low: forecast.low[i],
          high: forecast.high[i],
        });
      }
      if (prices.length && forecast.forecast.length) {
        rows.push({
          t: base + 0.5,
          date: "bridge",
          history: prices[prices.length - 1],
          forecast: forecast.forecast[0],
        });
      }
    }

    return rows.sort((a, b) => a.t - b.t);
  }, [prices, dates, forecast]);

  async function runForecast() {
    setErr("");
    setBusy("forecast");
    try {
      const data = await apiPost<ForecastResp>("/forecast", { prices, horizon, method });
      setForecast(data);
      setS(Number((data.forecast?.[0] ?? data.last_price).toFixed(6)));
      setTab("forecast");
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally {
      setBusy("");
    }
  }

  async function priceBS() {
    setErr("");
    setBusy("bs");
    try {
      const data = await apiPost<BSResp>("/pricing/black_scholes", { S, K, T, r, sigma, type });
      setBs(data);
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally {
      setBusy("");
    }
  }

  async function solveIV() {
    setErr("");
    setBusy("iv");
    try {
      const target = bs?.price ?? 10;
      const data = await apiPost<IVResp>("/pricing/implied_vol", { price: target, S, K, T, r, type });
      setIv(data);
      if (Number.isFinite(data.implied_vol)) setSigma(Number(data.implied_vol.toFixed(6)));
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally {
      setBusy("");
    }
  }

  async function runMC() {
    setErr("");
    setBusy("mc");
    try {
      const data = await apiPost<MCResp>("/sim/gbm_paths", {
        S0: S,
        mu: 0.10,
        sigma,
        T,
        steps: 252,
        sims: 30000,
        K,
      });
      setMc(data);
      setTab("options");
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally {
      setBusy("");
    }
  }

  // -----------------------
  // Phase 2: CSV upload
  // -----------------------
  function handleCSV(file: File) {
    setErr("");
    setBusy("csv");
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        try {
          const rows = results.data as Record<string, any>[];

          // Accept common column names
          const dateKey =
            findKey(rows[0], ["date", "Date", "datetime", "Datetime", "timestamp", "Timestamp"]) ?? null;
          const closeKey =
            findKey(rows[0], ["close", "Close", "adj close", "Adj Close", "adjclose", "AdjClose", "price", "Price"]) ??
            null;

          if (!closeKey) throw new Error("CSV missing a Close/Price column.");

          const parsedDates: string[] = [];
          const parsedPrices: number[] = [];

          for (const r of rows) {
            const px = Number.parseFloat(String(r[closeKey] ?? "").trim());
            if (!Number.isFinite(px)) continue;

            parsedPrices.push(px);
            parsedDates.push(dateKey ? String(r[dateKey]) : String(parsedPrices.length - 1));
          }

          if (parsedPrices.length < 30) throw new Error(`Need at least 30 rows; got ${parsedPrices.length}.`);

          setDates(parsedDates);
          setPricesText(toCSVPrices(parsedPrices));
          setSymbol("CSV");
          setForecast(null);
          setBs(null);
          setIv(null);
          setMc(null);
          setTab("forecast");
        } catch (e: any) {
          setErr(String(e?.message || e));
        } finally {
          setBusy("");
        }
      },
      error: (error) => {
        setErr(String(error?.message || error));
        setBusy("");
      },
    });
  }

  function findKey(obj: Record<string, any> | undefined, candidates: string[]) {
    if (!obj) return null;
    const keys = Object.keys(obj);
    const lowered = new Map(keys.map((k) => [k.toLowerCase(), k]));
    for (const c of candidates) {
      const k = lowered.get(c.toLowerCase());
      if (k) return k;
    }
    return null;
  }

  // -----------------------
  // Phase 2: Live fetch
  // -----------------------
  const [period, setPeriod] = useState("1y");
  const [interval, setInterval] = useState("1d");

  async function fetchLive() {
    setErr("");
    setBusy("fetch");
    try {
      const data = await apiPost<FetchResp>("/data/fetch", { symbol, period, interval });
      if (data.error) throw new Error(data.error);
      if (!data.prices?.length) throw new Error("No prices returned (symbol/period/interval may be invalid).");
      if (data.prices.length < 30) throw new Error(`Need >= 30 prices; got ${data.prices.length}. Try longer period.`);

      setDates(data.dates ?? []);
      setPricesText(toCSVPrices(data.prices));
      setForecast(null);
      setTab("forecast");
      setS(Number(data.prices[data.prices.length - 1].toFixed(6)));
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally {
      setBusy("");
    }
  }
const [risk, setRisk] = useState<PortfolioRiskResp | null>(null);
async function runPortfolioRisk() {
  setErr("");
  setBusy("risk");

  try {
    const data = await apiPost<PortfolioRiskResp>(
      "/risk/portfolio/auto",
      { prices: [prices], alpha: 0.05 }
    );
    setRisk(data);
    setTab("risk");
  } catch (e: any) {
    setErr(String(e?.message || e));
  } finally {
    setBusy("");
  }
}

  return (
    <main style={styles.page}>
      <div style={styles.header}>
        <div>
          <div style={styles.h1}>Quant Suite</div>
          <div style={styles.h2}>Market Data • Forecast • Options • Risk</div>
        </div>
        <div style={styles.badgeRow}>
          <span style={styles.badge}>Symbol: {symbol || "—"}</span>
          <span style={styles.badge}>N: {prices.length}</span>
        </div>
      </div>

      {err && (
        <div style={styles.errorBox}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Error</div>
          <div style={{ whiteSpace: "pre-wrap" }}>{err}</div>
        </div>
      )}

      <div style={styles.tabs}>
        <button style={pill(tab === "data")} onClick={() => setTab("data")}>Market Data</button>
        <button style={pill(tab === "forecast")} onClick={() => setTab("forecast")}>Forecast</button>
        <button style={pill(tab === "options")} onClick={() => setTab("options")}>Options</button>
        <button style={pill(tab === "risk")} onClick={() => setTab("risk")}>Risk</button>
      </div>

      {tab === "data" && (
        <div style={styles.grid2}>
          <Card title="Live Price Fetch" subtitle="Fetch adjusted close prices via backend (yfinance).">
            <div style={styles.row}>
              <label style={styles.inlineLabel}>
                Ticker
                <input value={symbol} onChange={(e) => setSymbol(e.target.value)} style={styles.input} />
              </label>
              <label style={styles.inlineLabel}>
                Period
                <select value={period} onChange={(e) => setPeriod(e.target.value)} style={styles.select}>
                  <option value="1mo">1mo</option>
                  <option value="3mo">3mo</option>
                  <option value="6mo">6mo</option>
                  <option value="1y">1y</option>
                  <option value="2y">2y</option>
                  <option value="5y">5y</option>
                  <option value="max">max</option>
                </select>
              </label>
              <label style={styles.inlineLabel}>
                Interval
                <select value={interval} onChange={(e) => setInterval(e.target.value)} style={styles.select}>
                  <option value="1d">1d</option>
                  <option value="1wk">1wk</option>
                  <option value="1mo">1mo</option>
                  <option value="1h">1h</option>
                  <option value="15m">15m</option>
                  <option value="5m">5m</option>
                  <option value="1m">1m</option>
                </select>
              </label>
              <button onClick={fetchLive} disabled={busy !== ""} style={styles.primaryBtn}>
                {busy === "fetch" ? "Fetching..." : "Fetch Prices"}
              </button>
            </div>

            <div style={{ marginTop: 10, color: "#9ca3af", fontSize: 13 }}>
              Tip: if interval is small (1m/5m), use shorter period like 1mo/3mo.
            </div>
          </Card>

          <Card title="CSV Upload" subtitle="Upload a CSV with columns like Date, Close (or date, close).">
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleCSV(f);
              }}
              style={{ ...styles.input, padding: 10 }}
            />
            <div style={{ marginTop: 10, color: "#9ca3af", fontSize: 13 }}>
              After upload, we auto-fill your chart + forecast inputs.
            </div>
          </Card>
        </div>
      )}

      {tab === "forecast" && (
        <div style={styles.grid2}>
          <Card title="Price History & Forecast" subtitle="History + forecast mean + band (low/high).">
            <div style={{ height: 320 }}>
  <PriceChart data={chartData} />
</div>

            {forecast && (
              <div style={styles.statRow}>
                <Stat label="Last price" value={forecast.last_price.toFixed(4)} />
                <Stat label="Vol est" value={forecast.vol_est.toFixed(6)} />
                <Stat
  label="Model"
  value={
    forecast.method === "ml"
      ? "XGB • LGBM • RF • ElasticNet"
      : "ARIMA"
  }
/>
                <Stat label="Horizon" value={`${forecast.horizon}`} />
              </div>
            )}
          </Card>

          <Card title="Forecast Controls" subtitle="Run ARIMA or ML forecast on your current dataset.">
            <label style={styles.label}>Prices (comma-separated)</label>
            <textarea value={pricesText} onChange={(e) => setPricesText(e.target.value)} rows={7} style={styles.textarea} />

            <div style={styles.row}>
              <label style={styles.inlineLabel}>
                Method
                <select value={method} onChange={(e) => setMethod(e.target.value as any)} style={styles.select}>
                  <option value="ml">Machine Learning</option>
                  <option value="arima">ARIMA</option>
                </select>
              </label>

              <label style={styles.inlineLabel}>
                Horizon
                <input type="number" min={1} max={60} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} style={styles.input} />
              </label>

              <button
                onClick={runForecast}
                disabled={!canForecast || busy !== ""}
                style={{ ...styles.primaryBtn, opacity: !canForecast || busy !== "" ? 0.6 : 1 }}
              >
                {busy === "forecast" ? "Running..." : "Run Forecast"}
              </button>
            </div>

            {!canForecast && (
              <div style={{ marginTop: 10, color: "#fecaca" }}>
                Need at least 30 prices (you have {prices.length}).
              </div>
            )}
          </Card>
        </div>
      )}

      {tab === "options" && (
  <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 14 }}>
          <Card title="Option Inputs" subtitle="Black-Scholes price + Greeks, plus implied volatility solve.">
            <div style={styles.formGrid}>
              <label style={styles.inlineLabel}>
                Spot (S)
                <input type="number" value={S} onChange={(e) => setS(Number(e.target.value))} style={styles.input} />
              </label>
              <label style={styles.inlineLabel}>
                Strike (K)
                <input type="number" value={K} onChange={(e) => setK(Number(e.target.value))} style={styles.input} />
              </label>
              <label style={styles.inlineLabel}>
                Time (T, yrs)
                <input type="number" step="0.01" value={T} onChange={(e) => setT(Number(e.target.value))} style={styles.input} />
              </label>
              <label style={styles.inlineLabel}>
                Rate (r)
                <input type="number" step="0.001" value={r} onChange={(e) => setR(Number(e.target.value))} style={styles.input} />
              </label>
              <label style={styles.inlineLabel}>
                Vol (sigma)
                <input type="number" step="0.01" value={sigma} onChange={(e) => setSigma(Number(e.target.value))} style={styles.input} />
              </label>
              <label style={styles.inlineLabel}>
                Type
                <select value={type} onChange={(e) => setType(e.target.value as any)} style={styles.select}>
                  <option value="call">Call</option>
                  <option value="put">Put</option>
                </select>
              </label>
            </div>

            <div style={styles.row}>
              <button onClick={priceBS} disabled={busy !== ""} style={styles.primaryBtn}>
                {busy === "bs" ? "Pricing..." : "Black-Scholes + Greeks"}
              </button>
              <button onClick={solveIV} disabled={busy !== ""} style={styles.secondaryBtn}>
                {busy === "iv" ? "Solving..." : "Solve Implied Vol"}
              </button>
              <button onClick={runMC} disabled={busy !== ""} style={styles.secondaryBtn}>
                {busy === "mc" ? "Simulating..." : "Run Monte Carlo"}
              </button>
            </div>

            {bs && (
              <div style={styles.statRow}>
                <Stat label="BS Price" value={bs.price.toFixed(6)} />
                <Stat label="Delta" value={bs.greeks.delta.toFixed(6)} />
                <Stat label="Gamma" value={bs.greeks.gamma.toFixed(6)} />
                <Stat label="Vega" value={bs.greeks.vega.toFixed(6)} />
                <Stat label="Theta" value={bs.greeks.theta.toFixed(6)} />
                <Stat label="Rho" value={bs.greeks.rho.toFixed(6)} />
              </div>
            )}

            {iv && <div style={{ marginTop: 10, color: "#e5e7eb" }}>Implied vol: <b>{iv.implied_vol.toFixed(6)}</b></div>}
            {mc && (
              <div style={styles.statRow}>
                <Stat label="Terminal mean" value={mc.terminal_mean.toFixed(4)} />
                <Stat label="Terminal p05" value={mc.terminal_p05.toFixed(4)} />
                <Stat label="Terminal p95" value={mc.terminal_p95.toFixed(4)} />
                <Stat label="Payoff mean" value={(mc.payoff_mean ?? 0).toFixed(4)} />
              </div>
            )}
          </Card>

          
        </div>
      )}

      {tab === "risk" && (
  <div style={styles.grid2}>
    <Card title="Portfolio Risk Summary" subtitle="VaR / CVaR @ 95%">
      <button
        onClick={runPortfolioRisk}
        disabled={busy !== ""}
        style={styles.primaryBtn}
      >
        {busy === "risk" ? "Computing..." : "Run Risk Analysis"}
      </button>

      {risk && (
        <div style={styles.statRow}>
          <Stat label="Mean Return" value={risk.summary.mean.toFixed(6)} />
          <Stat label="Volatility" value={risk.summary.vol.toFixed(6)} />
          <Stat label="VaR (95%)" value={risk.summary.VaR.toFixed(6)} />
          <Stat label="CVaR (95%)" value={risk.summary.CVaR.toFixed(6)} />
        </div>
      )}
    </Card>

    <Card title="Stress Scenarios">
      {risk &&
        Object.entries(risk.stress).map(([name, s]) => (
          <div key={name} style={{ marginBottom: 10 }}>
            <b>{name}</b><br />
            Mean: {s.mean.toFixed(6)} | Vol: {s.vol.toFixed(6)} | VaR: {s.VaR.toFixed(6)}
          </div>
        ))}
    </Card>
  </div>
)}


      <div style={styles.footer}>
        <div>Backend: <code>http://localhost:8000</code></div>
        <div>Frontend: <code>http://localhost:3000</code></div>
      </div>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    padding: 24,
    background: "linear-gradient(180deg, #0b1220 0%, #0b1220 40%, #0f172a 100%)",
    color: "#e5e7eb",
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial",
  },
  header: { display: "flex", justifyContent: "space-between", gap: 16, alignItems: "flex-end", marginBottom: 14 },
  h1: { fontSize: 28, fontWeight: 800, letterSpacing: -0.4 },
  h2: { marginTop: 6, color: "#9ca3af", fontSize: 14 },
  badgeRow: { display: "flex", gap: 8, flexWrap: "wrap" },
  badge: {
    fontSize: 12,
    color: "#cbd5e1",
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.10)",
    padding: "6px 10px",
    borderRadius: 999,
  },
  errorBox: {
    background: "rgba(239,68,68,0.10)",
    border: "1px solid rgba(239,68,68,0.35)",
    padding: 14,
    borderRadius: 14,
    marginBottom: 14,
    color: "#fecaca",
  },
  tabs: { display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" },
  pill: { borderRadius: 999, padding: "10px 14px", border: "1px solid rgba(255,255,255,0.12)", cursor: "pointer", fontWeight: 700, fontSize: 13 },
  pillActive: { background: "rgba(255,255,255,0.14)", color: "#ffffff" },
  pillInactive: { background: "rgba(0,0,0,0.15)", color: "#cbd5e1" },

  grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 },
  card: {
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.10)",
    borderRadius: 18,
    padding: 16,
    boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
    backdropFilter: "blur(8px)",
  },
  cardTitle: { fontSize: 16, fontWeight: 800, color: "#f8fafc" },
  cardSub: { marginTop: 4, fontSize: 13, color: "#9ca3af" },

  label: { display: "block", fontWeight: 800, fontSize: 13, color: "#e5e7eb", marginBottom: 6 },
  textarea: {
    width: "100%",
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.25)",
    color: "#e5e7eb",
    padding: 12,
    outline: "none",
    resize: "vertical",
  },
  row: { display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap", alignItems: "flex-end" },

  inlineLabel: { display: "flex", flexDirection: "column", gap: 6, fontSize: 12, color: "#cbd5e1", fontWeight: 700 },
  input: {
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.25)",
    color: "#e5e7eb",
    padding: "10px 12px",
    outline: "none",
    minWidth: 140,
  },
  select: {
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.25)",
    color: "#e5e7eb",
    padding: "10px 12px",
    outline: "none",
    minWidth: 160,
  },

  primaryBtn: {
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.18)",
    background: "rgba(255,255,255,0.14)",
    color: "#ffffff",
    padding: "11px 14px",
    cursor: "pointer",
    fontWeight: 900,
  },
  secondaryBtn: {
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.18)",
    background: "rgba(0,0,0,0.18)",
    color: "#e5e7eb",
    padding: "11px 14px",
    cursor: "pointer",
    fontWeight: 900,
  },

  statRow: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginTop: 12 },
  stat: { background: "rgba(0,0,0,0.18)", border: "1px solid rgba(255,255,255,0.10)", borderRadius: 14, padding: 12 },
  statLabel: { fontSize: 12, color: "#9ca3af", fontWeight: 800 },
  statValue: { marginTop: 6, fontSize: 16, fontWeight: 900, color: "#f8fafc" },

  formGrid: { display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10 },
  footer: { marginTop: 14, color: "#94a3b8", fontSize: 12 },
};
