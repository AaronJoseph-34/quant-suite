"use client";

import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Area,
  CartesianGrid,
} from "recharts";

type ChartRow = {
  t: number;
  date?: string;
  history?: number;
  forecast?: number;
  low?: number;
  high?: number;
  bbUpper?: number;
  bbLower?: number;
  bbMid?: number;
};

function computeBollinger(
  data: ChartRow[],
  window = 20,
  k = 2
): ChartRow[] {
  return data.map((d, i) => {
    if (i < window || d.history == null) return d;

    const slice = data
      .slice(i - window, i)
      .map((x) => x.history!)
      .filter(Number.isFinite);

    if (slice.length < 2) return d;
    const m = slice.reduce((sum, value) => sum + value, 0) / slice.length;
    const sd = Math.sqrt(slice.reduce((sum, value) => sum + (value - m) ** 2, 0) / (slice.length - 1));

    return {
      ...d,
      bbUpper: m + k * sd,
      bbLower: m - k * sd,
      bbMid: m,
    };
  });
}

export default function PriceChart({ data }: { data: ChartRow[] }) {
  const enriched = useMemo(() => computeBollinger(data).map((row) => ({
    ...row,
    forecastBand: row.low != null && row.high != null ? [row.low, row.high] : undefined,
  })), [data]);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={enriched}>
        {/* Grid */}
        <CartesianGrid
          stroke="rgba(148,163,184,0.15)"
          strokeDasharray="3 3"
        />

        <XAxis
          dataKey="t"
          tick={{ fontSize: 12, fill: "#CBD5E1" }}
        />
        <YAxis
          tick={{ fontSize: 12, fill: "#CBD5E1" }}
          domain={["auto", "auto"]}
        />

        <Tooltip
          contentStyle={{
            background: "#020617",
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: 10,
            color: "#E5E7EB",
            fontSize: 12,
          }}
          formatter={(value, name) => [String(value), String(name ?? "")]}

          labelFormatter={(label) => {
            const row = enriched.find((x) => x.t === label);
            return row?.date
              ? `t=${label} • ${row.date}`
              : `t=${label}`;
          }}
        />

        {/* Bollinger Upper Band */}
        <Area
          type="monotone"
          dataKey="bbUpper"
          stroke="none"
          fill="#60A5FA"
          fillOpacity={0.18}
          isAnimationActive={false}
        />

        {/* Bollinger Lower Band (used as mask) */}
        <Area
          type="monotone"
          dataKey="bbLower"
          stroke="none"
          fill="#0B1220"
          fillOpacity={1}
          isAnimationActive={false}
        />

        {/* Bollinger Mid */}
        <Line
          type="monotone"
          dataKey="bbMid"
          stroke="#94A3B8"
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />

        {/* Historical Price */}
        <Line
          type="monotone"
          dataKey="history"
          stroke="#E5E7EB"
          strokeWidth={2.2}
          dot={false}
          isAnimationActive={false}
        />

        <Area type="monotone" dataKey="forecastBand" name="Approx. 95% band"
          stroke="none" fill="#22D3EE" fillOpacity={0.16} isAnimationActive={false} />

        {/* Forecast */}
        <Line
          type="monotone"
          dataKey="forecast"
          stroke="#22D3EE"
          strokeWidth={2.2}
          strokeDasharray="6 6"
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
