export type PortfolioData = { symbols: string[]; dates: string[]; prices: number[][] };

/** A wide CSV supplies shared dates, so assets cannot silently shift in time. */
export function parsePortfolioRows(fields: string[], rows: Record<string, string>[]): PortfolioData {
  const dateKey = fields.find((field) => field.toLowerCase() === "date");
  const symbols = fields.filter((field) => field !== dateKey);
  if (!dateKey || symbols.length < 2 || new Set(fields).size !== fields.length) {
    throw new Error("Use unique columns Date,AAPL,MSFT (at least two assets).");
  }
  const seen = new Set<string>();
  const parsed = rows.map((row) => {
    const date = row[dateKey]?.trim();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(date)) || new Date(date).toISOString().slice(0, 10) !== date || seen.has(date)) {
      throw new Error("Dates must be unique, valid YYYY-MM-DD values.");
    }
    seen.add(date);
    const prices = symbols.map((symbol) => Number(row[symbol]));
    if (prices.some((price) => !Number.isFinite(price) || price <= 0)) {
      throw new Error(`Every asset needs a positive price on ${date}.`);
    }
    return { date, prices };
  }).sort((a, b) => a.date.localeCompare(b.date));
  if (parsed.length < 3) throw new Error("Need at least three shared dates.");
  return {
    symbols, dates: parsed.map((row) => row.date),
    prices: symbols.map((_, i) => parsed.map((row) => row.prices[i])),
  };
}
