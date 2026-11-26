import assert from "node:assert/strict";
import { test } from "node:test";
import { parsePortfolioRows } from "./portfolio.ts";

const fields = ["Date", "AAPL", "MSFT"];
const rows = [
  { Date: "2026-01-03", AAPL: "99", MSFT: "198" },
  { Date: "2026-01-01", AAPL: "100", MSFT: "200" },
  { Date: "2026-01-02", AAPL: "110", MSFT: "180" },
];

test("sorts shared dates and constructs an asset-by-time matrix", () => {
  const data = parsePortfolioRows(fields, rows);
  assert.deepEqual(data.symbols, ["AAPL", "MSFT"]);
  assert.deepEqual(data.prices, [[100, 110, 99], [200, 180, 198]]);
});

test("rejects missing, invalid, or misaligned observations", () => {
  for (const value of ["", "0", "-1", "NaN", "abc"]) {
    assert.throws(() => parsePortfolioRows(fields, [{ ...rows[0], MSFT: value }, ...rows.slice(1)]));
  }
  assert.throws(() => parsePortfolioRows(fields, [rows[0], rows[0], rows[2]]));
  assert.throws(() => parsePortfolioRows(fields, [{ ...rows[0], Date: "2026-02-30" }, ...rows.slice(1)]));
  assert.throws(() => parsePortfolioRows(["Date", "AAPL"], rows));
  assert.throws(() => parsePortfolioRows(fields, rows.slice(0, 2)));
});
