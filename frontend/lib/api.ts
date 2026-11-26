export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const base = (process.env.NEXT_PUBLIC_API_BASE?.trim() || "http://localhost:8000").replace(/\/+$/, "");
  const response = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map((e: { loc: string[]; msg: string }) => `${e.loc.slice(1).join(".")}: ${e.msg}`).join("; ")
      : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
