const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ??
  "http://localhost:8000";

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  console.log("API_BASE:", API_BASE);
  console.log("Calling:", `${API_BASE}${path}`);

  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }

  return res.json() as Promise<T>;
}
