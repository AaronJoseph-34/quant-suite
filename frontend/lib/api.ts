export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }

  return (await res.json()) as T;
}
