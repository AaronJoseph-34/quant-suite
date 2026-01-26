export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const API_BASE =
    process.env.NEXT_PUBLIC_API_URL ??
    "http://localhost:8000";
  
  // TEMPORARY DEBUG - Remove after fixing
  console.log("Environment variable:", process.env.NEXT_PUBLIC_API_URL);
  console.log("API_BASE:", API_BASE);
  console.log("Full URL:", `${API_BASE}${path}`);
  
  const res = await fetch(`${API_BASE}${path}`, {
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
