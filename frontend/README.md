# Quant Suite dashboard

See the [repository README](../README.md) for setup, configuration, API contracts, and methodology.

```bash
npm ci
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_BASE` before building for deployment. Tests and checks:

```bash
npm test
npm run lint
npx tsc --noEmit
npm run build
```
