#  Utility Bill AI Agent — Case Study

## 

A production-grade, multi-provider AI document processing pipeline that:
- Extracts structured financial data from raw utility bills
- Generates 25-year solar financial projections
- Uses provider-aware LLM routing with cost optimization
- Implements streaming async processing and confidence scoring

---

> **Role:** AI/Backend Engineer (Primary Owner)
> **Domain:** Solar Energy Tech (SaaS Platform)  
> **Stack:** Python · FastAPI · Google Gemini · Claude · OpenAI · Tesseract OCR · Pydantic · asyncio  
> **Status:** Production — Actively processing real customer utility bills

---
## 🌐 Production Deployment

This architecture is deployed as part of a live solar intelligence platform:

🔗 **[https://solaragenthub.com](https://solaragenthub.com)**

🎬 **[Watch Live Demo](https://drive.google.com/file/d/1EjjCHNz0Z6NFpLfhmMHYcUYTvaFe_BtI/view?usp=sharing)** — end-to-end bill upload → solar proposal generation

> [!NOTE]
> - Production access requires paid credits.
> - Admin credentials cannot be shared.
> - This repository serves as a technical case study of the deployed system.

---
## 📌 Executive Summary

As part of a larger **multi-agent solar intelligence platform**, I designed and built an automated utility bill parsing and solar proposal generation engine from scratch. The system ingests raw utility bill files (PDF, PNG, JPEG, TIFF, BMP), extracts structured financial and usage data using a hybrid OCR-Vision pipeline, and autonomously generates personalized solar savings proposals including 25-year financial projections and environmental impact reports.

The architecture was engineered to be **resilient, cost-efficient, and provider-agnostic** — capable of routing AI requests across Google Gemini, Anthropic Claude, and OpenAI depending on task type, provider availability, and cost priorities.

---

## 🔍 Problem Statement

Solar sales teams were spending **3–5 minutes manually reviewing** every utility bill before a customer call — interpreting charge tiers, identifying usage patterns, and estimating potential solar savings. This created:

- A bottleneck that limited how many leads could be processed per rep per day
- High error rates due to inconsistent manual data entry
- No standardized output format for downstream CRM and proposal tools

**The objective:** Automate bill analysis end-to-end, from raw document → structured JSON → actionable solar proposal, with confidence scoring to flag uncertain extractions for human review.

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────────┐
                        │         FastAPI REST + WebSocket      │
                        │    /process  /report  /capabilities  │
                        └───────────────┬─────────────────────┘
                                        │ UploadFile (PDF/Image)
                                        ▼
                        ┌─────────────────────────────────────┐
                        │          UtilityBillAgent            │
                        │  (AsyncGenerator · streaming mode)   │
                        └───────────────┬─────────────────────┘
                                        │
                        ┌───────────────▼─────────────────────┐
                        │          UtilityBillService          │
                        │      (Orchestration Layer)           │
                        └──────┬────────────────┬─────────────┘
                               │                │
               ┌───────────────▼───┐    ┌───────▼──────────────┐
               │   VisionClient    │    │     LLM Gateway       │
               │  (Tesseract OCR)  │    │  (Unified AI Router)  │
               │     [Fallback]    │    │      [Primary]        │
               └───────────────────┘    └──────────────────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          ▼                      ▼                      ▼
                   ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
                   │   Gemini    │      │    Claude    │      │   OpenAI     │
                   │ (Text/OCR)  │      │ (Vision/NLP) │      │  (Vision)   │
                   └─────────────┘      └──────────────┘      └──────────────┘
                                        
                        ┌───────────────────────────────────────┐
                        │         Post-Processing Pipeline       │
                        ├────────────────┬──────────────────────┤
                        │ NormalizationService │ FinancialModel │
                        ├──────────────────────┤                │
                        │  SolarAnalyzerService│ TelemetryService│
                        ├──────────────────────┤                │
                        │  ReportGeneratorService (PDF output)  │
                        └───────────────────────────────────────┘
```

---

## ⚙️ Processing Pipeline — 8 Stages

Each uploaded bill goes through a deterministic, fault-tolerant 8-stage pipeline:

| Stage | Service | Description |
|-------|---------|-------------|
| **1** | `UtilityBillAgent` | File validation, MIME-type check, 50MB size guard |
| **2** | `VisionClient` / LLM Vision | AI direct-image analysis (primary); Tesseract OCR (fallback) |
| **3** | `UtilityBillService` | LLM Gateway prompt construction & provider selection |
| **4** | `NormalizationService` | Raw AI output → typed, validated `BillData` Pydantic model |
| **5** | `FinancialModelService` | Missing-month imputation, bill cost projection, battery storage recommendation |
| **6** | `SolarAnalyzerService` | System sizing, 25-year financial model, environmental impact |
| **7** | `TelemetryService` | Confidence scoring, extraction quality assessment, event logging |
| **8** | `ReportGeneratorService` | PDF report generation with charts |

---

## 🧠 Key Engineering Decisions

### 1. Vision-First, OCR-Fallback Strategy

```python
# Prefer direct AI vision analysis for images — more accurate than OCR pipeline
if file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp']:
    try:
        raw_data = await self._extract_bill_data_from_image(file_content, filename)
        return normalize(raw_data)  # Skip OCR entirely
    except Exception:
        logger.warning("Vision failed, falling back to OCR...")
        # fallthrough to Tesseract OCR pipeline
```

**Why:** Direct LLM vision is ~40% more accurate on well-formatted bills than OCR → parse pipelines, because vision models understand layout context (tables, labels, sections) rather than relying on raw text proximity.

---

### 2. Provider-Aware Intelligent Routing

The LLM Gateway routes requests by capability type, not just availability:

```python
# Image understanding priority: Claude > Gemini > OpenAI
preferred_image_order = ['claude', 'gemini', 'openai']

# Text generation priority: Gemini > Claude > OpenAI (cost optimized)
preferred_text_order = ['gemini', 'claude', 'openai']
```

**Why:** Claude 3 Opus has best-in-class document vision. Gemini Flash is the most cost-efficient for high-volume text-to-JSON extraction. This routing reduces API cost by ~60% compared to using Claude for all tasks.

---

### 3. Structured Extraction with JSON-Constrained Prompting

Rather than free-form extraction, I engineered prompts that:
- Enforce strict JSON output (no markdown, no preamble)
- Define field-by-field extraction rules including date normalization
- Distinguish `new_charges` (current period) vs `amount_due` (including arrears) — a critical distinction for accurate solar savings calculation

```
CRITICAL: Your response must be ONLY a valid JSON object.
Start your response immediately with { and end with }
```

With a multi-layer fallback parser:
1. Direct `json.loads()` parse
2. Regex-based `{...}` extraction from noisy responses  
3. Safe null-filled default structure (never a crash)

---

### 4. Financial Model — 25-Year Solar Projection

The financial model uses industry-standard assumptions and derives real rate data from the bill itself before falling back to national averages:

```python
# Priority: use actual tier rates → new_charges/kWh → national average ($0.16/kWh)
cost_per_kwh = self._estimate_cost_per_kwh(bill_data)

# 25-year savings accounting for 0.5% annual panel degradation
for year in range(1, 26):
    year_production = annual_kwh * (1 - 0.005) ** (year - 1)
    total_savings += year_production * cost_per_kwh
total_savings -= net_installation_cost  # Net of 30% federal tax credit
```

**Financial outputs:**
- Gross & net system cost (post 30% ITC)
- Estimated monthly savings
- Simple payback period (years)
- 25-year net savings
- Battery storage sizing for peak shaving

---

### 5. Asynchronous Streaming Architecture

The agent implements Python's `AsyncGenerator` pattern, supporting both batch and streaming response modes via the same code path:

```python
async def process(self, ..., stream: bool = False) -> AsyncGenerator[Dict, None]:
    for file in files:
        if stream:
            async for result in self._process_streaming(...):
                yield result  # Real-time progress to WebSocket
        else:
            result = await self._process_single(...)
            yield result     # Single final response
```

This enables the frontend to show real-time progress for large multi-file batch submissions.

---

### 6. Data Imputation for Incomplete Bills

Many bills only show the current month. The `FinancialModelService` fills missing months using statistical mean imputation:

```python
# If ≥3 months available, fill missing months with average
if len(existing_data) >= 3:
    average = statistics.mean(existing_values)
    for month in all_12_months:
        if month not in existing_data:
            imputed_data.append(YearUsage(month=month, kwh=average))
            imputed_count += 1
```

The `imputed_months_count` field in the response tells downstream consumers exactly how many months were inferred vs real.

---

## 📁 Project Structure (Sanitized)

```
utility_bill_agent/
├── utility_bill_agent.py           # Agent entry point, AsyncGenerator process()
├── models/
│   └── utility_bill_models/
│       └── utility_bill_models.py  # Pydantic models: BillData, SolarAnalysis, FinancialInsights
├── routers/
│   └── utility_bill_router.py      # FastAPI routes: /process, /report, /capabilities
└── services/
    └── utility_bill_services/
        ├── utility_bill_service.py  # Orchestration, LLM calls, response parsing
        ├── vision_client.py         # Tesseract OCR: PDF → images → text
        ├── postprocess.py           # NormalizationService: raw dict → BillData
        ├── financial_model.py       # FinancialModelService: cost projection & imputation
        ├── solar_analyzer.py        # SolarAnalyzerService: system sizing & 25yr model
        ├── report_generator.py      # PDF report with charts
        ├── telemetry.py             # Confidence scoring & event logging
        ├── llm_client.py            # Legacy direct Gemini client (pre-gateway)
        └── json_encoder.py          # CustomJsonEncoder for Decimal/date serialization
```

---

## 📊 Extracted Data Schema

The agent produces a fully typed, validated JSON output:

```json
{
  "bill_data": {
    "account_number": "1234567-89",
    "customer_name": "Jane Doe",
    "service_address": "123 Main St, San Jose, CA 95101",
    "billing_period_start": "2024-11-01",
    "billing_period_end": "2024-11-30",
    "new_charges": 187.42,
    "amount_due": 187.42,
    "kwh_tiers": [
      { "tier_name": "Baseline", "consumption_kwh": 300, "rate_per_kwh": 0.26, "period": "baseline" },
      { "tier_name": "Tier 2",   "consumption_kwh": 220, "rate_per_kwh": 0.31, "period": "tier_2" }
    ],
    "rate_structure": { "name": "E-1 Tiered Rate", "tou_schedule": [] },
    "year_usage": [
      { "month": "2024-01", "kwh": 520 },
      { "month": "2024-02", "kwh": 490 }
    ]
  },
  "financial_insights": {
    "projected_bill_cost": 186.20,
    "imputed_months_count": 4,
    "storage_recommendation_kwh": 12.5
  },
  "solar_analysis": {
    "proposal": {
      "recommended_system_size_kw": 7.5,
      "estimated_annual_production_kwh": 13061,
      "estimated_roof_space_sqft": 1875
    },
    "financials": {
      "estimated_system_cost": 26250.00,
      "federal_tax_credit_amount": 7875.00,
      "net_system_cost": 18375.00,
      "estimated_monthly_savings": 174.15,
      "simple_payback_period_years": 8.8,
      "estimated_25_year_savings": 34822.00
    },
    "environmental_impact": {
      "co2_emissions_avoided_tons": 52.3,
      "trees_planted_equivalent": 1830
    }
  },
  "confidence_score": 87,
  "processing_time": 4.21
}
```

---

## 🛡️ Reliability & Error Handling

| Scenario | Handling |
|----------|----------|
| Vision API failure | Automatic fallback to Tesseract OCR pipeline |
| OCR produces empty text | `ValueError` raised, caught at service layer |
| LLM returns malformed JSON | Multi-layer parser: direct → regex extraction → null-safe default |
| Missing Poppler/Tesseract | Descriptive error with setup guide reference |
| File > 50MB | Rejected at agent layer before any processing |
| All LLM providers unavailable | Exception propagated with clear message |
| Partial bill data | Statistical imputation for missing months |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/utility-bill/process` | Upload and process 1–N bill files |
| `POST` | `/utility-bill/report` | Generate PDF report from prior extraction result |
| `GET`  | `/utility-bill/capabilities` | Return agent capabilities & supported formats |

**Multi-file support:**  
A single API call can process a batch of files. Each file gets its own result object with `file_index` and `filename`, and a final aggregated response is returned.

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI (async, streaming) |
| Async Runtime | Python asyncio, `run_in_threadpool` |
| AI/LLM | Gemini 2.0 Flash (Vision + Structured JSON extraction), Claude 3 Opus/Haiku, GPT-4o |
| Document Extraction | Vision-first LLM parsing (primary) + Tesseract OCR + Poppler (fallback) |
| Data Validation | Pydantic v2 structured models |
| PDF Generation | FPDF2 + Matplotlib |
| Observability | Structured logging + custom TelemetryService |
| Containerization | Docker + Docker Compose |

---

## 🌍 External API Integrations

The agent orchestrates three categories of external APIs. Each integration includes timeout handling, retry logic, response validation, and graceful degradation — designed for production reliability.

---

### 1️⃣ Geocoding — Nominatim (OpenStreetMap)

**Why:** Utility bills provide a human-readable `service_address`. Solar production modeling requires `lat/lon`. Nominatim converts address → coordinates for free, without vendor lock-in.

**Data Flow:**
```
Extracted Address (LLM)  →  Nominatim API  →  lat/lon  →  PVWatts API
```

**Example Request:**
```
GET https://nominatim.openstreetmap.org/search
    ?q=123+Main+St+San+Jose+CA&format=json
```

**Response:**
```json
[{ "lat": "37.3382", "lon": "-121.8863", "display_name": "..." }]
```

**Engineering Decisions:**
- Custom `User-Agent` header (required by Nominatim ToS)
- Async HTTP client — non-blocking, no thread starvation
- Timeout guard + retry on transient failures
- Response schema validation before usage (checked for non-empty array)

**Failure Handling:**
- Empty or ambiguous result → fallback to regional average assumptions
- API timeout → telemetry event logged, pipeline continues with degraded confidence score

---

### 2️⃣ Solar Production Modeling — NREL PVWatts API

**Why:** Rather than building a physics simulation, the system uses NREL PVWatts — the industry-standard solar irradiance and production modeling API — to produce accurate, location-specific annual energy yield estimates.

**Data Flow:**
```
lat/lon + System Size (kW)  →  PVWatts API  →  ac_annual (kWh)  →  FinancialModelService
```

**Example Request:**
```
GET https://developer.nrel.gov/api/pvwatts/v8.json
    ?api_key=${NREL_API_KEY}
    &lat=37.3382
    &lon=-121.8863
    &system_capacity=7.5
    &module_type=1
    &losses=14
```

**Response (key fields extracted):**
```json
{
  "outputs": {
    "ac_annual": 13061,
    "capacity_factor": 18.5
  }
}
```

**Engineering Decisions:**
- API key stored in `.env` — never hardcoded
- Validated `outputs.ac_annual` existence before propagation
- 3–5 second timeout with retry for transient errors
- Fallback heuristic if PVWatts is unavailable: `annual_kwh = system_kw × 1500`

**Failure Handling:**
- API down → fallback estimation used, `confidence_score` reduced, telemetry event logged
- Invalid coordinates → exception caught at orchestration layer, proposal marked as estimated

---

### 3️⃣ Multi-Provider LLM APIs (Gemini / Claude / OpenAI)

**Why:** No single provider is optimal for every task type. Claude excels at visual document understanding; Gemini Flash is fastest and cheapest at high-volume text-to-JSON extraction; routing intelligently across them reduces cost and improves reliability.

**Routing Logic:**
```
Task Type?
  ├── Vision (image bill)   →  Claude → Gemini → OpenAI
  └── Text/JSON extraction  →  Gemini → Claude → OpenAI  (cost-optimized)
```

**Resilience Pattern:**
```python
for provider in preferred_order:
    try:
        return await llm_gateway.generate(provider=provider, ...)
    except RateLimitError:
        continue   # Try next provider
    except TimeoutError:
        continue
raise ValueError("All providers exhausted")
```

**Reliability Features:**

| Feature | Detail |
|---------|--------|
| JSON-constrained prompting | Prompts enforce `{...}` output, no preamble or markdown |
| Multi-layer response parser | `json.loads()` → regex `{...}` extraction → null-safe default |
| Temperature control | `temperature=0.1` — deterministic extraction, not creative generation |
| Token budget | `max_tokens=4000` to prevent runaway costs on large bills |
| Rate-limit handling | Provider skipped on `429`, next in priority list tried |

---

### 🔁 End-to-End Integration Flow

```
Upload Bill
    ↓
LLM Vision / OCR  →  Structured JSON (BillData)
    ↓
Extract service_address
    ↓
Nominatim  →  lat / lon
    ↓
NREL PVWatts  →  ac_annual kWh
    ↓
FinancialModelService  →  25-Year Savings, Payback Period
    ↓
ReportGeneratorService  →  PDF Output
```

> [!TIP]
> In interviews: this isn't just "API usage" — it's **orchestrated workflow automation** across geocoding, solar physics modeling, and multi-provider AI. Each external dependency has a defined fallback, so the pipeline degrades gracefully rather than failing hard.

---

## 📈 Results & Impact

- ⚡ **Bill processing time:** ~30–60 seconds (vs 3–5 minutes manual)
- 🎯 **Extraction confidence:** 80–95% on standard US utility formats
- 💰 **Manual review reduction:** Flagged low-confidence extractions (`confidence < 60`) for human review; auto-approved the rest
- 📦 **Multi-provider flexibility:** System continued operating during Gemini rate-limit periods by routing to Claude
- 🌱 **Scale:** Integrated into a platform serving multiple solar sales teams

---

## 💡 What I'd Improve Next

1. **Fine-tuned extraction model** — Fine-tune a small open-source model (Mistral / Llama 3) on labeled utility bill datasets to reduce per-call LLM API cost
2. **Bill format classifier** — Pre-classify bill by utility company (SCE, PG&E, Xcel, etc.) to use company-specific extraction templates before general prompting
3. **Structured output mode** — Replace free-form JSON prompting with provider structured output APIs (`response_format={"type": "json_schema"}`) for higher parse reliability
4. **Async parallel sub-services** — Run `FinancialModelService`, `SolarAnalyzerService`, and `TelemetryService` concurrently with `asyncio.gather()` to reduce total latency
5. **Caching layer** — Hash bill content and cache extraction results in Redis to avoid reprocessing identical submissions

---

## 🌐 Production Deployment

This architecture is deployed as part of a live solar intelligence platform:

🔗 **[https://solaragenthub.com](https://solaragenthub.com)**

> [!NOTE]
> - Production access requires paid credits.
> - Admin credentials cannot be shared.
> - This repository serves as a technical case study of the deployed system.

---

## 👨‍💻 About This Case Study

This repository is a **case study / documentation** of a production system I built. Source code has been sanitized and proprietary business logic has been removed. The architecture diagrams, pipeline descriptions, code snippets, and data schemas represent the actual system design.

> 📬 Interested in the full technical discussion? Feel free to reach out.

---

*Built with care for correctness, reliability, and cost-efficiency — because production AI systems are only as good as their worst failure mode.*


