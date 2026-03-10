# OCR Models Research Report

**Date:** March 6, 2025  
**Scope:** Ollama, Hugging Face, accuracy comparison, long-document strategies

---

## 1. OCR Models on Ollama

Ollama does **not** ship dedicated OCR models. It provides **vision-capable LLMs** that can perform OCR and document understanding via image input.

### Vision/Document Models Available

| Model | Size | OCR Capability | Notes |
|-------|------|----------------|-------|
| **minicpm-v** | 8B | Strong | Best OCR on Ollama; outperforms GPT-4o on OCRBench; 1.8M pixel support; 75% fewer tokens |
| **llava** | 7B / 13B / 34B | Good | Text recognition, document/diagram analysis; LLaVA 1.6 improves OCR; up to 672×672 px |
| **llama3.2-vision** | 8B | Good | Alternative for OCR; widely used in Ollama-OCR tools |
| **qwen2.5vl** | 7B | Good | Multimodal, suitable for document understanding |
| **qwen3-vl** | — | Good | Newer Qwen vision model |
| **granite3.2-vision** | — | Good | IBM vision model |
| **moondream** | ~1.8B | Basic | Lightweight vision model |
| **bakllava** | — | Basic | LLaVA variant |

### Pros & Cons

| Pros | Cons |
|------|------|
| Local inference, no API cost | Not purpose-built for OCR |
| Simple API (`ollama run llava`, Python/JS) | Slower than dedicated OCR engines |
| Good for document Q&A and reasoning | Accuracy below specialized OCR models |
| **MiniCPM-V** is strong for OCR | Long docs need chunking |

**Recommendation:** Use **minicpm-v** for OCR on Ollama; **llava** or **llama3.2-vision** as fallbacks.

---

## 2. Best OCR Models on Hugging Face (Document PDFs / Scanned Images)

### Primary Models

| Model | Type | Best For | HF Hub |
|-------|------|----------|--------|
| **Donut** | OCR-free, end-to-end | Document understanding, parsing, receipts | `naver-clova-ix/donut-*` |
| **TrOCR** | Transformer OCR | Printed & handwritten text recognition | `microsoft/trocr-*` |
| **LayoutLM / LayoutLMv2** | Layout + text | Forms, receipts, document QA | `microsoft/layoutlm*` |
| **PaddleOCR (PP-OCRv5)** | Detection + recognition | Production, multi-language, tables | `PaddlePaddle/PP-OCRv5_*` |
| **PaddleOCR-VL-1.5** | 0.9B VLM | Document parsing, 109 languages | `PaddlePaddle/PaddleOCR-VL-1.5` |
| **EasyOCR** | Detection + recognition | Easy setup, 80+ languages | pip package (uses PyTorch) |
| **GLM-OCR** | Vision-language | General document OCR | `zai-org/GLM-OCR` |
| **FireRed-OCR** | Image-to-text | Trending OCR model | `FireRedTeam/FireRed-OCR` |

### Model Characteristics

**Donut**
- OCR-free: no separate OCR step
- Swin encoder + BART decoder
- 91.3% on CORD (receipt parsing)
- Good for receipts, invoices, document parsing

**TrOCR**
- Image encoder + autoregressive decoder
- ~97.4% on typed text, ~79.5% on handwritten
- Slower than Tesseract (~200×)
- Good for printed and handwritten text

**LayoutLM**
- Needs pre-OCR (e.g. Tesseract) for text + bboxes
- Strong on FUNSD (forms), SROIE (receipts), DocVQA
- LayoutLMv2: FUNSD 84.2%, DocVQA 86.7%

**PaddleOCR**
- PP-OCRv5: detection + recognition
- PaddleOCR-VL-1.5: 94.5% on OmniDocBench v1.5
- 109 languages, irregular layouts, seals
- Production-ready, widely used

**EasyOCR**
- Simple API, 80+ languages
- Better on complex scenes than Tesseract
- Prefers RGB; weaker on grayscale

---

## 3. Accuracy & Use Cases by Document Type

### Benchmarks (2024–2025)

| Document Type | Best Performers | Accuracy |
|---------------|-----------------|----------|
| **Printed text** | Azure Document Intelligence, GPT-5, Gemini 2.5 Pro | ~95–96% |
| **Complex layouts** | Gemini 2.5 Pro, Google Vision, Claude Sonnet | ~85% |
| **Handwriting** | GPT-5, olmOCR-2-7B | 93–95% |
| **Tables** | Tensorlake (cloud) | 86.8% TEDS |
| **Tables (open-source)** | PaddleOCR, LayoutLMv2 | <70% TEDS |
| **Forms / structured** | LayoutLMv2, Tensorlake | 84–92% F1 |

### Use Case Recommendations

| Use Case | Recommended Models | Notes |
|----------|-------------------|-------|
| **Tables** | PaddleOCR-VL, LayoutLMv2, cloud APIs | Open-source table parsing lags cloud |
| **Forms** | LayoutLM/LayoutLMv2, Donut | LayoutLM needs OCR preprocessing |
| **Scientific docs** | Donut, PaddleOCR-VL, TrOCR | Formulas/symbols still challenging |
| **Receipts/invoices** | Donut, LayoutLM | Donut is OCR-free |
| **Handwritten** | TrOCR (handwritten), GPT-5 | TrOCR strong for handwritten |
| **Multi-language** | PaddleOCR, EasyOCR | 80–109 languages |
| **Offline / local** | PaddleOCR, EasyOCR, Donut | No API dependency |

### Caveats

- LLMs (GPT-4, Claude, etc.) can hallucinate; avoid for high-precision extraction.
- Traditional OCR and document parsers are better when 100% fidelity matters.
- PaddleOCR and EasyOCR perform worse on grayscale images (trained on RGB).

---

## 4. Long Documents (50+ Pages): Chunking & Processing

### Strategies

| Strategy | Pros | Cons |
|----------|------|------|
| **Page-based** | Simple, preserves page boundaries | May split tables/figures |
| **Fixed-size** | Predictable, easy batching | Cuts mid-sentence/paragraph |
| **Semantic** | Keeps meaning together | Needs model, slower |
| **Layout-based** | Respects sections, tables, figures | Needs layout detection |
| **Heading-based** | Section-aware | Needs heading detection |

### Recommended Approach for 50+ Pages

1. **Page-based chunking** (e.g. 5–10 pages per chunk) for speed and simplicity.
2. **Layout analysis** (e.g. Unstructured.io, Chunkr) to avoid splitting tables/figures.
3. **Async / parallel processing** (e.g. Mistral chunking script, Chunktopus).
4. **Constraints:** max pages per chunk, max file size, API limits.

### Tools & Libraries

- **Mistral AI Cookbook** – chunking for large PDFs with page/size limits.
- **Chunktopus** – multithreaded OCR for PDF/DOCX/PPTX with configurable chunking.
- **Unstructured.io** – multimodal chunking (text, tables, images).
- **Chunkr** – layout-based segmentation.
- **ChuLo** – semantic chunking via keyphrase extraction.

### Practical Flow

```
PDF (50+ pages) → Split by pages (e.g. 5–10) → OCR per chunk (parallel) 
→ Merge results → Optional: semantic re-chunking for RAG/QA
```

---

## 5. Summary & Recommendations

### Quick Reference

| Need | Recommendation |
|------|----------------|
| **Ollama (local)** | minicpm-v for OCR; llava / llama3.2-vision as alternatives |
| **Hugging Face, general docs** | Donut (OCR-free) or PaddleOCR-VL-1.5 |
| **Tables** | PaddleOCR-VL, LayoutLMv2, or cloud APIs |
| **Forms** | LayoutLMv2, Donut |
| **Handwritten** | TrOCR (handwritten checkpoints) |
| **Multi-language** | PaddleOCR, EasyOCR |
| **Long docs (50+ pages)** | Page-based chunking (5–10 pages) + parallel OCR + layout-aware splitting |

### Top 3 Picks

1. **PaddleOCR-VL-1.5** – Strong all-rounder for document parsing, 109 languages, 94.5% on OmniDocBench.
2. **Donut** – OCR-free, good for receipts/invoices, no separate OCR pipeline.
3. **minicpm-v (Ollama)** – Best local OCR option with vision LLMs.
