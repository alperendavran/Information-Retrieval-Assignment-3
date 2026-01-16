# 📚 RAG Sistemi Strateji ve Referans Belgesi

## İçindekiler
1. [Anahtar Kavramlar](#anahtar-kavramlar)
2. [Önerilen Stratejiler](#önerilen-stratejiler)
3. [Deney Planı](#deney-planı)
4. [Değerlendirme Metodolojisi](#değerlendirme-metodolojisi)
5. [Akademik Referanslar](#akademik-referanslar)

---

## 🔑 Anahtar Kavramlar

### RAG Paradigmaları (Kaynak: Ders Slaytı + RAG-overview.pdf)

| Paradigma | Özellikler | Güçlü Yanlar |
|-----------|-----------|--------------|
| **Naive RAG** | Keyword-based retrieval (TF-IDF, BM25), statik dataset | Basit, uygulaması kolay |
| **Advanced RAG** | Dense retrieval (DPR), neural ranking, multi-hop retrieval | Yüksek precision, contextual relevance |
| **Modular RAG** | Hybrid retrieval, tool integration, composable pipelines | Yüksek esneklik, ölçeklenebilir |
| **Graph RAG** | Knowledge graph entegrasyonu, community detection | Relational reasoning, global sensemaking |
| **Agentic RAG** | Autonomous agents, iterative refinement | Dinamik adaptasyon, multi-domain |

### RAG Temel Bileşenleri (Lewis et al., 2020)

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│  1. CHUNKING     →  2. EMBEDDING  →  3. INDEXING               │
│       ↓                  ↓               ↓                      │
│  4. RETRIEVAL   →  5. AUGMENTATION → 6. GENERATION             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Önerilen Stratejiler

### 1. Document Chunking Stratejileri (%10)

#### Strateji A: Fixed-Size Chunking with Overlap
```python
CHUNK_SIZE = 200-300  # tokens (önerilen aralık)
OVERLAP = 0.1-0.2     # %10-20 overlap
```
**Referans:** "Practical guidelines: Chunk size 200–300 tokens, Overlap 10–20%" (Ders Slaytı)

#### Strateji B: Semantic Chunking
- Cümle sınırlarında bölme
- Paragraf bazlı bölme (doğal breakpoint'ler varsa)
- Markdown/HTML yapısına göre bölme

**Kaynak:** "Keep semantic coherence: don't split mid-sentence or mid-section" (Ders Slaytı)

#### Strateji C: LLM-based Fact Extraction (Advanced)
```python
# GPT-4 ile bilgi yoğunluğunu artırma
# Raw HTML: ~55,000 tokens → GPT-4 processed: 330 tokens
```
**Referans:** 15_Advanced_RAG_Techniques.pdf, Technique 1

### 2. Embedding & Indexing Stratejileri (%20)

#### Önerilen Embedding Modelleri (Yerel Çalışabilir)

| Model | Boyut | Avantaj | Kullanım |
|-------|-------|---------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384d | Hızlı, hafif | Küçük dataset |
| `sentence-transformers/all-mpnet-base-v2` | 768d | Daha doğru | Orta dataset |
| `BAAI/bge-small-en-v1.5` | 384d | SOTA performans | Genel kullanım |
| `intfloat/e5-small-v2` | 384d | Çok dilli destek | Türkçe içerik |

#### Indexing Yaklaşımları

**Option 1: FAISS (Önerilen)**
```python
import faiss
index = faiss.IndexFlatIP(embedding_dim)  # Cosine similarity için normalize edilmiş vektörler
# veya
index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)  # Büyük datasetler için
```

**Option 2: Scikit-learn**
```python
from sklearn.metrics.pairwise import cosine_similarity
```

**Referans:** "Build a similarity index (e.g., cosine similarity, FAISS, or sklearn)" (Assignment)

### 3. Retrieval Module Stratejileri (%20)

#### Dense Passage Retrieval (DPR)
```
query q → BERTq → embq
                        ↘
                         cosine_similarity → Top-k
                        ↗
document d → BERTd → embd
```
**Referans:** Karpukhin et al. (2020), Lewis et al. (2020)

#### Optimal k Değeri
- **k=3-5:** Küçük dataset için önerilen (assignment dataseti)
- **k=1:** Çok küçük, yetersiz context
- **k=20:** Muhtemelen çok büyük, noise ekler

#### Hybrid Retrieval (Advanced)
```python
# BM25 + Dense retrieval kombinasyonu
final_score = α * bm25_score + (1-α) * dense_score
```
**Referans:** Modular RAG - "Hybrid retrieval strategies combining sparse and dense" (RAG-overview.pdf)

### 4. Answer Generation Stratejileri (%30)

#### Prompt Template
```python
SYSTEM_PROMPT = """You are a helpful assistant for the University of Antwerp 
Computer Science Masters program. Answer questions based ONLY on the provided 
context. If the context doesn't contain the answer, say "I don't have enough 
information to answer this question."

Context:
{retrieved_passages}

Question: {user_query}

Answer:"""
```

#### Few-shot Learning (Opsiyonel)
```python
# 2-3 örnek eklemek performansı artırabilir
examples = [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
]
```

**Referans:** "Use frozen LLMs (GPT-4, etc.) with zero-shot/few-shot learning" (Ders Slaytı)

### 5. Evaluation Stratejileri (%20)

#### 5.1 Retrieval Quality Metrics

**Recall@k:**
```python
def recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    return len(retrieved_set & relevant_set) / len(relevant_set)
```

**Mean Reciprocal Rank (MRR):**
```python
def mrr(retrieved_docs, relevant_doc):
    for i, doc in enumerate(retrieved_docs):
        if doc == relevant_doc:
            return 1 / (i + 1)
    return 0
```

#### 5.2 Answer Quality - LLM as a Judge

**RAGAS Framework (Referans: Es et al., 2024):**
- **Faithfulness:** Cevap context'e sadık mı?
- **Answer Relevance:** Cevap soruyla alakalı mı?
- **Context Relevance:** Getirilen context alakalı mı?

**Comprehensiveness & Diversity (GraphRAG kriterleri):**
```python
judge_prompt = """Compare the following two answers:
Answer A: {answer_with_rag}
Answer B: {answer_without_rag}

Evaluate on:
1. Comprehensiveness (0-100): How detailed and complete?
2. Correctness (0-100): Is the information accurate?
3. Hallucination (0-100): Does it contain made-up facts?
"""
```
**Referans:** Edge et al. (2024), Zheng et al. (2024) - LLM-as-a-judge

#### 5.3 Error Analysis Checklist

| Hata Tipi | Örnek Senaryo | Olası Sebep |
|-----------|---------------|-------------|
| **Retrieval Failure** | Alakasız dokümanlar getirildi | Embedding model uyumsuzluğu |
| **Context Missing** | Doğru bilgi chunk'lanırken kayboldu | Chunk size çok küçük |
| **Hallucination** | Yanlış bilgi üretildi | Context yetersiz / prompt zayıf |
| **Incomplete Answer** | Kısmi cevap | k değeri düşük |

---

## 🧪 Deney Planı

### Deney 1: Chunk Size Optimization
```
Değişken: chunk_size = [100, 200, 300, 500] tokens
Sabit: overlap=0.1, embedding=all-MiniLM-L6-v2, k=5
Metrik: Recall@5, Answer Quality
```

### Deney 2: Embedding Model Comparison
```
Değişken: model = [MiniLM, MPNet, BGE-small, E5-small]
Sabit: chunk_size=200, overlap=0.1, k=5
Metrik: Recall@5, Retrieval Latency
```

### Deney 3: Top-k Optimization
```
Değişken: k = [1, 3, 5, 7, 10]
Sabit: chunk_size=200, embedding=MiniLM
Metrik: Answer Quality, Token Usage
```

### Deney 4: RAG vs No-RAG Baseline
```
Condition A: GPT-4o with retrieved context
Condition B: GPT-4o without context (baseline)
Metrik: Correctness, Hallucination Rate
```

### Deney 5: Overlap Ratio Effect
```
Değişken: overlap = [0, 0.1, 0.2, 0.3]
Sabit: chunk_size=200
Metrik: Context Coverage, Retrieval Quality
```

---

## 📊 Değerlendirme Metodolojisi

### Test Soru Seti Oluşturma

**Adım 1: Manual Question Generation**
- Dataset'ten 20-30 test sorusu oluştur
- Ground truth cevapları belirle
- Soru kategorileri: Factual, Comparative, Complex

**Adım 2: LLM-Generated Questions (GraphRAG yaklaşımı)**
```python
prompt = """Based on this document about the CS Masters program, 
generate 5 questions that would require understanding the content:
{document}
"""
```

### Evaluation Pipeline

```
1. Test Query → 2. Retrieval → 3. Generation → 4. Evaluation
      ↓              ↓              ↓              ↓
   20-30 Q's    Recall@k      GPT-4o Answer   LLM-as-Judge
                                              + Manual Review
```

---

## 📖 Akademik Referanslar

### Temel RAG Makaleleri

1. **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020).** 
   Retrieval-augmented generation for knowledge-intensive NLP tasks. 
   *Advances in Neural Information Processing Systems, 33*, 9459-9474.
   - **Kullanım:** End-to-end RAG architecture, RAG-Sequence vs RAG-Token

2. **Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020).**
   Dense passage retrieval for open-domain question answering. 
   *EMNLP 2020*.
   - **Kullanım:** Dense Passage Retrieval (DPR), bi-encoder architecture

3. **Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2024).**
   Retrieval-Augmented Generation for Large Language Models: A Survey.
   *arXiv preprint arXiv:2312.10997*.
   - **Kullanım:** RAG paradigmalarının karşılaştırması (Naive, Advanced, Modular)

### İleri RAG Teknikleri

4. **Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., ... & Larson, J. (2024).**
   From Local to Global: A GraphRAG Approach to Query-Focused Summarization.
   *arXiv preprint arXiv:2404.16130*.
   - **Kullanım:** Knowledge graph + community detection, global sensemaking

5. **Singh, A., Ehtesham, A., Kumar, S., & Khoei, T. T. (2025).**
   Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.
   *arXiv preprint arXiv:2501.09136*.
   - **Kullanım:** Agentic patterns (reflection, planning, tool use)

6. **15 Advanced RAG Techniques (2024).**
   From pre-retrieval to generation. [White Paper]
   - **Kullanım:** Hierarchical indexing, hypothetical question index, chunking strategies

### Değerlendirme Metodları

7. **Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024).**
   RAGAS: Automated Evaluation of Retrieval Augmented Generation.
   *Proceedings of the 18th EACL: System Demonstrations*.
   - **Kullanım:** Faithfulness, Answer Relevance, Context Relevance metrikleri

8. **Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Xing, E. P. (2024).**
   Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
   *NeurIPS 2023*.
   - **Kullanım:** LLM-as-a-judge paradigması

### Ders Kaynakları

9. **Calders, T. (2025-2026).**
   Retrieval Augmented Generation [Lecture Slides].
   University of Antwerp, Information Retrieval Course.
   - **Kullanım:** RAG components, chunking guidelines, evaluation

10. **Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020).**
    Language models are few-shot learners.
    *Advances in Neural Information Processing Systems, 33*, 1877-1901.
    - **Kullanım:** Few-shot/zero-shot learning, frozen LLM kullanımı

11. **Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022).**
    Training language models to follow instructions with human feedback.
    *Advances in Neural Information Processing Systems, 35*, 27730-27744.
    - **Kullanım:** Fine-tuning LLMs, instruction following

---

## 🛠️ Önerilen Teknoloji Stack

### Python Kütüphaneleri

```python
# Core
sentence-transformers>=2.2.0  # Embedding models
faiss-cpu>=1.7.4             # Vector indexing
openai>=1.0.0                # GPT-4o API

# Yardımcı
tiktoken>=0.5.0              # Token counting
numpy>=1.24.0                # Numerical operations
pandas>=2.0.0                # Data handling
scikit-learn>=1.3.0          # Cosine similarity, metrics

# Opsiyonel
langchain>=0.1.0             # RAG framework (opsiyonel)
chromadb>=0.4.0              # Vector database alternative
```

### Proje Yapısı Önerisi

```
rag-project/
├── data/
│   ├── raw/                 # Orijinal veriler
│   ├── processed/           # Chunk'lanmış veriler
│   └── embeddings/          # Kaydedilmiş embeddings
├── src/
│   ├── chunking.py          # Document chunking modülü
│   ├── embedding.py         # Embedding modülü
│   ├── indexing.py          # FAISS indexing
│   ├── retrieval.py         # Retrieval modülü
│   ├── generation.py        # GPT-4o integration
│   └── evaluation.py        # Evaluation metrikleri
├── notebooks/
│   └── experiments.ipynb    # Deney notebookları
├── tests/
│   └── test_rag.py          # Unit tests
├── config.py                # Konfigürasyon
├── main.py                  # Ana uygulama
├── requirements.txt
└── README.md
```

---

## ⚡ Hızlı Başlangıç Kontrol Listesi

- [ ] Dataset'i incele (cs-data/ klasörü)
- [ ] Chunking stratejisi belirle (200-300 token önerilen)
- [ ] Embedding model seç (all-MiniLM-L6-v2 başlangıç için)
- [ ] FAISS index oluştur
- [ ] Retrieval fonksiyonu implement et (top-k=5)
- [ ] GPT-4o prompt template hazırla
- [ ] Test soruları oluştur (20-30 soru)
- [ ] Recall@k hesapla
- [ ] RAG vs No-RAG karşılaştırması yap
- [ ] Error analysis için 3+ retrieval hatası bul
- [ ] Error analysis için 3+ hallucination örneği bul
- [ ] Rapor yaz (4-6 sayfa)

---

*Bu belge, IR Assignment 3 için hazırlanmıştır. Son güncelleme: Ocak 2026*
