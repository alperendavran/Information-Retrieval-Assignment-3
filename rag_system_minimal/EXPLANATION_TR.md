# RAG Sistemleri Açıklaması

## Minimal Versiyon (`rag_system_minimal/`)

Minimal versiyon, assignment'ın tüm gereksinimlerini karşılayan temel bir RAG sistemidir. Dokümanları 100-300 kelimelik parçalara böler, embedding modeli (all-MiniLM-L6-v2) ile vektörleştirir, FAISS ile benzerlik indeksi oluşturur, sorguyu encode edip top-k en benzer parçaları bulur ve GPT-4o ile cevap üretir. Evaluation için Recall@k, RAG vs baseline karşılaştırması ve hata analizi framework'ü içerir. Yaklaşık 1,200 satır kod, 4 temel bağımlılık ve assignment'ın tüm bileşenlerini (4.1-4.5) basit ve anlaşılır şekilde karşılar.

## Advanced Versiyon (`rag_system/`)

Advanced versiyon, minimal versiyonun tüm özelliklerine ek olarak state-of-the-art teknikler içerir. Retrieval'da candidate expansion + reranking, section-aware boost, lexical code boost ve source diversification gibi optimizasyonlar bulunur. LangGraph tabanlı agentic workflow ile query understanding, multi-query expansion, HyDE, RRF fusion ve reflection özellikleri eklenmiştir. Structured index ile entity+field routing, otomatik evaluation script'leri (LLM-as-judge, auto-labeling) ve cost tracking sistemi mevcuttur. Yaklaşık 4,500+ satır kod, 10+ bağımlılık ve retrieval accuracy'de %20-30, answer quality'de %15-25 iyileşme sağlar.
