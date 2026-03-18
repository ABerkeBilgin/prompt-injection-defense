## 📂 Proje Mimarisi

Proje, yapay zeka araştırma standartları ile yazılım mühendisliği disiplinini birleştiren aşağıdaki modüler yapıya sahiptir:

```text
prompt-injection-defense/
├── .github/                      # Takım çalışması ve otomasyon ayarları (PR, Issue şablonları)
├── app/                          # Proje bitiminde sunulacak canlı demo arayüzü (Streamlit/Gradio)
├── data/                         # Veri setleri (Git'e eklenmez)
│   ├── processed/                # İşlenmiş, modelin okuyabileceği veriler
│   └── raw/                      # Cleaned Alpaca, TaskTracker gibi ham veriler
├── docs/                         # Tüm yazılı içerikler ve akademik belgeler
│   ├── literatur/                # Baz makale (DefensiveTokens) ve referans makaleler
│   ├── raporlar/                 # İlerleme raporları ve inceleme notları
│   └── tez_taslagi/              # Bitirme tezinin taslak dosyaları
├── notebooks/                    # Keşifsel veri analizi ve grafik (ASR vs Utility) çizimleri
├── scripts/                      # Eğitim ve test işlemlerini terminalden başlatan bash betikleri
├── src/                          # Asıl Python kaynak kodları (Modüler yapı)
│   ├── data/                     # Veri indirme ve temizleme kodları
│   ├── evaluation/               # ASR ve WinRate hesaplama metotları
│   └── model/                    # Model yükleme ve token optimizasyon mantığı
├── .gitignore                    # İstenmeyen dosyaların (ağırlıklar, büyük veriler) Git'e gitmesini engelleyen kurallar
├── docker-compose.yml            # Demo arayüzünü izole ortamda ayağa kaldırma dosyası
├── Dockerfile                    # Sanal ortam kurulum adımları
├── LICENSE                       # Projenin açık kaynak lisansı
├── Makefile                      # "make train", "make eval" gibi terminal komut kısayolları
├── README.md                     # Projenin vitrini
└── requirements.txt              # Gerekli Python kütüphaneleri listesi