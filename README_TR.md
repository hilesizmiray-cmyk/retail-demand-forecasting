# Perakende Talep Tahmini ve Stok Önerisi

10 mağaza ve 50 ürün için günlük satış tahmini yapan ve bu tahmini, gerçek taleple backtest edilmiş haftalık stok önerilerine dönüştüren bir proje.

**Sonuç:** Log-doğrusal regresyon modeli, tahmin hatasını mevsimsel naif baseline'a göre **%30 azaltıyor** (MAE 8,35 → 5,85). Elde edilen stok önerisi, %95 hedefe karşı mağaza-ürün-haftaların **%98,3'ünde** talebi karşılıyor.

---

## Problem

Bir perakendecinin her mağazada her üründen ne kadar stok tutması gerektiğini bilmesi gerekir. Az stok, boş raf ve kaçan satış demektir; fazla stok ise israf ve depoda bağlanan sermaye. Bu proje iki soruyu cevaplıyor:

1. **Tahmin:** Her mağaza her üründen günde kaç adet satacak?
2. **Karar:** Talebi güvenilir şekilde karşılamak için her mağaza haftalık ne kadar stok tutmalı?

## Veri

- **Kaynak:** [Kaggle – Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) (`train.csv`)
- **Büyüklük:** 913.000 satır — 10 mağaza × 50 ürün × günlük satış, 01.01.2013 – 31.12.2017
- **Alanlar:** `date` (tarih), `store` (mağaza), `item` (ürün), `sales` (satış)

Veri, yarışma kuralları gereği bu depoda yer almıyor. `train.csv` dosyasını Kaggle'dan indirip notebook ile aynı klasöre koyun.

## Yöntem

### 1. Veri profili
- Eksik veri yok; her mağaza-ürün çifti için günlük geçmiş eksiksiz.
- Üç belirgin örüntü:
  - **Haftalık döngü:** Satışlar hafta boyunca artıyor; Pazar, Pazartesi'den yaklaşık %50 fazla.
  - **Yıllık mevsimsellik:** Ocak'ta düşük, Temmuz'da zirve, Aralık'ta düşüş.
  - **Trend:** Ortalama günlük satış 2013'te ~43,5 iken 2017'de ~59'a çıktı.

### 2. Değerlendirme kurgusu
- **Test dönemi:** Son 3 ay (Ekim–Aralık 2017, 46.000 satır). Tüm modelleme kararlarından ayrı tutuldu.
- **Ölçüt:** Ortalama Mutlak Hata (MAE); birimi mağaza-ürün-gün başına adet.
- Tüm parametreler (büyüme oranı, hata tahminleri) veri sızıntısını önlemek için **yalnızca test öncesi veriden** hesaplandı.

### 3. Tahmin yöntemleri

| Yöntem | Açıklama | Test MAE |
|---|---|---|
| Mevsimsel naif (baseline) | 52 hafta önceki aynı gün | 8,35 |
| + Trend düzeltmesi | Baseline × yıllık büyüme oranı (%3,9) | 8,31 |
| Mevsimsel profil | 2016 ortalaması (mağaza × ürün × ay × haftanın günü) × büyüme | 6,41 |
| **Log-doğrusal regresyon** | `log(satış)` üzerinde doğrusal regresyon; mağaza, ürün, haftanın günü, ay (one-hot) ve yıl (trend) | **5,85** |

**Neden logaritma?** Haftanın günü ve mevsim etkileri toplamsal değil, çarpımsal (örneğin Pazar ≈ ×1,5). `log(satış)` modellenince doğrusal model yüzdesel etkileri öğrenebiliyor.

### 4. Stok önerisi
- Haftalık önerilen stok = haftalık tahmin + güvenlik stoğu
- Güvenlik stoğu = 1,65 × günlük tahmin hatasının standart sapması × √(haftadaki gün sayısı) → **%95 hizmet seviyesi** hedefi
- Tahmin hatasının standart sapması, her mağaza-ürün için Temmuz–Eylül 2017 (test öncesi) verisinden hesaplandı.

**Backtest:** Önerilen stok, mağaza-ürün-haftaların **%98,3'ünde** gerçek haftalık talebi karşıladı.

## Temel bulgular

- Trend düzeltmesi, baseline'ın sistematik düşük tahmin sorununu giderdi (tahmin ortalaması 52,4 → 54,5; gerçek 54,7). Ancak MAE'yi neredeyse değiştirmedi; çünkü baseline hatasının büyük kısmı seviye sapmasından değil, tek günlük gürültüden geliyor.
- Benzer günlerin (aynı ay ve aynı haftanın günü) ortalamasını almak bu gürültünün büyük kısmını sildi: MAE 8,31 → 6,41.
- **Debug notu:** Haftanın günü özelliği başta ayın günü olarak hesaplanmıştı ve MAE 10,74'e çıktı. Satır sayıları, eksik değerler ve örnek satırlar kontrol edilerek sorun bulundu; düzeltilince MAE 6,41'e indi.
- Stok politikası biraz temkinli (%95 hedefe karşı %98,3). Muhtemel sebep: hata, satışların yüksek olduğu yaz aylarından tahmin edildi; test döneminde ise satış hacmi daha düşük.

## Kısıtlar ve sonraki adımlar

- Güvenlik stoğunu mevsime göre ayarlamak; böylece hizmet seviyesi hedefe yaklaşır ve fazla stok azalır.
- Gecikme (lag) ve hareketli pencere özellikleriyle gradient boosting (örneğin LightGBM) denemek.
- Tahmin doğruluğunu ürün hacmine göre (hızlı ve yavaş satan ürünler) karşılaştırmak.
- Farklı hizmet seviyesi hedefleri için stok seviyelerini gösteren basit bir what-if görünümü eklemek.

## Nasıl çalıştırılır

1. Yukarıdaki Kaggle bağlantısından `train.csv` dosyasını indirin.
2. `demand_forecasting.ipynb` dosyasını Google Colab'da veya Jupyter'da açın.
3. `train.csv` dosyasını yükleyip tüm hücreleri çalıştırın.

**Kütüphaneler:** pandas, numpy, matplotlib, scikit-learn

## AI destekli çalışma süreci

Bu proje, Claude bir AI eşli programlama asistanı olarak kullanılarak geliştirildi. Kod ve açıklama taslaklarını AI hazırladı; ben her adımı çalıştırdım, çıktıları beklentilerle karşılaştırdım ve mantıklı görünmeyen sonuçları sorguladım. Yukarıdaki ayın günü hatası buna bir örnek: Sonuç yanlış göründü, tanı kontrolleri çalıştırdım, kök nedeni buldum ve devam etmeden önce düzelttim.

## Yazar

**Miray Hilesiz** — Yönetim Bilişim Sistemleri Yüksek Lisans, Haliç Üniversitesi
[LinkedIn](https://linkedin.com/in/mirayhilesiz) · [GitHub](https://github.com/hilesizmiray-cmyk) · [Web sitesi](https://mirayhilesiz.com)
