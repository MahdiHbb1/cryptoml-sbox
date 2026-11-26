# Issue #17 — Pilot RF vs Top-K acak

Dataset: 150 keluarga × 10 baris, seed 42. RF: 300 pohon, seed 42, fit hanya pada split train. Skor memakai `evaluate_sbox` dan `compute_composite_score` yang ada.

Setiap pasangan memakai titik awal yang sama, mutasi single swap, 50 iterasi, pool 20, dan Top-K 3. Semua run memakai 151 evaluasi penuh, termasuk evaluasi awal.

| Seed | Awal | RF akhir | Acak akhir | RF − acak |
| --- | ---: | ---: | ---: | ---: |
| 100 | 75.903 | 87.429 | 89.413 | -1.984 |
| 101 | 75.875 | 91.088 | 88.062 | +3.026 |
| 102 | 76.881 | 87.567 | 87.563 | +0.004 |
| 103 | 84.471 | 93.430 | 89.089 | +4.341 |
| 104 | 77.534 | 82.721 | 87.510 | -4.789 |

Rata-rata selisih +0.120; simpangan baku sampel 3.701. Pilot lima seed belum menunjukkan keunggulan RF yang konsisten. Anggaran disamakan berdasarkan evaluasi penuh, bukan waktu total. Ini membandingkan hasil akhir pencarian, bukan membuktikan keamanan.

## Diagnostik ranking per pool

20 pool dari seed 200–219, masing-masing 20 mutasi single swap unik dari permutasi awal. Rata-rata Spearman RF: 0.00334 (20 pool valid). Rata-rata selisih skor kandidat terbaik di seluruh pool terhadap kandidat terbaik di shortlist Top-3: 2.57549 untuk RF, 2.10392 untuk ekspektasi pemilihan acak tanpa pengembalian. Lebih kecil lebih baik. Diagnostik ini hanya menguji pool dari state awal, bukan pool selama rekonstruksi.

Ekspor `save_grouped_rf` menyimpan `rf.joblib` dan `manifest.json`. Uji lokal memuat ulang model dan memverifikasi prediksi identik untuk semua 230 sampel test. Muat hanya model Joblib dari sumber tepercaya.

## Jalankan ulang

```bash
mkdir -p /tmp/opencode
python -m src.train_rf_v2 /tmp/opencode/issue17-model
python -m src.ml_reconstruction /tmp/opencode/issue17-rf.json --model /tmp/opencode/issue17-model --seed 100
python -m src.ml_reconstruction /tmp/opencode/issue17-random.json --seed 100
```

Gunakan path keluaran baru setiap kali. JSON berisi S-Box awal, akhir dan invers, histori, seed, serta jumlah evaluasi penuh aktual. CLI sekarang memakai evaluator dan skor dari checkout lokal; hasilnya bukan training final yang bergantung pada kontrak #16.
