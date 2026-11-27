# Konsolidasi Pipeline S-Box Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Satu pipeline aktif untuk S-Box penelitian, tanpa generator chaotic atau AES-affine.

**Architecture:** `src/dataset_builder.py` menjadi pintu tetap untuk dataset permutasi. Ketiga trainer berbagi dataset, schema fitur, serta split keluarga; engine rekonstruksi memakai model terlatih. AES/K44 dan Mandrill tetap terpisah sebagai baseline dan evaluasi.

**Tech Stack:** Python, NumPy, scikit-learn, XGBoost, pytest.

**Spec:** Issue #17 dan persetujuan konsolidasi dalam percakapan; integrasi lintas issue #16 dan #18.

## Global Constraints

- Kerjakan di `feat/v2-yafi-issue-17`, bukan `main`.
- Jangan ubah rumus skor atau operasi GF(2⁸) untuk AES.
- Jangan hapus API lama sebelum seluruh pemanggil aktif pindah. Riwayat Git menyimpan implementasi lama setelah penghapusan.
- Hasil legacy tidak boleh dilabeli sebagai hasil rekonstruksi baru.

## Tahap 1 — Konsolidasi dataset (#17)

- [ ] Pindahkan tes dataset ke import `src.dataset_builder.build_dataset(n_families, family_size, seed, evaluate)`. Tegaskan 160 baris unik untuk 20 × 8, induk/anak satu split, dan hasil seed 42 identik. Tambah tes evaluator dengan skor non-finite harus ditolak.
- [ ] Jalankan `.venv/bin/python -m pytest -q tests/test_dataset_v2.py`; tes kontrak baru harus gagal karena API belum ada.
- [ ] Pindahkan `build_dataset`, `extract_features`, `FEATURE_NAMES`, dan `FEATURE_SCHEMA_VERSION` dari `src/dataset_v2.py` ke `src/dataset_builder.py`. Hapus modul duplikat setelah impor pindah. Pertahankan API legacy sementara karena RF, MLP, dan XGBoost masih mengimpor simbol lama. Generator mutasi tetap di `src/permutation_v2.py`.
- [ ] Jalankan tes terarah dan suite penuh. Commit khusus konsolidasi dataset bila lulus.

## Tahap 2 — Migrasi RF dan engine (#17)

- [ ] Pin kontrak input RF: baris dataset baru, `FEATURE_NAMES` sebagai urutan kolom, train untuk fitting, validation untuk tuning, test hanya untuk evaluasi. Tes harus menangkap fitur tertukar dan kebocoran keluarga.
- [ ] Tambah engine yang menerima `predict`, evaluator dan anggaran eksplisit. Tes perubahan ranking mengubah shortlist, baseline Top-K acak mematuhi anggaran yang sama, global best tidak turun, dan kandidat akhir tetap bijektif.
- [ ] Profilkan dataset dan smoke run dengan evaluator yang disepakati #16; simpan artefak baru terpisah sampai dashboard mendukung schema baru.

## Tahap 3 — Penghapusan jalur lama (setelah #16/#18)

- [ ] Pindahkan XGBoost dan MLP ke kontrak yang sama. Audit `app.py`, `src/web_helpers.py`, notebook, README, dan dependensi silang trainer.
- [ ] Tulis tes smoke tiga trainer dan dashboard yang membaca artefak aktual. Tes AES/K44 dan round-trip Mandrill harus tetap lulus.
- [ ] Hapus generator kandidat chaotic/AES-affine lama dari jalur aktif setelah tidak ada pemakai. Jangan hapus pembentuk baseline, operasi AES, atau artefak yang masih dibaca dashboard tanpa pengganti yang jelas.
- [ ] Jalankan `.venv/bin/python -m pytest -q`, cek import tersisa dan `git diff --check`. Review sebelum PR; jangan merge atau push langsung ke `main`.
