@echo off
setlocal enabledelayedexpansion

echo 1) Menampilkan status git sebelum add...
git status

echo 2) Menambahkan semua file ke staging area...
git add .

echo 3) Menampilkan status git setelah add...
git status

REM Ambil seluruh parameter sebagai pesan commit
set commitmsg=%*

REM Cek apakah pesan commit kosong, jika kosong tanya input pesan commit
if "!commitmsg!"=="" (
    set /p commitmsg=Masukkan pesan commit: 
)

REM Jalankan commit dengan pesan dalam tanda petik agar diinterpretasi sebagai satu string
git commit -am "!commitmsg!"

git push -u origin main

pause
