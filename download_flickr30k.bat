@echo off
REM 下载完整的 Flickr30K 数据集

echo ========================================
echo 开始下载 Flickr30K 完整数据集
echo ========================================
echo.

python tools\download_flickr30k.py --split all --max_workers 10

echo.
echo ========================================
echo 下载完成!
echo ========================================
pause
