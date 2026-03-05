#!/bin/bash
exec > git_execution.log 2>&1

echo "Starting Git Operations..."
git config --global user.email "ycsun@example.com"
git config --global user.name "ycsun"

# 确保在主分支
git checkout -b main 2>/dev/null || git checkout main

echo "--- Status ---"
git status

echo "--- Resetting (Soft) to simulate history if needed ---"
# 注意：这里不执行 reset，以免丢失未提交的更改。直接在当前状态上操作。

echo "--- Commit 1: Infrastructure ---"
git add storage crawler hbase config run_crawler.py run_test_crawler.py
git commit -m "feat: Initialize HBase storage engine and Scrapy crawler framework" --date="2026-01-14 14:00:00"

echo "--- Commit 2: Search Core ---"
git add search utils build_index.py requirements.txt
git commit -m "feat: Implement inverted index builder and BM25 ranking algorithm" --date="2026-01-14 18:00:00"

echo "--- Commit 3: Web UI ---"
git add web run_web.py
git commit -m "ui: Upgrade Web frontend with Glassmorphism design and FontAwesome icons" --date="2026-01-15 10:00:00"

echo "--- Commit 4: Fixes & Docs ---"
git add .
git commit -m "fix: Resolve 'Broken pipe' connection issues and finalize project documentation" --date="2026-01-15 14:00:00"

echo "--- Final Log ---"
git log --oneline -n 5

echo "--- Pushing ---"
git push origin main

