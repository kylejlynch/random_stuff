#!/bin/bash

echo "====================================="
echo "GitHub Update Script for Clustering"
echo "====================================="
echo

echo "Step 1: Checking git status..."
git status

echo
echo "Step 2: Adding all new and modified files..."
git add -A

echo
echo "Step 3: Checking what will be committed..."
git status

echo
echo "Step 4: Creating commit with comprehensive updates..."
git commit -m "Major update: Add parameter tuning and alert coverage analysis

Features added:
- Comprehensive grid search for optimal clustering parameters
- Alert coverage analysis to find weak pockets in fraud detection
- Work environment compatibility (sklearn.externals fixes)
- Interactive visualizations for parameter tuning and alert analysis
- Auto-tuned pipeline with intelligent parameter selection
- Enhanced fraud pipeline with integrated alert coverage
- Examples and documentation for all new features

Technical improvements:
- Package version compatibility for restricted environments
- HDBSCAN v0.8.20 compatibility (removed cluster_selection_epsilon)
- File cleanup and dependency consolidation
- Comprehensive error handling and logging
- Performance optimizations for different dataset sizes

🤖 Generated with Claude Code
https://claude.ai/code

Co-Authored-By: Claude <noreply@anthropic.com>"

echo
echo "Step 5: Pushing to GitHub..."
git push origin master

echo
echo "====================================="
echo "GitHub update complete!"
echo "====================================="
echo
echo "New features available:"
echo "- Parameter tuning system"
echo "- Alert coverage analysis"
echo "- Work environment compatibility"
echo "- Enhanced visualizations"
echo
echo "Check the README.md and CHANGELOG.md for details."