# Changelog - FractalGenesis

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- PROJECT_PLAN.md - Comprehensive project planning document
- test_system.py - Comprehensive test and verification system
- renderers/verification.py - Automated verification for fractal renders
- renderers/unified.py - Unified renderer interface abstraction
- renderers/mandelbulber/renderer.py - Fixed Mandelbulber 2.34+ support

### Architecture
- Multiple renderer support: Python 3D, Mandelbulber, Flam3
- Evolution framework design
- AI integration points for preference learning

---

## [0.1.0] - 2026-02-16

### Added
- Initial project structure
- Python 3D fractal renderer (Numba JIT)
- Mandelbulber CLI integration
- Flam3 renderer for fractal flames
- Genetic algorithm framework
- Preference learner AI model
- Visual evolution GUI
- Selection interface for evolution

### Known Issues
- Some renderers need performance optimization
- UI requires display for interactive selection
- Evolution engine needs integration with renderers

---

## Project Context

**Created by:** minimax-m2.5 (opencode AI)  
**Date:** 2026-02-16  
**Purpose:** AI-driven fractal art and music visualization pipeline