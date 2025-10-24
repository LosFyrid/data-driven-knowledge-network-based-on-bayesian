# Aetherium Documentation

Welcome to the Aetherium project documentation. This guide will help you navigate the documentation structure and find the information you need.

---

## 🚀 Quick Start

**New to the project?**
1. Read [Design Philosophy](00_guides/design_philosophy.md) - Understand the core vision
2. Review [CLAUDE.md](../CLAUDE.md) - Get project overview and setup
3. Check [Latest Dev Log](03_dev_logs/) - See current development status

**Want to use existing code?**
- Browse [Implemented Systems](01_implemented/) for usage documentation
- Check [src/rules/base.py](../src/rules/base.py) for RuleClass framework

**Want to contribute?**
1. Read [Latest Dev Log](03_dev_logs/README.md) - Understand current progress
2. Check [Planning Docs](02_planning/) - See upcoming features design
3. Review [CLAUDE.md](../CLAUDE.md) - Setup development environment

---

## 📚 Documentation Structure

### 📖 [00_guides/](00_guides/) - Architectural Guides
**Purpose**: Timeless, foundational documents that explain core concepts and design philosophy.

**Contents**:
- [design_philosophy.md](00_guides/design_philosophy.md) - Core vision and dual-loop architecture

**Characteristics**:
- ✅ Stable and rarely changed
- ✅ High-level conceptual
- ✅ Reference material for architectural decisions

---

### ✅ [01_implemented/](01_implemented/) - Completed Systems
**Purpose**: Documentation for features that have been implemented and are ready to use.

**Contents**:
- [diagnostics.md](01_implemented/diagnostics.md) - Diagnostic tools documentation
- More to come as features are completed...

**Characteristics**:
- ✅ How-to guides and API references
- ✅ Updated when implementation changes
- ✅ Practical usage examples

---

### 🏗️ [02_planning/](02_planning/) - Design Specifications
**Purpose**: Detailed design documents for features currently being developed or planned.

**Contents**:
- [meta_model_design.md](02_planning/meta_model_design.md) - SuperBN (Hybrid Bayesian Network) design 🚧

**Characteristics**:
- ✅ Technical specifications and architecture decisions
- ✅ Implementation strategies and algorithms
- ✅ Moves to `01_implemented/` when feature is complete

---

### 📅 [03_dev_logs/](03_dev_logs/) - Development Logs
**Purpose**: Chronological record of development progress, decisions, and context.

**Contents**: See [Dev Logs Index](03_dev_logs/README.md)

**Characteristics**:
- ✅ Time-stamped with creation/update dates
- ✅ Status-tracked (🚧 Active | ✅ Archived | ⏸️ Paused)
- ✅ Numbered sequentially (001, 002, 003...)
- ✅ Contains "Current Status" and "Next Actions"

**Format**:
```
NNN_YYYY-MM-DD_description.md
Example: 003_2025-10-24_meta_model_kickoff.md
```

---

## 🎯 Status Indicators

| Emoji | Status | Meaning |
|-------|--------|---------|
| 🚧 | Active | Currently being worked on |
| ✅ | Archived | Completed and archived for reference |
| ⏸️ | Paused | Temporarily on hold |
| 🔄 | Updated | Recently updated (last 7 days) |
| 📌 | Pinned | Important reference document |

---

## 🔗 Key Documents Quick Links

### Essential Reading
- [Design Philosophy](00_guides/design_philosophy.md) - Why Aetherium exists
- [CLAUDE.md](../CLAUDE.md) - Project overview and development setup
- [Latest Dev Log](03_dev_logs/README.md) - Current development status

### Current Development Focus
- [Meta Model Design](02_planning/meta_model_design.md) - SuperBN implementation plan
- [Dev Log 003](03_dev_logs/003_2025-10-24_meta_model_kickoff.md) - Current work session

### Implementation Reference
- [RuleClass System](01_implemented/) - Inner loop documentation (coming soon)
- [Diagnostics Tools](01_implemented/diagnostics.md) - Debugging and validation

---

## 🗂️ Documentation Workflow

```
Planning Phase
    ↓
02_planning/feature_design.md (Design spec written)
    ↓
03_dev_logs/NNN_..._feature_dev.md (Development starts)
    ↓
src/module/ (Code implemented)
    ↓
01_implemented/feature_usage.md (Documentation written)
    ↓
03_dev_logs/NNN_..._feature_dev.md → Status: ✅ Archived
```

---

## 📞 Need Help?

- **Can't find what you need?** Check [03_dev_logs/README.md](03_dev_logs/README.md) for the latest updates
- **Want to understand a decision?** Look for related dev logs or design docs
- **Confused about architecture?** Start with [design_philosophy.md](00_guides/design_philosophy.md)

---

**Last Updated**: 2025-10-24
**Maintained By**: Aetherium Development Team
