# ONNX Autodoc - Brainlift Documentation

Daily learning logs as required by the Uncharted Territory Challenge.

**Related Documents:**
- [PRD.md](PRD.md) - Product requirements and specifications
- [BACKLOG.md](BACKLOG.md) - Epic/Story/Task tracking
- [THE UNCHARTED TERRITORY CHALLENGE.md](THE%20UNCHARTED%20TERRITORY%20CHALLENGE.md) - Challenge requirements

---

## Log Template

Copy this template for each day:

```markdown
## Day N - [Date]

### Goals for Today
- [ ] Goal 1
- [ ] Goal 2

### What I Accomplished
- Accomplishment 1
- Accomplishment 2

### AI Prompts Used

**Prompt 1:**
> "..."

*Response summary:* ...

**Prompt 2:**
> "..."

*Response summary:* ...

### Learning Breakthroughs
- What new concept did you understand today?
- What was confusing that became clear?

### Technical Decisions

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| ... | ... | ... |

### What Changed
- **Files modified:**
- **New code added:**
- **Bugs fixed:**

### Challenges and Solutions

| Challenge | How I Solved It |
|-----------|----------------|
| ... | ... |

### Tomorrow's Focus
- Priority 1:
- Priority 2:

### Time Spent
- Total: X hours
- Breakdown: Research (Xh), Coding (Xh), Debugging (Xh), Documentation (Xh)
```

---

## Daily Logs

---

## Day 1 - December 1, 2025

### Goals for Today
- [x] Fork ONNX Runtime repository
- [x] Set up local development environment
- [ ] Successfully build ONNX Runtime from source (in progress - CUDA kernels compiling)
- [x] Create project documentation structure

### What I Accomplished
- Forked microsoft/onnxruntime
- Set up CMake build with CUDA support
- Created consolidated documentation:
  - PRD.md (product requirements)
  - BACKLOG.md (task tracking)
  - BRAINLIFT.md (this file)
  - Architecture.md (system design)
  - README.md (quick start guide)

### AI Prompts Used

**Prompt 1:**
> "Check out /docs/marcu and its contents in this repo. You'll see the project requirements... We should combine everything into a single PRD.md file, build out a comprehensive to-do list, and make readme + Architecture.md files as well."

*Response summary:* AI consolidated 4 existing documents into a unified documentation structure with PRD, README, and Architecture files. Created Jira-style backlog with 9 Epics, 27 Stories, and 100+ Tasks.

**Prompt 2:**
> "Is that PRD going to be too large to manage from a context window perspective?"

*Response summary:* AI recommended splitting the PRD into 3 files (PRD.md ~600 lines for specs, BACKLOG.md ~350 lines for tasks, BRAINLIFT.md for logs) to optimize context window usage during implementation.

**Prompt 3:**
> "What would be the pros/cons of slapping a frontend on this thing vs going full CLI?"

*Response summary:* AI provided trade-off analysis recommending CLI-first approach with optional static HTML reports as stretch goal. Full web UI deferred to post-gauntlet.

### Learning Breakthroughs
- ONNX Runtime builds are extremely slow when including CUDA provider (flash attention kernels are compiled for multiple GPU architectures)
- The codebase is massive (~2M+ lines of code) - need to be strategic about which parts to explore

### Technical Decisions

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Python-first implementation | Faster dev time, easier ONNX integration | C++ core with Python bindings |
| CLI-first, no web UI for MVP | Fits 7-day timeline, matches ORT ecosystem | Flask/React dashboard |
| Split PRD into 3 files | Optimize context window for AI assistance | Single large PRD |

### What Changed
- **Files created:**
  - `docs/marcu/PRD.md`
  - `docs/marcu/BACKLOG.md`
  - `docs/marcu/BRAINLIFT.md`
  - `docs/marcu/Architecture.md`
  - `docs/marcu/README.md`
- **Files removed:**
  - `docs/marcu/onnx_autodoc_starter_pack.md` (merged into PRD)
  - `docs/marcu/autodoc_visualization_extension.md` (merged into PRD)

### Challenges and Solutions

| Challenge | How I Solved It |
|-----------|----------------|
| CUDA kernel compilation taking forever | Let it run in background, focus on documentation |
| Large PRD consuming context window | Split into PRD + BACKLOG + BRAINLIFT |

### Tomorrow's Focus
- Priority 1: Complete ORT build and verify with tests
- Priority 2: Explore existing Python tools in the codebase
- Priority 3: Create autodoc module scaffolding

### Time Spent
- Total: ~3 hours
- Breakdown: Research (0.5h), Documentation (2h), Build setup (0.5h)

---

*Add new days above this line*
