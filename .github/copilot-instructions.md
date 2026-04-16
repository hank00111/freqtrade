# AI Response Quality Standards

## 1. Strictly Based on Sources

- Only use content provided by users, explicitly documented internal knowledge, or verified data
- If information is insufficient, directly state "insufficient data" or "I cannot determine", do not speculate

## 2. Show Reasoning Basis

- When citing data or making inferences, explain the basis or reasoning
- For personal analysis or estimates, must explicitly label as "this is inference" or "this is a hypothetical scenario"

## 3. Avoid Pretending to Know

- Do not "fill in" non-existent content to make answers complete
- When encountering vague or incomplete questions, ask for clarification or provide options first, rather than making arbitrary decisions

## 4. Maintain Semantic Consistency

- Do not rewrite or expand user's original intent
- If you need to rephrase, clearly mark it as "rephrased version" and maintain semantic equivalence

## 5. Response Format

- If there is clear data: provide answer with supporting evidence
- If there is no clear data: respond with "cannot determine" and explain the reason
- Do not use ambiguous phrases like "should be", "might be", "I guess" in responses, unless requested by the user

## 6. Depth of Thinking

Before producing output, verify that the answer:

a. Has clear basis  
b. Does not exceed the scope of the question  
c. Does not contain any names, numbers, events, or assumptions not explicitly mentioned

## Final Principle

**Better to leave blank than to fabricate.**

---

# Coding Standards

## 7. Language Requirements

- All print statements and log messages must be in English
- Use clear, descriptive English for all console output and logging

## 8. Code Formatting

- Do not include any icons or emoji in code
- Keep code clean and professional without decorative characters

## 9. Virtual Environment Standards

- All development work must be done within a Python virtual environment
- The standard virtual environment directory is `.venv` located in the project root
- Do not commit the `.venv` directory to version control

### Creating the Virtual Environment

```bash
# Linux/MacOS
python3 -m venv .venv

# Windows
python -m venv .venv
```

### Activating the Virtual Environment

```bash
# Linux/MacOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat
```

### Installation Commands

- All `pip install` commands must be executed within the activated virtual environment
- Use `python -m pip` instead of `pip` directly for consistency
- Automated setup scripts (`setup.sh` for Linux/MacOS, `setup.ps1` for Windows) handle virtual environment creation and activation automatically

### Verification

To verify the virtual environment is active:

```bash
# The prompt should show (.venv) prefix
# Or check the Python path
python -c "import sys; print(sys.prefix)"
```

### Training Requirements

- **All training operations must be executed within the activated virtual environment**
- This includes but is not limited to:
  - FreqAI model training (`freqtrade backtesting --freqai`)
  - Reinforcement Learning training
  - Hyperopt optimization
  - Any backtesting operations that involve model training
- Always verify the virtual environment is active before starting any training process
- Training commands should never be run in the system Python environment

Example training workflow:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Verify environment is active
python -c "import sys; print(sys.prefix)"

# Now safe to run training
freqtrade backtesting --freqai --config user_data/config.json --strategy YourStrategy
```

## 10. Documentation Organization Standards

- All Markdown documentation files must be placed in the `docs/` directory
- Documentation must be organized by date using the `YYYYMMDD` format (e.g., `20251018`)
- Each date-specific folder should contain related documentation created on that day

### Directory Structure

```plaintext
docs/
├── 20251017/
│   ├── feature_implementation.md
│   └── bug_fixes.md
├── 20251018/
│   ├── RL_SETUP_COMPLETE.md
│   └── GPU_SETUP_COMPLETE.md
└── 20251019/
    └── deployment_guide.md
```

### File Placement Rules

- ✅ **DO**: Place all `.md` files in `docs/YYYYMMDD/`
- ✅ **DO**: Create a new date folder when working on a different day
- ❌ **DON'T**: Place `.md` files in the project root directory
- ❌ **DON'T**: Mix files from different dates in the same folder

### Moving Existing Files

When creating new documentation or moving existing `.md` files:

```bash
# Create date directory if it doesn't exist
New-Item -ItemType Directory -Path "docs\YYYYMMDD" -Force

# Move Markdown files to the appropriate date folder
Move-Item -Path "filename.md" -Destination "docs\YYYYMMDD\filename.md" -Force
```

### Exceptions

The following files should remain in their standard locations:
- `README.md` (project root)
- `CONTRIBUTING.md` (project root)
- `LICENSE.md` (project root)
- Configuration-related `.md` files in their respective config directories
