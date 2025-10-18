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
