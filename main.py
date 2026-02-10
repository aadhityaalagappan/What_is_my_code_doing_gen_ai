import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Open AI API Key not found")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://what-is-my-code-doing-gen-ai-zenm.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class CodeSolution(BaseModel):
    explanation: str
    timecomplexity: str
    spacecomplexity: str
    commonerrors: str

class ExplainRequest(BaseModel):
    code: str = Field(..., min_length=1)
    level: str = Field(..., description="beginner | intermediate | advanced")

llm = ChatOpenAI(
    model='gpt-4o-mini', 
    temperature=0.0,
    api_key=API_KEY,
    max_tokens=8000,
    model_kwargs={"response_format": {"type": "json_object"}}
)

@app.get("/test")
def test():
    return {"message": "API working"}

@app.post("/explain-code")
async def explain_code(req: ExplainRequest):
    if not req:
        raise HTTPException(status_code=400, detail="No code provided")
    
    code = req.code.strip()
    level = req.level.strip().lower()
    
    if level not in {"beginner", "intermediate", "advanced"}:
        raise HTTPException(status_code=400, detail="Invalid level")
    
    prompt = f"""Analyze this code and output JSON with 4 fields: explanation, timecomplexity, spacecomplexity, commonerrors.

CODE:
```
{code}
```

LEVEL: {level}

JSON FORMAT:
{{
  "explanation": "markdown string",
  "timecomplexity": "markdown string",
  "spacecomplexity": "markdown string",
  "commonerrors": "markdown string"
}}

EXPLANATION FIELD MUST HAVE:

## What This Code Does
[2 sentences about what it solves]

## Algorithm Used
**Algorithm:** [type]
**Strategy:** [approach]

## Code Breakdown
**Init:** [variables]
**Logic:** [main loop/recursion]
**Return:** [what returns]

## Example Walkthrough

### Input: `concrete small value like s="abc", t="ab" or arr=[2,1,5] or n=3`

CRITICAL EXECUTION RULES:
You MUST simulate execution on a CONCRETE small input and show ALL steps UNTIL termination.

Define "step" precisely based on the pattern:

A) FOR LOOPS:
- Each step = one loop iteration (one update of the loop index).
- Continue until loop ends.

B) WHILE LOOPS:
- Each step = one evaluation of the loop body.
- Continue until the loop condition becomes false.

C) RECURSION / DFS:
- Each step = one function call entry OR one return (use entry steps only if too long).
- Show the call stack at each step.
- Continue until all recursive calls complete.

D) BFS / GRAPH TRAVERSAL:
- Each step = one queue pop (dequeue).
- At each step, show:
  - current node
  - queue contents AFTER pushing neighbors
  - visited set
  - parent map (if building path/tree)

E) TREE TRAVERSAL:
- Each step = visiting a node (preorder visit), or one queue pop for level-order.
- Show the tree state using either:
  - ASCII tree, or
  - Level arrays (recommended)

GRAPH/TREE STATE VISUALIZATION (REQUIRED WHEN BFS/DFS/TREE/GRAPH):

**Initial:**
```
variables = initial values
```

### Iteration 1
**Before:** i=0, var=value
**Code:** `actual line with values substituted`
**After:** updated values
**DP Table (if DP):** Show current state
**What:** [brief explanation]

### Iteration 2
[Same format, show DP table if DP]

### Iteration 3
[Same format]

### Iteration 4
[Same format]
.
.
.
.
.
.
.
.
### Iteration n
[Same format]

Display until nth iteration, don't stop only until 4 iterations

**Final:** result = answer

## For Tree and Graph type problems solvable by Depth first search and Breadth First search, construct tree at each iteration

## Key Insights
- Point 1
- Point 2

TIME COMPLEXITY FIELD:

## Time Complexity: O(?)

### Big-O Quick Intro
O(1)=constant, O(n)=linear, O(n²)=quadratic, O(log n)=logarithmic, O(2^n)=exponential

### This Code's Complexity
[Simple analogy for this specific O()]

### Line-by-Line Analysis
**Line X:** `actual code line from THIS code`
- Runs: N times
**Line Y:** `nested loop from THIS code`
- Runs: M times per outer
**Total:** N × M = O(N·M)

### Real Numbers
n=10, m=5: 50 ops
n=100, m=50: 5000 ops
n=1000, m=500: 500,000 ops

### Why O(?)
[Math showing how we got this]

SPACE COMPLEXITY FIELD:

## Space Complexity: O(?)

### Memory Breakdown
**Stack:** O(?) - [recursion depth or O(1)]
**DP Table:** `int[][] dp = new int[n+1][m+1]` → O(n·m)
**Variables:** O(1)
**Total:** O(?)

### Real Memory
n=100, m=50: ~20 KB
n=1000, m=500: ~2 MB

COMMON ERRORS FIELD:

## Common Mistakes

### 1. [Error]
**Wrong:** `code`
**Why:** [reason]
**Fix:** `corrected code`

### 2-5. [More errors same format]

CRITICAL RULES:
1. Use concrete input (s="abc" not "string s")
2. Show DP table state AFTER each iteration if DP
3. Analyze THIS code's actual lines for time complexity
4. Calculate real memory from THIS code's data structures
5. Show at least 4 iterations

Output ONLY valid JSON."""

    for attempt in range(3):
        try:
            logger.info(f"Attempt {attempt + 1}/3")
            
            response = llm.invoke(prompt)
            result_text = response.content
            
            logger.info(f"Response length: {len(result_text)} chars")
            
            result_json = json.loads(result_text)
            
            if not all(k in result_json for k in ["explanation", "timecomplexity", "spacecomplexity", "commonerrors"]):
                raise ValueError("Missing fields")
            
            solution = CodeSolution(**result_json)
            logger.info("Success!")
            return solution
            
        except json.JSONDecodeError as je:
            logger.error(f"JSON error: {str(je)}")
            if attempt == 2:
                raise HTTPException(status_code=500, detail="Invalid JSON response")
        except Exception as ex:
            logger.error(f"Error: {str(ex)}")
            if attempt == 2:
                raise HTTPException(status_code=500, detail=str(ex))
