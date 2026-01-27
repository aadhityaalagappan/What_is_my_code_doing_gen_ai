import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Open AI API Key not found")

class CodeSolution(BaseModel): #Response
    explanation: str = Field("Provide the detailed explanantion of the code based on the level chosen")
    timecomplexity: str = Field("Provide the time complexity of the code")
    commonerrors: str = Field("Give some common errors that users might encounter")
    
class ExplainRequest(BaseModel): #Request
    code: str = Field(..., min_length=1, description="Code snippet to explain")
    level: str = Field(..., description="beginner | intermediate | advanced")    
    
parser = PydanticOutputParser(pydantic_object = CodeSolution)   # Format Output

#Langchain output
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise assistant who gives the putput following formatting instructions strictly" ),
    ("user", """
        User level: {level}
        
        Explain the following code clearly based on the user level. Give step by step and line by line explanation with example iteration
        
        Provide the time and space complexity and how it is calculated.
        
        Mention the common mistakes users commit in this problem
        
        Code: {code}
        
        {format_instructions}
     
     
     """.strip())
])

# Initialised OpenAI Client
llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0.2, api_key = API_KEY)

#Initilaising the chain which passes series of steps

chain = prompt | llm | parser



@app.get("/")
def test():
    return {"message": "API is working"}

@app.post("/explain-code")
def explain_code(req: ExplainRequest):
    if not req:
        raise HTTPException(status_code=400, detail="Code not given")
    
    code = req.code.strip()
    level = req.level.strip().lower()
    
    if level not in {"beginner", "intermediate", "advanced"}:
        raise HTTPException(status_code=400, detail="level must be beginner/intermediate/advanced")
    
    try:
        result = chain.invoke({
            "code": code,
            "level": level,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Failed to generate response due to : {str(ex)}")
        
    
    
    
    
   
    
    
    
    
    
        



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


