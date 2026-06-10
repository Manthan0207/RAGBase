from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ShouldRetrieve(BaseModel) :
    retrieve : bool = Field(description="Whether to retrieve from vector store or not")


class LTMData(BaseModel) :
    category : Literal["details" , "preferences" , "goals"] = Field(...,description="Category of the information.",examples=["preferences"])
    text : str = Field(... , description="The important information to be stored in LTM.")
    key: str = Field(..., description="Short snake_case identifier for this fact. e.g. 'name', 'location', 'preferred_language', 'current_project'. Must be consistent — same concept always gets same key.")

class LTMStorageNodeOutput(BaseModel) :
    should_store : bool = Field(... , description="Whether to store any of the important information in long-term memory.")
    ltm_data : Optional[List[LTMData]] = Field(None , description="The important information to be stored in LTM along with its category.")

class SummarizeAndSaveToLTM(BaseModel) :
    summary : str = Field(... , description="The updated summary of the conversation so far.")
    ltm_output : Optional[LTMStorageNodeOutput] = Field(None , description="The important information to be stored in LTM along with its category.")
     