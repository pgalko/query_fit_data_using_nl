from pydantic import BaseModel
from typing import Optional


class UserAuth(BaseModel):
    username: str 
    password: str 
    
class UserOut(BaseModel):
    id: Optional[int] = None
    username: str
    disabled: Optional[bool] = False

class TokenSchema(BaseModel):
    access_token: str
    refresh_token: str

class TokenPayload(BaseModel):
    sub: str = None
    exp: int = None
