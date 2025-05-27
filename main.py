from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chat, files
from logs.logger import setup_logging
# 1. Initialize FastAPI app
app = FastAPI()

setup_logging()



app.add_middleware(
    CORSMiddleware,
    # Allow requests from any origin (you can restrict this to specific domains)
    allow_origins=["*"],
    # Allow credentials (cookies, authorization headers)
    allow_credentials=True,
    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    # Allow all headers
    allow_headers=["*"],
)


app.include_router(chat.router ,tags=["chat"])
app.include_router(files.router,tags=["files"])



# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
