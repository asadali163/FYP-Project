#%%
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import List, Optional
import numpy as np
from PIL import Image
import cv2 as cv
import io
import tensorflow as tf
import uvicorn
from pymongo import MongoClient
from pydantic import BaseModel, EmailStr
import bcrypt
import os
import uuid
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import DuplicateKeyError
import re
import datetime
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


#%%

client = MongoClient('mongodb://localhost:27017/')
db = client['model_pnemonia']
users_collection = db['users']
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



#%%

# Define User model
class User(BaseModel):
    FullName: str
    email: EmailStr
    password: str
    Predictions: Optional[List[dict]] = []
    
    
# Define User model
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Define a request model for resetting passwords
class ResetPasswordRequest(BaseModel):
    email: EmailStr
    new_password: str

    
#%%

# Load the model
# Check different .h5 files for best result.

def load_classification_model(weights_path):
    # Define the architecture of the classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    classification_layer = Dense(3, activation='softmax')(x)

    # Create a new model that includes only the classification layers
    classification_model = Model(inputs=base_model.input, outputs=classification_layer)

    # Load the weights for the classification layers
    classification_model.load_weights(weights_path)

    return classification_model

# Example usage
weights_path = r'F:\FYP\FYP Versions\20022024\vgg16_f5_model.h5'
model = load_classification_model(weights_path)

# Preprocess image function
def preprocess_image(image):
    # Preprocess the image as needed (resize, normalize, etc.)
    #print("Image shape is: ", np.array(image).shape)
    image = cv.resize(image, (256, 256))
    #image = image.resize((256, 256,3))  # Assuming 224x224 is the input size for your model
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image.reshape(1, 256, 256, 3)
    return image

#%%

def serialize_user(user_doc):
    user_dict = {
        "id": str(user_doc["_id"]),  # Convert ObjectId to string
        "FullName": user_doc["FullName"],
        "email": user_doc["email"],
        "password": user_doc["password"],
        "Predictions": user_doc["Predictions"]
    }
    return user_dict

#%%
app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
#%% Middleware

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL in production for security
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

#%%

@app.get("/")
def index():
    return {"message":"Hello world!"}

#%%
# Prediction endpoint
@app.post("/predict/")
async def predict(email: str = Form(...), file: UploadFile = File(...)):
    # Fetch user document based on email
    user_doc = users_collection.find_one({"email": email})
    
    if user_doc:
        # Read file contents
        contents = await file.read()
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR) 
        image = preprocess_image(image)
        #image = cv.resize(image, (256,256))
        ##image = np.array(image)
        #image = image.reshape(1, 256, 256, 3)
        
        #print("Image shape is :", np.array(image).shape)
        
        # Generate a unique filename
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        
        # Save the file to the uploads folder
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        
        #model = ret_model()
        # Add prediction result to user document
        prediction = model.predict(image)
        print("Prediction is :",prediction)
        prediction = np.squeeze(prediction)
        print(prediction.shape)
        
        current_date = datetime.datetime.now()
        
        prediction_data = {
           "fileUrl": file_path,
           "result": prediction.tolist(),
           "date": current_date.strftime("%Y-%m-%d %H:%M:%S")  # Format the date as desired
        }
        
        # Append prediction data to user document
        user_doc['Predictions'].append(prediction_data)
        
        # Update user document in the database
        users_collection.update_one({"email": email}, {"$set": user_doc})
        
        return {
            "message": "Prediction saved successfully",
            "prediction": prediction.tolist()
        }
    else:
        return {"message": "User not found"}
#%%

# Regular expression for password validation (at least 6 characters long)
password_regex = re.compile(r'^(?=.*[A-Za-z0-9])(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')


@app.post("/user")
async def add_user(user: User):
    # Check if email already exists
    if users_collection.find_one({'email': user.email}):
        raise HTTPException(status_code=400, detail="Email already exists")

    # Validate password
    if not password_regex.match(user.password):
        raise HTTPException(status_code=400, detail="Password must be alphanumeric and at least 8 characters long with 1 special character ")

    # Encrypt the password before storing it
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    user.password = hashed_password.decode('utf-8')
    
    # Insert user into database
    try:
        result = users_collection.insert_one(user.dict())
        return {
            "message": "User added successfully",
            "user_id": str(result.inserted_id),
            "user": user
        }
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username or email already exists")


#%%
@app.post("/login")
async def login(user: UserLogin):
    user_doc = users_collection.find_one({"email": user.email})
    user_doc = serialize_user(user_doc)
    if user_doc and bcrypt.checkpw(user.password.encode('utf-8'), user_doc['password'].encode('utf-8')):
        del user_doc['password']  # Remove password from response
        return {"user": user_doc}
    else:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    

#%% Forgot password.
@app.post("/reset_password/")
async def reset_password(request: ResetPasswordRequest):
    # Fetch user document based on email
    user_doc = users_collection.find_one({"email": request.email})
    
    if user_doc:
        # Validate new password
        if not password_regex.match(request.new_password):
            print("I am here in password reset")
            raise HTTPException(status_code=400, detail="New password must be alphanumeric and at least 8 characters long with 1 special character ")
        
        # Encrypt the new password
        hashed_password = bcrypt.hashpw(request.new_password.encode('utf-8'), bcrypt.gensalt())
        new_hashed_password = hashed_password.decode('utf-8')
        
        # Update user's password in the database
        users_collection.update_one({"email": request.email}, {"$set": {"password": new_hashed_password}})
        
        return {"message": "Password reset successfully"}
    else:
        raise HTTPException(status_code=404, detail="User not found")


    
#%%    # for testing purpose
@app.get("/users")
async def get_users():
    users = list(users_collection.find({}, {"_id": 0, "password": 0}))
    return {"users": users}

#%%
@app.get("/predicted_results/")
async def get_predicted_results(email: str):
    user_doc = users_collection.find_one({"email": email})
    if user_doc:
        predictions = user_doc.get("Predictions", [])
        return {"predictions": predictions}
    else:
        raise HTTPException(status_code=404, detail="User not found")


#%%
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
